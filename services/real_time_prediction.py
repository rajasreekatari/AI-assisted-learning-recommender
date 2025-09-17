"""
Real-time Prediction Service with Scalable Architecture
Handles high-throughput, low-latency predictions with microservices architecture
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Async and concurrency
import aiohttp
import aioredis
from asyncio import Queue, Semaphore
import concurrent.futures

# FastAPI and async processing
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Message queuing and streaming
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis

# Model serving
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil

# Load balancing and circuit breaker
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class PredictionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class PredictionRequest:
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    priority: RequestPriority
    timestamp: datetime
    timeout_seconds: int = 30
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class PredictionResponse:
    request_id: str
    prediction: Any
    confidence: float
    latency_ms: float
    model_version: str
    timestamp: datetime
    status: PredictionStatus
    error_message: Optional[str] = None

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class LoadBalancer:
    """Simple round-robin load balancer for model replicas"""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_index = 0
        self.circuit_breakers = {endpoint: CircuitBreaker() for endpoint in endpoints}
    
    def get_next_endpoint(self) -> Optional[str]:
        if not self.endpoints:
            return None
        
        # Find next healthy endpoint
        for _ in range(len(self.endpoints)):
            endpoint = self.endpoints[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.endpoints)
            
            if self.circuit_breakers[endpoint].can_execute():
                return endpoint
        
        return None
    
    def mark_success(self, endpoint: str):
        if endpoint in self.circuit_breakers:
            self.circuit_breakers[endpoint].on_success()
    
    def mark_failure(self, endpoint: str):
        if endpoint in self.circuit_breakers:
            self.circuit_breakers[endpoint].on_failure()

class PredictionCache:
    """Redis-based prediction cache for frequently requested predictions"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, model_id: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key from model ID and input data"""
        input_hash = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        return f"prediction:{model_id}:{input_hash}"
    
    async def get(self, model_id: str, input_data: Dict[str, Any]) -> Optional[PredictionResponse]:
        """Get prediction from cache"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(model_id, input_data)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                response_dict = json.loads(cached_data)
                # Convert timestamp back to datetime
                response_dict['timestamp'] = datetime.fromisoformat(response_dict['timestamp'])
                response_dict['status'] = PredictionStatus(response_dict['status'])
                return PredictionResponse(**response_dict)
            
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    async def set(self, model_id: str, input_data: Dict[str, Any], response: PredictionResponse):
        """Store prediction in cache"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._generate_cache_key(model_id, input_data)
            response_dict = asdict(response)
            # Convert datetime to string for JSON serialization
            response_dict['timestamp'] = response.timestamp.isoformat()
            response_dict['status'] = response.status.value
            
            await self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(response_dict)
            )
            
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

class RealTimePredictionService:
    """
    High-performance real-time prediction service with scalable architecture
    """
    
    def __init__(self, 
                 max_concurrent_requests: int = 100,
                 request_timeout: int = 30,
                 model_endpoints: List[str] = None,
                 kafka_bootstrap_servers: List[str] = None,
                 redis_url: str = "redis://localhost:6379"):
        
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.model_endpoints = model_endpoints or ["http://localhost:8000"]
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or ["localhost:9092"]
        self.redis_url = redis_url
        
        # Initialize components
        self.load_balancer = LoadBalancer(self.model_endpoints)
        self.prediction_cache = PredictionCache(redis_url)
        self.request_queue = Queue()
        self.response_queue = Queue()
        
        # Concurrency control
        self.semaphore = Semaphore(max_concurrent_requests)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # Message queues
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Metrics
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        # Request tracking
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_history: List[PredictionRequest] = []
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.request_counter = Counter(
            'prediction_requests_total',
            'Total prediction requests',
            ['model_id', 'priority', 'status'],
            registry=self.registry
        )
        
        self.latency_histogram = Histogram(
            'prediction_latency_seconds',
            'Prediction latency',
            ['model_id'],
            registry=self.registry
        )
        
        self.queue_size_gauge = Gauge(
            'prediction_queue_size',
            'Current prediction queue size',
            registry=self.registry
        )
        
        self.active_requests_gauge = Gauge(
            'active_prediction_requests',
            'Current active prediction requests',
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'prediction_cache_hit_ratio',
            'Prediction cache hit ratio',
            registry=self.registry
        )
    
    async def initialize(self):
        """Initialize the prediction service"""
        logger.info("Initializing Real-time Prediction Service...")
        
        # Initialize cache
        await self.prediction_cache.initialize()
        
        # Initialize Kafka (if configured)
        await self._initialize_kafka()
        
        # Start background tasks
        asyncio.create_task(self._process_request_queue())
        asyncio.create_task(self._update_metrics())
        
        logger.info("Real-time Prediction Service initialized successfully")
    
    async def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            if self.kafka_bootstrap_servers:
                # Initialize producer
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    retries=3,
                    acks='all'
                )
                
                # Initialize consumer for async processing
                self.kafka_consumer = KafkaConsumer(
                    'prediction-requests',
                    bootstrap_servers=self.kafka_bootstrap_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id='prediction-service'
                )
                
                # Start Kafka consumer task
                asyncio.create_task(self._consume_kafka_messages())
                
                logger.info("Kafka initialized successfully")
                
        except Exception as e:
            logger.warning(f"Kafka initialization failed: {e}")
    
    async def _consume_kafka_messages(self):
        """Consume messages from Kafka for async processing"""
        try:
            for message in self.kafka_consumer:
                request_data = message.value
                request = PredictionRequest(
                    request_id=request_data['request_id'],
                    model_id=request_data['model_id'],
                    input_data=request_data['input_data'],
                    priority=RequestPriority(request_data['priority']),
                    timestamp=datetime.fromisoformat(request_data['timestamp']),
                    timeout_seconds=request_data.get('timeout_seconds', 30),
                    callback_url=request_data.get('callback_url'),
                    metadata=request_data.get('metadata', {})
                )
                
                await self.request_queue.put(request)
                
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
    
    async def predict(self, 
                     model_id: str,
                     input_data: Dict[str, Any],
                     priority: RequestPriority = RequestPriority.NORMAL,
                     use_cache: bool = True,
                     timeout_seconds: int = 30) -> PredictionResponse:
        """Make a real-time prediction"""
        
        request_id = str(uuid.uuid4())
        request = PredictionRequest(
            request_id=request_id,
            model_id=model_id,
            input_data=input_data,
            priority=priority,
            timestamp=datetime.now(timezone.utc),
            timeout_seconds=timeout_seconds
        )
        
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_response = await self.prediction_cache.get(model_id, input_data)
                if cached_response:
                    self.request_counter.labels(
                        model_id=model_id, 
                        priority=priority.value, 
                        status='cache_hit'
                    ).inc()
                    return cached_response
            
            # Add to request queue
            await self.request_queue.put(request)
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(
                    self._wait_for_response(request_id),
                    timeout=timeout_seconds
                )
                
                # Cache successful response
                if use_cache and response.status == PredictionStatus.COMPLETED:
                    await self.prediction_cache.set(model_id, input_data, response)
                
                # Update metrics
                latency = time.time() - start_time
                self.request_counter.labels(
                    model_id=model_id,
                    priority=priority.value,
                    status=response.status.value
                ).inc()
                self.latency_histogram.labels(model_id=model_id).observe(latency)
                
                return response
                
            except asyncio.TimeoutError:
                # Handle timeout
                response = PredictionResponse(
                    request_id=request_id,
                    prediction=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    model_version="unknown",
                    timestamp=datetime.now(timezone.utc),
                    status=PredictionStatus.TIMEOUT,
                    error_message="Request timeout"
                )
                
                self.request_counter.labels(
                    model_id=model_id,
                    priority=priority.value,
                    status='timeout'
                ).inc()
                
                return response
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            response = PredictionResponse(
                request_id=request_id,
                prediction=None,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                model_version="unknown",
                timestamp=datetime.now(timezone.utc),
                status=PredictionStatus.FAILED,
                error_message=str(e)
            )
            
            self.request_counter.labels(
                model_id=model_id,
                priority=priority.value,
                status='error'
            ).inc()
            
            return response
    
    async def _process_request_queue(self):
        """Process requests from the queue"""
        while True:
            try:
                # Get request from queue
                request = await self.request_queue.get()
                
                # Acquire semaphore for concurrency control
                async with self.semaphore:
                    # Process request
                    task = asyncio.create_task(self._handle_prediction_request(request))
                    self.active_requests[request.request_id] = task
                    
                    # Wait for completion or timeout
                    try:
                        await asyncio.wait_for(task, timeout=request.timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.warning(f"Request {request.request_id} timed out")
                        task.cancel()
                    
                    # Clean up
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]
                
            except Exception as e:
                logger.error(f"Error processing request queue: {e}")
    
    async def _handle_prediction_request(self, request: PredictionRequest) -> PredictionResponse:
        """Handle individual prediction request"""
        start_time = time.time()
        
        try:
            # Get model endpoint
            endpoint = self.load_balancer.get_next_endpoint()
            if not endpoint:
                raise Exception("No healthy model endpoints available")
            
            # Make prediction request
            prediction_data = {
                "model_id": request.model_id,
                "input_data": request.input_data,
                "request_id": request.request_id
            }
            
            # Use async HTTP client for better performance
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/predict",
                    json=prediction_data,
                    timeout=aiohttp.ClientTimeout(total=request.timeout_seconds)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        response_obj = PredictionResponse(
                            request_id=request.request_id,
                            prediction=result.get('prediction'),
                            confidence=result.get('confidence', 0.0),
                            latency_ms=(time.time() - start_time) * 1000,
                            model_version=result.get('model_version', 'unknown'),
                            timestamp=datetime.now(timezone.utc),
                            status=PredictionStatus.COMPLETED
                        )
                        
                        # Mark endpoint as successful
                        self.load_balancer.mark_success(endpoint)
                        
                        # Send to response queue
                        await self.response_queue.put(response_obj)
                        
                        return response_obj
                    else:
                        raise Exception(f"Model endpoint returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            
            # Mark endpoint as failed
            if endpoint:
                self.load_balancer.mark_failure(endpoint)
            
            response_obj = PredictionResponse(
                request_id=request.request_id,
                prediction=None,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                model_version="unknown",
                timestamp=datetime.now(timezone.utc),
                status=PredictionStatus.FAILED,
                error_message=str(e)
            )
            
            await self.response_queue.put(response_obj)
            return response_obj
    
    async def _wait_for_response(self, request_id: str) -> PredictionResponse:
        """Wait for response from the response queue"""
        while True:
            try:
                response = await asyncio.wait_for(self.response_queue.get(), timeout=1.0)
                if response.request_id == request_id:
                    return response
            except asyncio.TimeoutError:
                continue
    
    async def _update_metrics(self):
        """Update Prometheus metrics periodically"""
        while True:
            try:
                # Update queue size
                self.queue_size_gauge.set(self.request_queue.qsize())
                
                # Update active requests
                self.active_requests_gauge.set(len(self.active_requests))
                
                # Update cache hit ratio (simplified calculation)
                # In production, you'd track cache hits/misses more precisely
                self.cache_hit_ratio.set(0.3)  # Placeholder
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(10)
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queue_size": self.request_queue.qsize(),
            "active_requests": len(self.active_requests),
            "healthy_endpoints": len([
                ep for ep in self.model_endpoints 
                if self.load_balancer.circuit_breakers[ep].can_execute()
            ]),
            "total_endpoints": len(self.model_endpoints),
            "cache_available": self.prediction_cache.redis_client is not None
        }

# FastAPI application for the prediction service
app = FastAPI(title="Real-time Prediction Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
prediction_service = RealTimePredictionService()

# Pydantic models for API
class PredictionRequestModel(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    priority: str = Field("normal", description="Request priority")
    use_cache: bool = Field(True, description="Use prediction cache")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")

class PredictionResponseModel(BaseModel):
    request_id: str
    prediction: Any
    confidence: float
    latency_ms: float
    model_version: str
    timestamp: str
    status: str
    error_message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup"""
    await prediction_service.initialize()

@app.post("/predict", response_model=PredictionResponseModel)
async def predict_endpoint(request: PredictionRequestModel):
    """Main prediction endpoint"""
    try:
        response = await prediction_service.predict(
            model_id=request.model_id,
            input_data=request.input_data,
            priority=RequestPriority(request.priority),
            use_cache=request.use_cache,
            timeout_seconds=request.timeout_seconds
        )
        
        return PredictionResponseModel(
            request_id=response.request_id,
            prediction=response.prediction,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            model_version=response.model_version,
            timestamp=response.timestamp.isoformat(),
            status=response.status.value,
            error_message=response.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await prediction_service.get_health_status()

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    metrics = await prediction_service.get_metrics()
    return metrics

@app.get("/status")
async def status_endpoint():
    """Service status endpoint"""
    return {
        "service": "real-time-prediction",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
