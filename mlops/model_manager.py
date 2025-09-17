"""
MLOps Model Management System
Handles model versioning, deployment, monitoring, and lifecycle management
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import joblib

# MLOps Libraries
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from clearml import Task, Dataset, Model
import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient

# Monitoring and Observability
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import requests

# Model serving
import torch
import tensorflow as tf
from sklearn.base import BaseEstimator
import numpy as np

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    model_type: str  # sklearn, pytorch, tensorflow, custom
    framework_version: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    accuracy_metrics: Dict[str, float]
    training_data_hash: str
    model_size_mb: float
    dependencies: List[str]
    hyperparameters: Dict[str, Any]
    tags: Dict[str, str]
    description: str
    author: str

@dataclass
class DeploymentConfig:
    model_id: str
    environment: DeploymentEnvironment
    replicas: int
    cpu_requests: str
    memory_requests: str
    cpu_limits: str
    memory_limits: str
    health_check_path: str
    scaling_config: Dict[str, Any]
    deployment_timestamp: datetime

@dataclass
class ModelPerformance:
    model_id: str
    timestamp: datetime
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class ModelManager:
    """
    Comprehensive MLOps model management system with versioning, deployment, and monitoring
    """
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://localhost:5000",
                 clearml_project: str = "learning-recommender",
                 prometheus_port: int = 8001):
        
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.clearml_project = clearml_project
        self.prometheus_port = prometheus_port
        
        # Initialize MLOps platforms
        self._initialize_mlflow()
        self._initialize_clearml()
        self._initialize_prometheus()
        
        # Model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        self.deployments: Dict[str, DeploymentConfig] = {}
        
        # Performance tracking
        self.performance_history: List[ModelPerformance] = []
        
        # Metrics
        self.model_deployments = Counter('model_deployments_total', 'Total model deployments')
        self.model_predictions = Counter('model_predictions_total', 'Total model predictions', ['model_id', 'status'])
        self.prediction_latency = Histogram('model_prediction_latency_seconds', 'Model prediction latency', ['model_id'])
        self.model_accuracy = Gauge('model_accuracy', 'Model accuracy', ['model_id'])
        self.model_error_rate = Gauge('model_error_rate', 'Model error rate', ['model_id'])
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("learning-recommender-models")
            logger.info("MLflow initialized successfully")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
    
    def _initialize_clearml(self):
        """Initialize ClearML"""
        try:
            self.clearml_task = Task.init(
                project_name=self.clearml_project,
                task_name="model-management",
                auto_connect_frameworks=True
            )
            logger.info("ClearML initialized successfully")
        except Exception as e:
            logger.warning(f"ClearML initialization failed: {e}")
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        try:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.warning(f"Prometheus initialization failed: {e}")
    
    def register_model(self, 
                      model: Union[BaseEstimator, torch.nn.Module, tf.keras.Model],
                      name: str,
                      model_type: str,
                      training_data_hash: str,
                      accuracy_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      description: str = "",
                      author: str = "system",
                      tags: Dict[str, str] = None) -> str:
        """Register a new model in the system"""
        
        model_id = self._generate_model_id(name)
        version = self._get_next_version(name)
        
        # Calculate model size
        model_size = self._calculate_model_size(model)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            framework_version=self._get_framework_version(model_type),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status=ModelStatus.TRAINED,
            accuracy_metrics=accuracy_metrics,
            training_data_hash=training_data_hash,
            model_size_mb=model_size,
            dependencies=self._get_model_dependencies(model_type),
            hyperparameters=hyperparameters,
            tags=tags or {},
            description=description,
            author=author
        )
        
        # Store model
        self._store_model(model, metadata)
        
        # Register in MLflow
        self._register_in_mlflow(model, metadata)
        
        # Register in ClearML
        self._register_in_clearml(model, metadata)
        
        # Update registry
        self.model_registry[model_id] = metadata
        
        logger.info(f"Model {name} v{version} registered with ID {model_id}")
        return model_id
    
    def _generate_model_id(self, name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        hash_input = f"{name}_{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:16]
    
    def _get_next_version(self, name: str) -> str:
        """Get next version number for a model"""
        existing_versions = [
            meta.version for meta in self.model_registry.values() 
            if meta.name == name
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Simple version increment (in production, use semantic versioning)
        latest_version = max(existing_versions)
        major, minor, patch = map(int, latest_version.split('.'))
        
        if patch < 9:
            patch += 1
        elif minor < 9:
            minor += 1
            patch = 0
        else:
            major += 1
            minor = 0
            patch = 0
        
        return f"{major}.{minor}.{patch}"
    
    def _calculate_model_size(self, model: Union[BaseEstimator, torch.nn.Module, tf.keras.Model]) -> float:
        """Calculate model size in MB"""
        try:
            # Save model to temporary file and check size
            temp_path = "/tmp/temp_model.pkl"
            
            if hasattr(model, 'save'):
                if hasattr(model, 'save_model'):  # TensorFlow
                    model.save(temp_path)
                else:  # PyTorch
                    torch.save(model.state_dict(), temp_path)
            else:  # Scikit-learn
                joblib.dump(model, temp_path)
            
            size_bytes = os.path.getsize(temp_path)
            os.remove(temp_path)
            
            return size_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not calculate model size: {e}")
            return 0.0
    
    def _get_framework_version(self, model_type: str) -> str:
        """Get framework version"""
        version_map = {
            'sklearn': '1.3.0',
            'pytorch': torch.__version__,
            'tensorflow': tf.__version__,
            'custom': '1.0.0'
        }
        return version_map.get(model_type, 'unknown')
    
    def _get_model_dependencies(self, model_type: str) -> List[str]:
        """Get model dependencies"""
        base_deps = ['numpy', 'pandas']
        
        type_deps = {
            'sklearn': ['scikit-learn'],
            'pytorch': ['torch', 'torchvision'],
            'tensorflow': ['tensorflow', 'keras'],
            'custom': []
        }
        
        return base_deps + type_deps.get(model_type, [])
    
    def _store_model(self, model: Union[BaseEstimator, torch.nn.Module, tf.keras.Model], 
                    metadata: ModelMetadata):
        """Store model in local registry"""
        model_dir = f"models/registry/{metadata.name}/{metadata.version}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "model.pkl")
        
        try:
            if hasattr(model, 'save'):
                if hasattr(model, 'save_model'):  # TensorFlow
                    model.save(model_path)
                else:  # PyTorch
                    torch.save(model.state_dict(), model_path)
            else:  # Scikit-learn
                joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error storing model: {e}")
            raise
    
    def _register_in_mlflow(self, model: Union[BaseEstimator, torch.nn.Module, tf.keras.Model], 
                           metadata: ModelMetadata):
        """Register model in MLflow"""
        try:
            with mlflow.start_run(run_name=f"{metadata.name}_{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                mlflow.log_metrics(metadata.accuracy_metrics)
                
                # Log model
                if metadata.model_type == 'sklearn':
                    mlflow.sklearn.log_model(model, "model")
                elif metadata.model_type == 'pytorch':
                    mlflow.pytorch.log_model(model, "model")
                elif metadata.model_type == 'tensorflow':
                    mlflow.tensorflow.log_model(model, "model")
                
                # Log metadata
                mlflow.log_text(json.dumps(metadata.tags, indent=2), "tags.json")
                mlflow.set_tag("model_id", metadata.model_id)
                mlflow.set_tag("author", metadata.author)
                
        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
    
    def _register_in_clearml(self, model: Union[BaseEstimator, torch.nn.Module, tf.keras.Model], 
                            metadata: ModelMetadata):
        """Register model in ClearML"""
        try:
            clearml_model = Model.create(
                project_name=self.clearml_project,
                name=metadata.name,
                tags=list(metadata.tags.keys())
            )
            
            # Upload model
            model_path = f"models/registry/{metadata.name}/{metadata.version}/model.pkl"
            clearml_model.update_weights(weights_url=model_path)
            
            # Set metadata
            clearml_model.update_design(config_dict=metadata.hyperparameters)
            clearml_model.update_labels(labels=metadata.tags)
            
        except Exception as e:
            logger.warning(f"ClearML registration failed: {e}")
    
    def deploy_model(self, 
                    model_id: str,
                    environment: DeploymentEnvironment,
                    replicas: int = 1,
                    cpu_requests: str = "500m",
                    memory_requests: str = "1Gi",
                    cpu_limits: str = "1",
                    memory_limits: str = "2Gi") -> str:
        """Deploy model to specified environment"""
        
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.model_registry[model_id]
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            model_id=model_id,
            environment=environment,
            replicas=replicas,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            cpu_limits=cpu_limits,
            memory_limits=memory_limits,
            health_check_path="/health",
            scaling_config={
                "min_replicas": 1,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            },
            deployment_timestamp=datetime.now(timezone.utc)
        )
        
        # Deploy to Triton (if available)
        self._deploy_to_triton(metadata, deployment_config)
        
        # Update model status
        metadata.status = ModelStatus.DEPLOYED
        metadata.updated_at = datetime.now(timezone.utc)
        
        # Store deployment config
        self.deployments[model_id] = deployment_config
        
        # Update metrics
        self.model_deployments.inc()
        
        logger.info(f"Model {metadata.name} v{metadata.version} deployed to {environment.value}")
        return model_id
    
    def _deploy_to_triton(self, metadata: ModelMetadata, deployment_config: DeploymentConfig):
        """Deploy model to Triton Inference Server"""
        try:
            # This is a simplified deployment - in production, you'd use Triton's model repository
            # and proper model configuration files
            
            triton_client = tritonhttpclient.InferenceServerClient(
                url=f"localhost:8000",
                verbose=False
            )
            
            # Check if server is ready
            if not triton_client.is_server_ready():
                logger.warning("Triton server not ready, skipping deployment")
                return
            
            # Create model repository structure
            model_repo_path = f"/models/{metadata.name}/{metadata.version}"
            os.makedirs(model_repo_path, exist_ok=True)
            
            # Copy model files
            source_model_path = f"models/registry/{metadata.name}/{metadata.version}/model.pkl"
            if os.path.exists(source_model_path):
                import shutil
                shutil.copy2(source_model_path, model_repo_path)
            
            logger.info(f"Model deployed to Triton at {model_repo_path}")
            
        except Exception as e:
            logger.warning(f"Triton deployment failed: {e}")
    
    def predict(self, model_id: str, input_data: Any) -> Dict[str, Any]:
        """Make prediction using deployed model"""
        start_time = datetime.now()
        
        try:
            if model_id not in self.deployments:
                raise ValueError(f"Model {model_id} not deployed")
            
            # Load model
            metadata = self.model_registry[model_id]
            model_path = f"models/registry/{metadata.name}/{metadata.version}/model.pkl"
            
            if metadata.model_type == 'sklearn':
                model = joblib.load(model_path)
                prediction = model.predict(input_data)
            elif metadata.model_type == 'pytorch':
                model = torch.load(model_path)
                model.eval()
                with torch.no_grad():
                    prediction = model(input_data)
            elif metadata.model_type == 'tensorflow':
                model = tf.keras.models.load_model(model_path)
                prediction = model.predict(input_data)
            else:
                raise ValueError(f"Unsupported model type: {metadata.model_type}")
            
            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.model_predictions.labels(model_id=model_id, status='success').inc()
            self.prediction_latency.labels(model_id=model_id).observe(latency)
            
            return {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'model_id': model_id,
                'latency': latency,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Update error metrics
            self.model_predictions.labels(model_id=model_id, status='error').inc()
            logger.error(f"Prediction failed for model {model_id}: {e}")
            raise
    
    def monitor_model_performance(self, model_id: str) -> ModelPerformance:
        """Monitor model performance metrics"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Get model-specific metrics (simplified)
            latency_p50 = 0.1  # Would be calculated from actual metrics
            latency_p95 = 0.5
            latency_p99 = 1.0
            throughput = 100  # requests per second
            error_rate = 0.01
            
            # Get accuracy metrics from registry
            metadata = self.model_registry.get(model_id)
            accuracy = metadata.accuracy_metrics.get('accuracy', 0.0) if metadata else 0.0
            precision = metadata.accuracy_metrics.get('precision', 0.0) if metadata else 0.0
            recall = metadata.accuracy_metrics.get('recall', 0.0) if metadata else 0.0
            f1_score = metadata.accuracy_metrics.get('f1_score', 0.0) if metadata else 0.0
            
            performance = ModelPerformance(
                model_id=model_id,
                timestamp=datetime.now(timezone.utc),
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                throughput=throughput,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=None,  # Would be implemented with GPU monitoring
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            # Update Prometheus metrics
            self.model_accuracy.labels(model_id=model_id).set(accuracy)
            self.model_error_rate.labels(model_id=model_id).set(error_rate)
            
            # Store performance data
            self.performance_history.append(performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance monitoring failed for model {model_id}: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[ModelMetadata]:
        """Get all versions of a model"""
        return [
            metadata for metadata in self.model_registry.values()
            if metadata.name == model_name
        ]
    
    def get_deployment_status(self, model_id: str) -> Optional[DeploymentConfig]:
        """Get deployment status for a model"""
        return self.deployments.get(model_id)
    
    def deprecate_model(self, model_id: str, reason: str = ""):
        """Deprecate a model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.model_registry[model_id]
        metadata.status = ModelStatus.DEPRECATED
        metadata.updated_at = datetime.now(timezone.utc)
        
        if reason:
            metadata.tags['deprecation_reason'] = reason
        
        logger.info(f"Model {metadata.name} v{metadata.version} deprecated: {reason}")
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model metrics"""
        total_models = len(self.model_registry)
        deployed_models = len(self.deployments)
        deprecated_models = len([
            m for m in self.model_registry.values() 
            if m.status == ModelStatus.DEPRECATED
        ])
        
        return {
            'total_models': total_models,
            'deployed_models': deployed_models,
            'deprecated_models': deprecated_models,
            'average_model_size_mb': np.mean([
                m.model_size_mb for m in self.model_registry.values()
            ]) if self.model_registry else 0,
            'models_by_status': {
                status.value: len([
                    m for m in self.model_registry.values() 
                    if m.status == status
                ]) for status in ModelStatus
            }
        }

# Factory function
def create_model_manager(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    """Factory function to create model manager with configuration"""
    if config is None:
        config = {}
    
    return ModelManager(
        mlflow_tracking_uri=config.get('mlflow_uri', 'http://localhost:5000'),
        clearml_project=config.get('clearml_project', 'learning-recommender'),
        prometheus_port=config.get('prometheus_port', 8001)
    )
