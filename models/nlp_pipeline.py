"""
Advanced NLP Pipeline for Clinical Documentation and Unstructured Text Analysis
Designed for healthcare domain with machine learning model integration
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# NLP Libraries
import spacy
import nltk
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    AutoModelForSequenceClassification, pipeline
)
import torch
from sentence_transformers import SentenceTransformer

# Healthcare-specific libraries
import pydicom
from biopython import SeqIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    CLINICAL_NOTE = "clinical_note"
    LAB_REPORT = "lab_report"
    IMAGING_REPORT = "imaging_report"
    PRESCRIPTION = "prescription"
    DISCHARGE_SUMMARY = "discharge_summary"
    JOB_DESCRIPTION = "job_description"
    LEARNING_RESOURCE = "learning_resource"

class EntityType(Enum):
    MEDICAL_CONDITION = "medical_condition"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    ANATOMICAL_STRUCTURE = "anatomical_structure"
    LAB_VALUE = "lab_value"
    SKILL = "skill"
    TECHNOLOGY = "technology"
    EXPERIENCE_LEVEL = "experience_level"

@dataclass
class ExtractedEntity:
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float
    context: str

@dataclass
class DocumentAnalysis:
    document_id: str
    document_type: DocumentType
    entities: List[ExtractedEntity]
    key_concepts: List[str]
    sentiment_score: float
    complexity_score: float
    structured_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]

class AdvancedNLPPipeline:
    """
    Advanced NLP pipeline for processing clinical documentation and unstructured text
    with healthcare domain expertise and machine learning integration
    """
    
    def __init__(self, 
                 clinical_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 ner_model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.clinical_model_name = clinical_model_name
        self.ner_model_name = ner_model_name
        self.sentence_model_name = sentence_model_name
        
        # Initialize models
        self._initialize_models()
        
        # Healthcare-specific patterns
        self.medical_patterns = {
            'conditions': [
                r'\b(?:diabetes|hypertension|asthma|copd|pneumonia|stroke|heart attack)\b',
                r'\b(?:cancer|carcinoma|tumor|neoplasm|malignancy)\b',
                r'\b(?:fracture|broken|dislocation|sprain|strain)\b'
            ],
            'medications': [
                r'\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril)\b',
                r'\b(?:mg|mcg|tablet|capsule|injection|drops)\b'
            ],
            'procedures': [
                r'\b(?:surgery|operation|biopsy|endoscopy|catheterization)\b',
                r'\b(?:x-ray|mri|ct scan|ultrasound|mammography)\b'
            ],
            'lab_values': [
                r'\b(?:glucose|cholesterol|hemoglobin|white blood cell|platelet)\b',
                r'\b(?:\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|g/dl|cells/μl))\b'
            ]
        }
        
        # Technical skills patterns
        self.tech_patterns = {
            'programming_languages': [
                r'\b(?:python|java|javascript|typescript|c\+\+|c#|go|rust|scala|r|matlab)\b'
            ],
            'frameworks': [
                r'\b(?:react|angular|vue|django|flask|spring|tensorflow|pytorch|spark)\b'
            ],
            'cloud_platforms': [
                r'\b(?:aws|azure|gcp|docker|kubernetes|terraform|jenkins)\b'
            ],
            'databases': [
                r'\b(?:postgresql|mysql|mongodb|redis|elasticsearch|snowflake)\b'
            ]
        }
        
        # Experience level indicators
        self.experience_patterns = {
            'junior': [r'\b(?:entry|junior|associate|trainee|intern|0-2|1-2)\s*(?:years?|yrs?)\b'],
            'mid': [r'\b(?:mid|intermediate|3-5|4-6)\s*(?:years?|yrs?)\b'],
            'senior': [r'\b(?:senior|lead|principal|staff|5\+|6\+|7\+)\s*(?:years?|yrs?)\b'],
            'expert': [r'\b(?:expert|architect|director|vp|10\+)\s*(?:years?|yrs?)\b']
        }
    
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            logger.info("Loading clinical BERT model...")
            self.clinical_tokenizer = AutoTokenizer.from_pretrained(self.clinical_model_name)
            self.clinical_model = AutoModelForSequenceClassification.from_pretrained(
                self.clinical_model_name
            )
            
            logger.info("Loading NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple"
            )
            
            logger.info("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer(self.sentence_model_name)
            
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def detect_document_type(self, text: str) -> DocumentType:
        """Detect document type using keyword matching and ML"""
        text_lower = text.lower()
        
        # Keyword-based detection
        type_indicators = {
            DocumentType.CLINICAL_NOTE: [
                'patient', 'chief complaint', 'history of present illness',
                'physical examination', 'assessment and plan'
            ],
            DocumentType.LAB_REPORT: [
                'laboratory results', 'blood test', 'urinalysis',
                'reference range', 'normal values'
            ],
            DocumentType.IMAGING_REPORT: [
                'radiograph', 'mri', 'ct scan', 'ultrasound',
                'impression', 'findings'
            ],
            DocumentType.PRESCRIPTION: [
                'prescription', 'medication', 'dosage', 'frequency',
                'pharmacy', 'refills'
            ],
            DocumentType.DISCHARGE_SUMMARY: [
                'discharge summary', 'discharge instructions',
                'follow-up', 'discharge date'
            ],
            DocumentType.JOB_DESCRIPTION: [
                'requirements', 'responsibilities', 'qualifications',
                'experience', 'skills', 'benefits'
            ],
            DocumentType.LEARNING_RESOURCE: [
                'course', 'tutorial', 'learning objectives',
                'curriculum', 'syllabus'
            ]
        }
        
        # Count matches for each type
        type_scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            type_scores[doc_type] = score
        
        # Return type with highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def extract_entities(self, text: str, document_type: DocumentType) -> List[ExtractedEntity]:
        """Extract entities using multiple approaches"""
        entities = []
        
        # Use NER model for general entities
        try:
            ner_results = self.ner_pipeline(text)
            for result in ner_results:
                entity_type = self._map_ner_label_to_entity_type(result['entity_group'])
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=result['word'],
                        label=entity_type,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        context=text[max(0, result['start']-50):result['end']+50]
                    ))
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        # Use pattern matching for domain-specific entities
        if document_type == DocumentType.CLINICAL_NOTE:
            entities.extend(self._extract_medical_entities(text))
        elif document_type in [DocumentType.JOB_DESCRIPTION, DocumentType.LEARNING_RESOURCE]:
            entities.extend(self._extract_technical_entities(text))
        
        return entities
    
    def _extract_medical_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract medical entities using patterns"""
        entities = []
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_type = self._map_category_to_entity_type(category)
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # Pattern-based confidence
                        context=text[max(0, match.start()-50):match.end()+50]
                    ))
        
        return entities
    
    def _extract_technical_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract technical entities using patterns"""
        entities = []
        
        for category, patterns in self.tech_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_type = self._map_category_to_entity_type(category)
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        context=text[max(0, match.start()-50):match.end()+50]
                    ))
        
        # Extract experience levels
        for level, patterns in self.experience_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=EntityType.EXPERIENCE_LEVEL,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        context=text[max(0, match.start()-50):match.end()+50]
                    ))
        
        return entities
    
    def _map_ner_label_to_entity_type(self, label: str) -> Optional[EntityType]:
        """Map NER labels to our entity types"""
        mapping = {
            'PER': None,  # Person names
            'ORG': None,  # Organizations
            'LOC': None,  # Locations
            'MISC': None  # Miscellaneous
        }
        return mapping.get(label)
    
    def _map_category_to_entity_type(self, category: str) -> EntityType:
        """Map pattern categories to entity types"""
        mapping = {
            'conditions': EntityType.MEDICAL_CONDITION,
            'medications': EntityType.MEDICATION,
            'procedures': EntityType.PROCEDURE,
            'lab_values': EntityType.LAB_VALUE,
            'programming_languages': EntityType.SKILL,
            'frameworks': EntityType.TECHNOLOGY,
            'cloud_platforms': EntityType.TECHNOLOGY,
            'databases': EntityType.TECHNOLOGY
        }
        return mapping.get(category, EntityType.SKILL)
    
    def extract_key_concepts(self, text: str, entities: List[ExtractedEntity]) -> List[str]:
        """Extract key concepts using spaCy and statistical methods"""
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities
        concepts = []
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to reasonable length
                concepts.append(chunk.text.lower())
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:  # Skip these
                continue
            concepts.append(ent.text.lower())
        
        # Add extracted entities
        for entity in entities:
            concepts.append(entity.text.lower())
        
        # Remove duplicates and filter
        concepts = list(set(concepts))
        concepts = [c for c in concepts if len(c) > 2 and c.isalpha()]
        
        return concepts[:20]  # Limit to top 20
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using clinical BERT"""
        try:
            # Tokenize and get predictions
            inputs = self.clinical_tokenizer(text, return_tensors="pt", 
                                           truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.clinical_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Return positive sentiment score
            return float(predictions[0][1])  # Assuming 1 is positive class
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.5  # Neutral sentiment as fallback
    
    def calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score"""
        doc = self.nlp(text)
        
        # Factors affecting complexity
        avg_sentence_length = sum(len(sent) for sent in doc.sents) / len(list(doc.sents))
        avg_word_length = sum(len(token) for token in doc if token.is_alpha) / len([t for t in doc if t.is_alpha])
        unique_word_ratio = len(set(token.text.lower() for token in doc if token.is_alpha)) / len([t for t in doc if t.is_alpha])
        
        # Normalize scores (rough normalization)
        complexity = (avg_sentence_length / 20 + avg_word_length / 8 + unique_word_ratio) / 3
        
        return min(1.0, max(0.0, complexity))
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate sentence embeddings for similarity analysis"""
        try:
            embedding = self.sentence_model.encode(text)
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def process_document(self, document_id: str, text: str) -> DocumentAnalysis:
        """Main method to process a document end-to-end"""
        logger.info(f"Processing document {document_id}")
        
        # Detect document type
        doc_type = self.detect_document_type(text)
        
        # Extract entities
        entities = self.extract_entities(text, doc_type)
        
        # Extract key concepts
        key_concepts = self.extract_key_concepts(text, entities)
        
        # Analyze sentiment
        sentiment_score = self.analyze_sentiment(text)
        
        # Calculate complexity
        complexity_score = self.calculate_complexity_score(text)
        
        # Generate embeddings
        embedding = self.generate_embeddings(text)
        
        # Create structured data
        structured_data = {
            'entities_by_type': self._group_entities_by_type(entities),
            'embedding': embedding.tolist(),
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(list(self.nlp(text).sents))
        }
        
        # Processing metadata
        processing_metadata = {
            'models_used': [
                self.clinical_model_name,
                self.ner_model_name,
                self.sentence_model_name
            ],
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'document_type_confidence': 0.8  # Could be calculated based on indicators
        }
        
        return DocumentAnalysis(
            document_id=document_id,
            document_type=doc_type,
            entities=entities,
            key_concepts=key_concepts,
            sentiment_score=sentiment_score,
            complexity_score=complexity_score,
            structured_data=structured_data,
            processing_metadata=processing_metadata
        )
    
    def _group_entities_by_type(self, entities: List[ExtractedEntity]) -> Dict[str, List[str]]:
        """Group entities by type for easier access"""
        grouped = {}
        for entity in entities:
            entity_type = entity.label.value
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity.text)
        
        return grouped
    
    def batch_process_documents(self, documents: List[Tuple[str, str]]) -> List[DocumentAnalysis]:
        """Process multiple documents in batch for efficiency"""
        results = []
        
        for doc_id, text in documents:
            try:
                analysis = self.process_document(doc_id, text)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                # Create minimal analysis for failed documents
                results.append(DocumentAnalysis(
                    document_id=doc_id,
                    document_type=DocumentType.JOB_DESCRIPTION,  # Default
                    entities=[],
                    key_concepts=[],
                    sentiment_score=0.5,
                    complexity_score=0.5,
                    structured_data={},
                    processing_metadata={'error': str(e)}
                ))
        
        return results

# Factory function for easy instantiation
def create_nlp_pipeline(config: Optional[Dict[str, str]] = None) -> AdvancedNLPPipeline:
    """Factory function to create NLP pipeline with optional configuration"""
    if config is None:
        config = {}
    
    return AdvancedNLPPipeline(
        clinical_model_name=config.get('clinical_model', 'emilyalsentzer/Bio_ClinicalBERT'),
        ner_model_name=config.get('ner_model', 'dbmdz/bert-large-cased-finetuned-conll03-english'),
        sentence_model_name=config.get('sentence_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
