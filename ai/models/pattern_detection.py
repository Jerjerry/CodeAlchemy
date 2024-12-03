"""
CodeInsight Pattern Detection Model

This module provides advanced pattern recognition and anomaly detection
capabilities for code analysis. It can:
1. Identify common code patterns and anti-patterns
2. Detect anomalous code segments
3. Recognize design patterns
4. Find potential code smells
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger

from .code_embedding import CodeEmbedding, CodeEmbeddingManager


@dataclass
class PatternMatch:
    """Represents a detected code pattern."""
    pattern_name: str
    pattern_type: str  # 'design_pattern', 'anti_pattern', 'code_smell'
    confidence: float
    locations: List[Tuple[int, int]]  # (start_line, end_line)
    explanation: str
    suggestions: List[str]
    severity: str  # 'high', 'medium', 'low'
    references: List[str]


@dataclass
class CodeAnomaly:
    """Represents a detected code anomaly."""
    anomaly_type: str
    severity: float  # 0.0 to 1.0
    location: Tuple[int, int]  # (start_line, end_line)
    description: str
    potential_causes: List[str]
    recommendations: List[str]
    confidence: float
    context: Optional[str]


class PatternDetectionModel(nn.Module):
    """
    Neural model for detecting patterns and anomalies in code.
    Uses a combination of transformer models and custom architectures.
    """

    def __init__(
        self,
        pattern_model: str = "microsoft/codebert-base",
        anomaly_model: str = "microsoft/codebert-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        
        # Initialize embedding manager
        self.embedding_manager = CodeEmbeddingManager()
        
        # Initialize pattern detection model
        self.pattern_tokenizer = RobertaTokenizer.from_pretrained(pattern_model)
        self.pattern_model = RobertaModel.from_pretrained(pattern_model)
        self.pattern_model = self.pattern_model.to(device)
        
        # Initialize anomaly detection model
        self.anomaly_tokenizer = RobertaTokenizer.from_pretrained(anomaly_model)
        self.anomaly_model = RobertaModel.from_pretrained(anomaly_model)
        self.anomaly_model = self.anomaly_model.to(device)
        
        # Pattern detection heads
        self.pattern_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.get_known_patterns()))
        )
        
        # Anomaly detection heads
        self.anomaly_detector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Load pattern definitions
        self.patterns = self._load_patterns()
        
        logger.info(f"Initialized PatternDetectionModel on {device}")

    def detect_patterns(
        self,
        code: str,
        language: str = "python",
        min_confidence: float = 0.7
    ) -> List[PatternMatch]:
        """
        Detect code patterns in the given code snippet.
        
        Args:
            code: Source code to analyze
            language: Programming language
            min_confidence: Minimum confidence threshold
            
        Returns:
            List[PatternMatch]: List of detected patterns
        """
        # Get code embedding
        embedding = self.embedding_manager.get_embedding(code, language)
        
        # Detect patterns using transformer
        inputs = self.pattern_tokenizer(
            code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.pattern_model(**inputs)
            pattern_logits = self.pattern_classifier(outputs.last_hidden_state[:, 0, :])
            pattern_probs = torch.sigmoid(pattern_logits)
        
        # Convert to pattern matches
        patterns = []
        for idx, prob in enumerate(pattern_probs[0]):
            if prob.item() >= min_confidence:
                pattern_def = self.patterns[idx]
                pattern = PatternMatch(
                    pattern_name=pattern_def["name"],
                    pattern_type=pattern_def["type"],
                    confidence=prob.item(),
                    locations=self._find_pattern_locations(code, pattern_def),
                    explanation=pattern_def["explanation"],
                    suggestions=pattern_def["suggestions"],
                    severity=pattern_def["severity"],
                    references=pattern_def["references"]
                )
                patterns.append(pattern)
        
        return patterns

    def detect_anomalies(
        self,
        code: str,
        language: str = "python",
        context: Optional[List[str]] = None,
        threshold: float = 0.8
    ) -> List[CodeAnomaly]:
        """
        Detect anomalies in the given code snippet.
        
        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional surrounding code context
            threshold: Anomaly detection threshold
            
        Returns:
            List[CodeAnomaly]: List of detected anomalies
        """
        # Get code embedding
        embedding = self.embedding_manager.get_embedding(code, language, context)
        
        # Detect anomalies using transformer
        inputs = self.anomaly_tokenizer(
            code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.anomaly_model(**inputs)
            anomaly_scores = self.anomaly_detector(outputs.last_hidden_state)
        
        # Find anomalous segments
        anomalies = []
        for i, score in enumerate(anomaly_scores[0]):
            if score.item() > threshold:
                anomaly = self._analyze_anomaly(
                    code,
                    i,
                    score.item(),
                    embedding,
                    context
                )
                anomalies.append(anomaly)
        
        return anomalies

    @staticmethod
    def get_known_patterns() -> List[str]:
        """Get list of known pattern types."""
        return [
            # Design Patterns
            "singleton",
            "factory",
            "observer",
            "strategy",
            "decorator",
            # Anti-patterns
            "god_object",
            "spaghetti_code",
            "copy_paste",
            "magic_numbers",
            # Code Smells
            "long_method",
            "large_class",
            "duplicate_code",
            "dead_code",
            "complex_condition"
        ]

    def _load_patterns(self) -> List[Dict]:
        """Load pattern definitions."""
        # In practice, would load from a configuration file
        return [
            {
                "name": "singleton",
                "type": "design_pattern",
                "explanation": "Ensures a class has only one instance",
                "suggestions": ["Use singleton when exactly one object is needed"],
                "severity": "low",
                "references": ["Design Patterns: Elements of Reusable Object-Oriented Software"]
            },
            # Add more patterns...
        ]

    def _find_pattern_locations(
        self,
        code: str,
        pattern_def: Dict
    ) -> List[Tuple[int, int]]:
        """Find locations of pattern in code."""
        # In practice, would use more sophisticated pattern matching
        return [(1, 10)]  # Placeholder

    def _analyze_anomaly(
        self,
        code: str,
        position: int,
        score: float,
        embedding: CodeEmbedding,
        context: Optional[List[str]]
    ) -> CodeAnomaly:
        """Analyze detected anomaly in detail."""
        # In practice, would perform more sophisticated analysis
        return CodeAnomaly(
            anomaly_type="unusual_pattern",
            severity=score,
            location=(position, position + 1),
            description="Unusual code pattern detected",
            potential_causes=["Possible error", "Unconventional implementation"],
            recommendations=["Review code logic", "Consider refactoring"],
            confidence=score,
            context=context[position] if context else None
        )

    def _get_pattern_features(
        self,
        embedding: CodeEmbedding
    ) -> torch.Tensor:
        """Extract pattern-specific features from code embedding."""
        # Project embedding for pattern detection
        pattern_features = self.pattern_classifier(embedding.vector)
        return pattern_features

    def _get_anomaly_features(
        self,
        embedding: CodeEmbedding
    ) -> torch.Tensor:
        """Extract anomaly-specific features from code embedding."""
        # Project embedding for anomaly detection
        anomaly_features = self.anomaly_detector(embedding.vector)
        return anomaly_features

    def explain_pattern(
        self,
        pattern: PatternMatch,
        code: str,
        language: str = "python"
    ) -> str:
        """Generate detailed explanation of detected pattern."""
        explanation = f"Pattern: {pattern.pattern_name}\n"
        explanation += f"Type: {pattern.pattern_type}\n"
        explanation += f"Confidence: {pattern.confidence:.2f}\n\n"
        explanation += f"Explanation: {pattern.explanation}\n\n"
        explanation += "Suggestions:\n"
        for suggestion in pattern.suggestions:
            explanation += f"- {suggestion}\n"
        explanation += f"\nSeverity: {pattern.severity}\n"
        explanation += "\nReferences:\n"
        for ref in pattern.references:
            explanation += f"- {ref}\n"
        return explanation

    def explain_anomaly(
        self,
        anomaly: CodeAnomaly,
        code: str,
        language: str = "python"
    ) -> str:
        """Generate detailed explanation of detected anomaly."""
        explanation = f"Anomaly Type: {anomaly.anomaly_type}\n"
        explanation += f"Severity: {anomaly.severity:.2f}\n"
        explanation += f"Confidence: {anomaly.confidence:.2f}\n\n"
        explanation += f"Description: {anomaly.description}\n\n"
        explanation += "Potential Causes:\n"
        for cause in anomaly.potential_causes:
            explanation += f"- {cause}\n"
        explanation += "\nRecommendations:\n"
        for rec in anomaly.recommendations:
            explanation += f"- {rec}\n"
        if anomaly.context:
            explanation += f"\nContext:\n{anomaly.context}"
        return explanation
