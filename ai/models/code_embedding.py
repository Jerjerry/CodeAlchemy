"""
CodeInsight Code Embedding Model

This module provides advanced code embedding capabilities using state-of-the-art
transformer models. It converts code into semantic vector representations that
capture both syntactic structure and semantic meaning.

Key features:
1. Multi-language support through unified embeddings
2. Context-aware code understanding
3. Hierarchical code representation
4. Attention-based relevance scoring
"""

import torch
import torch.nn as nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    T5EncoderModel,
    T5Tokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

@dataclass
class CodeEmbedding:
    """Represents an embedded code fragment with its metadata."""
    vector: torch.Tensor
    attention_weights: torch.Tensor
    token_map: Dict[int, Tuple[int, int]]  # Maps embedding indices to code positions
    confidence: float
    language: str
    context_window: Optional[List[str]]  # Surrounding code context


class HierarchicalCodeEncoder(nn.Module):
    """
    Hierarchical encoder that processes code at multiple levels:
    1. Token level (individual syntax tokens)
    2. Statement level (complete code statements)
    3. Block level (logical code blocks)
    4. Function/class level (complete definitions)
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length

        # Initialize base transformer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.transformer = RobertaModel.from_pretrained(model_name)
        
        # Move to appropriate device
        self.transformer = self.transformer.to(device)
        
        # Hierarchical components
        self.token_encoder = nn.Linear(self.transformer.config.hidden_size, 256)
        self.statement_encoder = nn.GRU(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.block_encoder = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.global_pooling = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        # Attention mechanisms
        self.token_attention = nn.MultiheadAttention(256, 4, dropout=0.1)
        self.statement_attention = nn.MultiheadAttention(1024, 8, dropout=0.1)
        
        logger.info(f"Initialized HierarchicalCodeEncoder on {device}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the hierarchical encoder.
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_length]
            attention_mask: Attention mask for padding [batch_size, seq_length]
            token_type_ids: Optional token type IDs for special tokens
            
        Returns:
            tuple: (final_embedding, attention_weights)
                - final_embedding: Code embedding tensor [batch_size, 256]
                - attention_weights: Dict of attention weights at each level
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        # Get base transformer embeddings
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        hidden_states = transformer_outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        attention_weights = {}

        # 1. Token-level encoding
        token_embeddings = self.token_encoder(hidden_states)  # [batch_size, seq_length, 256]
        token_embeddings = token_embeddings.permute(1, 0, 2)  # [seq_length, batch_size, 256]
        
        # Apply token-level attention
        token_output, token_attn = self.token_attention(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        attention_weights['token'] = token_attn
        
        # 2. Statement-level encoding
        token_output = token_output.permute(1, 0, 2)  # [batch_size, seq_length, 256]
        statement_output, _ = self.statement_encoder(token_output)
        # Combine bidirectional outputs
        statement_output = statement_output.view(
            statement_output.shape[0],
            statement_output.shape[1],
            2,
            512
        ).sum(dim=2)  # [batch_size, seq_length, 1024]
        
        # 3. Block-level encoding
        block_output = self.block_encoder(statement_output.permute(1, 0, 2))
        block_output = block_output.permute(1, 0, 2)  # [batch_size, seq_length, 1024]
        
        # Apply statement-level attention
        statement_output = statement_output.permute(1, 0, 2)  # [seq_length, batch_size, 1024]
        block_attn_output, block_attn = self.statement_attention(
            statement_output,
            statement_output,
            statement_output,
            key_padding_mask=~attention_mask.bool()
        )
        attention_weights['block'] = block_attn
        
        # 4. Global code representation
        # Use attention-weighted sum of block representations
        global_embedding = self.global_pooling(
            (block_attn_output.permute(1, 0, 2) * attention_mask.unsqueeze(-1)).sum(dim=1)
        )  # [batch_size, 256]

        return global_embedding, attention_weights

    def encode_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[List[str]] = None
    ) -> CodeEmbedding:
        """
        Encode a code snippet into its embedding representation.
        
        Args:
            code: The source code to encode
            language: Programming language of the code
            context: Optional surrounding code context
            
        Returns:
            CodeEmbedding: The embedded representation with metadata
        """
        # Prepare input
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate embedding
        with torch.no_grad():
            embedding, attention = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs.get('token_type_ids', None)
            )

        # Create token map
        token_map = {}
        for i, (token_start, token_end) in enumerate(self.tokenizer.encode_plus(
            code,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True
        )['offset_mapping']):
            if token_start != token_end:  # Skip special tokens
                token_map[i] = (token_start, token_end)

        # Calculate confidence based on attention patterns
        confidence = self._calculate_confidence(attention)

        return CodeEmbedding(
            vector=embedding[0],  # Take first (only) batch item
            attention_weights=attention['block'][0],  # First batch item
            token_map=token_map,
            confidence=confidence,
            language=language,
            context_window=context
        )

    def _calculate_confidence(self, attention_weights: Dict[str, torch.Tensor]) -> float:
        """Calculate embedding confidence score based on attention patterns."""
        # Use attention entropy as a proxy for confidence
        block_attention = attention_weights['block']
        
        # Calculate attention entropy
        attention_probs = torch.softmax(block_attention, dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1).mean()
        
        # Convert to confidence score (lower entropy = higher confidence)
        confidence = 1.0 - torch.clamp(entropy / np.log(attention_probs.size(-1)), 0.0, 1.0)
        
        return confidence.item()


class CodeEmbeddingManager:
    """
    Manages code embeddings and provides utilities for embedding operations.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.encoder = HierarchicalCodeEncoder(
            model_name=model_path or "microsoft/codebert-base"
        )
        self.encoder.eval()  # Set to evaluation mode
        
        # Initialize cache for embeddings
        self.embedding_cache: Dict[str, CodeEmbedding] = {}
        logger.info("CodeEmbeddingManager initialized")

    def get_embedding(
        self,
        code: str,
        language: str = "python",
        context: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> CodeEmbedding:
        """Get embedding for a code snippet, using cache if available."""
        cache_key = f"{language}:{hash(code)}"
        
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        embedding = self.encoder.encode_code(code, language, context)
        
        if use_cache:
            self.embedding_cache[cache_key] = embedding
            
        return embedding

    def compute_similarity(
        self,
        code1: Union[str, CodeEmbedding],
        code2: Union[str, CodeEmbedding],
        language: str = "python"
    ) -> float:
        """Compute semantic similarity between two code snippets."""
        # Get embeddings
        emb1 = code1 if isinstance(code1, CodeEmbedding) else self.get_embedding(code1, language)
        emb2 = code2 if isinstance(code2, CodeEmbedding) else self.get_embedding(code2, language)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.vector.unsqueeze(0),
            emb2.vector.unsqueeze(0)
        )
        
        return similarity.item()

    def find_similar_code(
        self,
        query: Union[str, CodeEmbedding],
        candidates: List[str],
        language: str = "python",
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find similar code snippets from a list of candidates."""
        # Get query embedding
        query_emb = query if isinstance(query, CodeEmbedding) else self.get_embedding(query, language)
        
        # Compute similarities
        similarities = []
        for candidate in candidates:
            similarity = self.compute_similarity(query_emb, candidate, language)
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
