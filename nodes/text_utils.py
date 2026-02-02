"""Text utilities for chunking long text (adapted from VibeVoice)."""

import re
from typing import List


def split_text_into_chunks(text: str, max_words: int = 250) -> List[str]:
    """Split long text into manageable chunks at sentence boundaries.
    
    Adapted from VibeVoice implementation.
    
    Args:
        text: The text to split
        max_words: Maximum words per chunk (default 250 for safety)
    
    Returns:
        List of text chunks
    """
    # Split into sentences (handling common abbreviations)
    # This regex tries to split on sentence endings while avoiding common abbreviations
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    # If regex split didn't work well, fall back to simple split
    if len(sentences) == 1 and len(text.split()) > max_words:
        # Fall back to splitting on any period followed by space
        sentences = text.replace('. ', '.|').split('|')
        sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If single sentence is too long, split it further
        if sentence_word_count > max_words:
            # Split long sentence at commas or semicolons
            sub_parts = re.split(r'[,;]', sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                part_words = part.split()
                part_word_count = len(part_words)
                
                if current_word_count + part_word_count > max_words and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [part]
                    current_word_count = part_word_count
                else:
                    current_chunk.append(part)
                    current_word_count += part_word_count
        else:
            # Check if adding this sentence would exceed the limit
            if current_word_count + sentence_word_count > max_words and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # If no chunks were created, return the original text
    if not chunks:
        chunks = [text]
    
    return chunks
