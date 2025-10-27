"""
Topic Segmentation Visual Example

This example demonstrates how LightMem performs topic segmentation
using the two-stage approach (attention + semantic similarity).
"""

import numpy as np
from typing import List, Dict


def visualize_attention_matrix(buffer_texts: List[str]):
    """
    Simulates the attention matrix computation in LlmLingua2Segmenter
    
    Args:
        buffer_texts: List of user messages
        
    Returns:
        M: Attention matrix where M[i,j] = attention from sentence i to j
    """
    print("\n=== STAGE 1: Attention-Based Coarse Segmentation ===\n")
    print("Input messages:")
    for i, msg in enumerate(buffer_texts):
        print(f"  {i}. {msg}")
    
    # Simulate attention matrix (in reality, computed from transformer)
    # Higher values = more attention (more related topics)
    n = len(buffer_texts)
    M = np.zeros((n, n))
    
    # Example: Sentences within same topic attend more to each other
    # Topic 1: messages 0-2
    # Topic 2: messages 3-4
    # Topic 3: messages 5-7
    
    M[0, 0] = 1.0  # Self-attention
    M[1, 0] = 0.8; M[1, 1] = 1.0  # Same topic (movies)
    M[2, 0] = 0.7; M[2, 1] = 0.8; M[2, 2] = 1.0
    
    M[3, 3] = 1.0  # Topic shift (work)
    M[4, 3] = 0.85; M[4, 4] = 1.0
    
    M[5, 5] = 1.0  # Topic shift (trip)
    M[6, 5] = 0.9; M[6, 6] = 1.0
    M[7, 5] = 0.8; M[7, 6] = 0.85; M[7, 7] = 1.0
    
    print(f"\nAttention Matrix M (shape {M.shape}):")
    print("     ", " ".join(f"{i:6}" for i in range(n)))
    for i in range(n):
        print(f"{i:3}", end="  ")
        for j in range(n):
            print(f"{M[i,j]:6.2f}", end=" ")
        print()
    
    # Compute "outer" attention (attention to immediately previous sentence)
    outer = [M[i, i-1] if i > 0 else 1.0 for i in range(n)]
    print(f"\nOuter attention (attention to previous sentence):")
    for i, val in enumerate(outer):
        print(f"  outer[{i}] = {val:.2f}")
    
    # Find boundaries (local maxima indicate topic shifts)
    boundaries = []
    for k in range(1, len(outer)-1):
        if outer[k-1] < outer[k] > outer[k+1]:
            boundaries.append(k)
            print(f"  → Boundary detected at position {k}!")
    
    print(f"\nCoarse boundaries: {boundaries}")
    return boundaries, M


def visualize_semantic_segmentation(buffer_messages: List[Dict]):
    """
    Simulates the semantic similarity-based fine-grained segmentation
    
    Args:
        buffer_messages: List of message dicts with user/assistant pairs
    """
    print("\n=== STAGE 2: Semantic Similarity Fine-Grained Segmentation ===\n")
    
    # Group into turns (user + assistant)
    turns = []
    for i in range(0, len(buffer_messages), 2):
        user = buffer_messages[i]["content"]
        assistant = buffer_messages[i+1]["content"] if i+1 < len(buffer_messages) else ""
        turn_text = f"{user} {assistant}".strip()
        turns.append(turn_text)
        print(f"Turn {len(turns)-1}: {turn_text[:50]}...")
    
    # Simulate embeddings and cosine similarity
    # In reality, computed by text_embedder.embed()
    n_turns = len(turns)
    embeddings = np.random.rand(n_turns, 384)  # Simulated embeddings
    
    print(f"\nSemantic similarities between consecutive turns:")
    for i in range(n_turns - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        print(f"  Turn {i} ↔ {i+1}: sim = {sim:.3f}")
    
    # Find semantic boundaries using threshold
    fine_boundaries = []
    threshold = 0.2
    
    print(f"\nFinding boundaries with threshold = {threshold}")
    for i in range(n_turns - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        if sim < threshold:
            fine_boundaries.append(i + 1)
            print(f"  → Semantic boundary at turn {i+1}")
    
    return fine_boundaries


def demonstrate_complete_segmentation():
    """
    Complete demonstration of the segmentation process
    """
    print("=" * 70)
    print("LIGHTMEM TOPIC SEGMENTATION DEMONSTRATION")
    print("=" * 70)
    
    # Example conversation
    buffer_texts = [
        "I really enjoy watching sci-fi movies.",
        "My favorite director is Christopher Nolan.",
        "Inception and Interstellar are amazing films.",
        "At work, I'm leading a new project.",
        "It's about developing a new AI system.",
        "I'm planning a trip to Japan next month.",
        "I want to visit Tokyo and Kyoto.",
        "Been saving up for this for a while."
    ]
    
    # Stage 1: Coarse boundaries
    coarse_boundaries, M = visualize_attention_matrix(buffer_texts)
    
    # Stage 2: Semantic refinement
    buffer_messages = [
        {"role": "user", "content": text} for text in buffer_texts
    ]
    fine_boundaries = visualize_semantic_segmentation(buffer_messages)
    
    # Stage 3: Adjustment
    print("\n=== STAGE 3: Boundary Adjustment ===\n")
    adjusted_boundaries = []
    for fb in fine_boundaries:
        for cb in coarse_boundaries:
            if abs(fb - cb) <= 3:
                adjusted_boundaries.append(fb)
                print(f"Fine boundary {fb} is within 3 of coarse {cb} → keep")
                break
    
    if not adjusted_boundaries:
        adjusted_boundaries = fine_boundaries
        print("No coarse/fine alignment, using fine boundaries")
    
    print(f"\nFinal adjusted boundaries: {adjusted_boundaries}")
    
    # Stage 4: Create segments
    print("\n=== STAGE 4: Segment Creation ===\n")
    boundaries = sorted(set(adjusted_boundaries))
    segments = []
    start = 0
    
    for boundary in boundaries:
        end = 2 * boundary
        seg = buffer_messages[start:end]
        segments.append(seg)
        print(f"Segment {len(segments)}: messages {start}-{end-1}")
        start = end
    
    if start < len(buffer_messages):
        seg = buffer_messages[start:]
        segments.append(seg)
        print(f"Segment {len(segments)}: messages {start}-{len(buffer_messages)-1}")
    
    print(f"\nCreated {len(segments)} segments")
    return segments


if __name__ == "__main__":
    segments = demonstrate_complete_segmentation()
    print("\n" + "=" * 70)
    print("Segmentation complete!")
    print("=" * 70)

