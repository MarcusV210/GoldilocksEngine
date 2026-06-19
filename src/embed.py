import pandas as pd
from sentence_transformers import SentenceTransformer, util

def calculate_semantic_scores(parquet_path: str, jd_text: str, output_path: str):
    import time
    import numpy as np
    import os
    
    print("Loading preprocessed data...")
    df = pd.read_parquet(parquet_path)
    
    # Load the CPU-friendly Model2Vec embedding model
    print("Loading Model2Vec embedding model...")
    from model2vec import StaticModel
    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    
    # 1. Embed the Job Description
    print("Embedding Job Description...")
    jd_embedding = model.encode([jd_text])[0]
    
    # 2. Embed Candidate Profiles
    print(f"Embedding {len(df)} candidate profiles...")
    start_time = time.time()
    
    sentences = df['composite_semantic_text'].tolist()
    sentences_truncated = [str(text)[:600] for text in sentences]
    
    import torch
    
    chunk_embeddings = model.encode(sentences_truncated)
    
    # Calculate Cosine Similarity
    jd_embedding_tensor = torch.tensor(jd_embedding).unsqueeze(0)
    chunk_embeddings_tensor = torch.tensor(chunk_embeddings)
    cosine_scores = util.cos_sim(chunk_embeddings_tensor, jd_embedding_tensor).cpu().numpy().flatten()
    
    print(f"Embedding completed in {time.time() - start_time:.2f} seconds.")
    
    # 3. Save the real scores back to the DataFrame
    df['semantic_similarity_score'] = cosine_scores
    
    # Save the updated dataset
    df.to_parquet(output_path, index=False)
    print(f"Success! Real semantic scores saved to {output_path}")

if __name__ == "__main__":
    # The core requirements of the role to match against
    jd_requirements = """
    Production experience with embeddings-based retrieval systems like sentence-transformers, OpenAI embeddings, BGE, or E5 deployed to real users.
    Production experience with vector databases or hybrid search infrastructure like Pinecone, Weaviate, Qdrant, or Milvus.
    Strong Python programming and code quality.
    Hands-on experience designing evaluation frameworks for ranking systems using NDCG, MRR, MAP, and A/B test interpretation.
    """
    
    calculate_semantic_scores("data/processed_candidates.parquet", jd_requirements, "data/scored_candidates.parquet")