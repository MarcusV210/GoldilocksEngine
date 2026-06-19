from llama_cpp import Llama
from sentence_transformers import CrossEncoder
import argparse
import pandas as pd
import numpy as np
import time
import sys
import os

def run_llm_ranking(parquet_path: str, model_path: str, output_csv_path: str):
    start_time = time.time()
    print(f"Loading precomputed data from {parquet_path}...")
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        sys.exit(1)
    
    # 1. Hard Filters
    df = df[df['meets_work_mode'] == True]
    df = df[df['meets_notice_period'] == True]
        
    # 2. Candidate Selection (Retrieve top 150 candidates by raw semantic score)
    df_sorted = df.sort_values(by=['semantic_similarity_score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    top_candidates = df_sorted.head(150).copy()
    
    print(f"Retrieved top 150 candidates. Loading Cross-Encoder ranking model...", flush=True)
    
    # 3. Deep AI Ranking (Cross-Encoder Re-ranker)
    try:
        cross_encoder = CrossEncoder("models/cross-encoder")
    except Exception as e:
        print(f"Failed to load Cross-Encoder model: {e}", flush=True)
        sys.exit(1)
        
    jd_text = """
    Production experience with embeddings-based retrieval systems like sentence-transformers, OpenAI embeddings, BGE, or E5 deployed to real users.
    Production experience with vector databases or hybrid search infrastructure like Pinecone, Weaviate, Qdrant, or Milvus.
    Strong Python programming and code quality.
    Hands-on experience designing evaluation frameworks for ranking systems using NDCG, MRR, MAP, and A/B test interpretation.
    """
    
    pairs = [[jd_text, str(row['composite_semantic_text'])] for _, row in top_candidates.iterrows()]
    
    print("Computing Cross-Encoder scores for candidate ranking...", flush=True)
    scores = cross_encoder.predict(pairs)
    top_candidates['score'] = scores
    
    # Sort by Cross-Encoder score descending
    final_sorted = top_candidates.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    # Isolate exactly Top 100
    top_100 = final_sorted.head(100).copy()
    top_100['rank'] = range(1, 101)
    
    print(f"Cross-Encoder ranking complete in {time.time() - start_time:.2f}s. Booting fast LLM via llama-cpp...", flush=True)

    # 4. Initialize Local LLM
    try:
        llm = Llama(
            model_path=model_path,
            n_threads=4, 
            n_ctx=1024,  
            verbose=False 
        )
    except Exception as e:
        print(f"Failed to load GGUF model: {e}", flush=True)
        sys.exit(1)

    reasonings = []
    
    # 5. The Generation Loop
    print("Generating factual reasoning for Top 100 candidates...", flush=True)
    for index, row in top_100.iterrows():
        rank = index + 1
        profile_context = row['composite_semantic_text']
        prompt = f"""<|im_start|>system
Write a single sentence summarizing the candidate's core metrics. Include their current title, years of experience, AI core skills, and response rate. Do not add any conversational preamble.<|im_end|>
<|im_start|>user
Candidate Data: 
- Years of Experience: {row['years_of_experience']:.1f}
- Current Title: {row['current_title']}
- AI Core Skills: {int(row.get('ai_core_skills', 0))}
- Response Rate: {row.get('response_rate', 0.0):.2f}<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = llm(
                prompt,
                max_tokens=40, 
                stop=["<|im_end|>", "\n"],
                temperature=0.5 # Increased for variance
            )
            generated_text = output['choices'][0]['text'].strip().replace('\n', ' ').replace('\r', '')
        except Exception as e:
            print(f"LLM generation failed for rank {rank}: {e}. Falling back to default description.", flush=True)
            generated_text = f"{row['current_title']} with {row['years_of_experience']:.1f} yrs; {int(row.get('ai_core_skills', 0))} AI core skills; response rate {row.get('response_rate', 0.0):.2f}."
            
        reasonings.append(generated_text)
        
        if rank % 10 == 0 or rank <= 3:
            print(f"Processed {rank}/100. Rationale: '{generated_text}'. Elapsed: {time.time() - start_time:.1f}s", flush=True)

    top_100['reasoning'] = reasonings

    # 6. Format to Spec and Export
    submission_df = top_100[['candidate_id', 'rank', 'score', 'reasoning']]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    submission_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    total_time = time.time() - start_time
    print(f"\nSuccess! Generated exactly 100 rows in {total_time:.2f} seconds.")
    if total_time > 300:
        print("WARNING: Exceeded 5-minute wall-clock limit!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final Redrob ranking CSV.")
    parser.add_argument("--precomputed", default="data/scored_candidates.parquet", help="Path to precomputed parquet file")
    parser.add_argument("--model", default="models/qwen-2.5/qwen2.5-0.5b-instruct-q4_k_m.gguf", help="Path to the quantized .gguf model")
    parser.add_argument("--out", default="outputs/results_1.csv", help="Path to output CSV")
    
    args = parser.parse_args()
    run_llm_ranking(args.precomputed, args.model, args.out)