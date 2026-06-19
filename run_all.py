from llama_cpp import Llama
import time
import os
import sys

# Ensure current directory is in python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run full Redrob discovery pipeline.")
    parser.add_argument("--force", action="store_true", help="Force run preprocessing and embedding steps even if precomputed file exists.")
    args = parser.parse_args()

    start_total = time.time()
    
    # ------------------ Configurations ------------------
    candidates_json_path = "India_runs_data_and_ai_challenge/candidates.jsonl"
    processed_parquet_path = "data/processed_candidates_15.parquet"
    scored_parquet_path = "data/scored_candidates_15.parquet"
    output_csv_path = "outputs/results_15.csv"
    model_path = "models/liquid/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    
    jd_requirements_criteria = {
        "max_notice_days": 90,
        "max_budget_lpa": 45.0,
        "allowed_work_modes": ["hybrid", "remote", "flexible"]
    }
    
    jd_file_path = "India_runs_data_and_ai_challenge/job_description.md"
    with open(jd_file_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
    
    t_preprocess = 0.0
    t_embed = 0.0
    
    skip_precompute = os.path.exists(scored_parquet_path) and not args.force
    
    if skip_precompute:
        print("\n" + "="*50)
        print("STEP 1 & 2: Preprocessing & Embeddings (SKIPPED)")
        print("="*50)
        print(f"--> Found existing scored candidate data at {scored_parquet_path}.")
        print("--> Skipping pre-computation. Use --force to rebuild them.")
    else:
        # ------------------ Step 1: Preprocessing ------------------
        print("\n" + "="*50)
        print("STEP 1: Preprocessing & Heuristic Scoring")
        print("="*50)
        t0 = time.time()
        from src.preprocess import run_pipeline as run_preprocess
        
        run_preprocess(
            input_json_path=candidates_json_path,
            output_parquet_path=processed_parquet_path,
            jd_reqs=jd_requirements_criteria
        )
        t_preprocess = time.time() - t0
        print(f"--> Step 1 completed in {t_preprocess:.2f} seconds.")
        
        # ------------------ Step 2: Semantic Similarity ------------------
        print("\n" + "="*50)
        print("STEP 2: Generating Semantic Embeddings & Scores")
        print("="*50)
        t0 = time.time()
        from src.embed import calculate_semantic_scores as run_embed
        
        run_embed(
            parquet_path=processed_parquet_path,
            jd_text=jd_text,
            output_path=scored_parquet_path
        )
        t_embed = time.time() - t0
        print(f"--> Step 2 completed in {t_embed:.2f} seconds.")
    
    # ------------------ Step 3: LLM & Heuristic Ranking ------------------
    print("\n" + "="*50)
    print("STEP 3: Filtering, Ranking, & Match Rationale")
    print("="*50)
    t0 = time.time()
    from src.rank import run_llm_ranking as run_rank
    
    run_rank(
        parquet_path=scored_parquet_path,
        model_path=model_path,
        output_csv_path=output_csv_path
    )
    t_rank = time.time() - t0
    print(f"--> Step 3 completed in {t_rank:.2f} seconds.")
    
    # ------------------ Summary ------------------
    total_time = time.time() - start_total
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Preprocess Stage : {t_preprocess:.2f}s")
    print(f"Embedding Stage  : {t_embed:.2f}s")
    print(f"Ranking Stage    : {t_rank:.2f}s")
    print(f"Total Time       : {total_time:.2f}s ({(total_time/60):.2f} minutes)")
    print(f"Output File      : {output_csv_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
