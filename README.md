# GoldilocksEngine: High-Velocity AI Candidate Ranking

This project implements a highly optimized, fully local, and privacy-preserving candidate ranking pipeline. It is designed to match 100,000 candidate profiles against a Job Description and extract reasoning for the top 100 fits, all while reliably executing under a strict 5-minute wall-clock limit on a local 16 GB CPU-only machine.

## Architecture

The pipeline consists of three highly-optimized stages:
1. **Preprocessing & Heuristics** (`src/preprocess.py`): Parses the JSON candidate data into a columnar Parquet format and extracts a subset of textual profiles.
2. **Semantic Embedding & Scoring** (`src/embed.py`): Utilizes the ultra-fast CPU-optimized `minishlab/potion-base-32M` (Model2Vec) to generate dense vector embeddings of the profiles and the JD, executing cosine similarity scoring across the entire dataset in seconds.
3. **Filtering, Ranking & Rationale** (`src/rank.py`): Ranks the candidates by their vector similarity score. The top 100 candidates are then processed by the "Goldilocks" model: a quantized local Small Language Model (SLM) `LiquidAI/LFM2.5-1.2B-Instruct` using `llama.cpp`. This model hits the perfect intelligence-to-speed ratio, allowing it to dynamically generate a factual justification of each candidate's rank based on their profile data without breaking the 5-minute competition limit.

## Project Structure

- `src/preprocess.py`: Data ingestion and heuristic processing.
- `src/embed.py`: Vector embeddings utilizing Model2Vec.
- `src/rank.py`: Final ranking and SLM rationale generation.
- `src/checker.py`: Script to structurally validate the output `submission.csv` against strict Hackathon constraints.
- `src/setup_models.py`: Utility script to download the required GGUF weights locally.
- `run_all.py`: Orchestrator script to execute the pipeline end-to-end.
- `comparision.md`: Detailed benchmarking data across various SLMs and temperature settings.

## Setup & Execution

### 1. Install Dependencies
Make sure you are on Python 3.11+.
```bash
pip install -r requirements.txt
```

### 2. Download Model Weights
You must download the localized LLM weights into the `models/` directory before running the pipeline.
```bash
python src/setup_models.py
```

### 3. Run Pipeline
Execute the full pipeline. The script reads from the `India_runs_data_and_ai_challenge/` dataset folder and writes intermediate files to `data/`. The final output is written to `outputs/`.
```bash
python run_all.py --force
```

### 4. Validate Submission
Verify that the output format strictly aligns with the Hackathon submission specifications.
```bash
python src/checker.py outputs/results_15.csv
```
