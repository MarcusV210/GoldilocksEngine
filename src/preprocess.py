import json
import pandas as pd
from datetime import datetime
import re

# Hard coded titles and their approximate ranking
TITLE_TIERS = {
    "intern": 1,
    "junior": 2, "associate": 2, "trainee": 2,
    "engineer": 3, "developer": 3, "analyst": 3, "consultant": 3,
    "senior": 4, "sr.": 4,
    "lead": 5, "manager": 5,
    "staff": 6, "principal": 6, "architect": 6,
    "director": 7,
    "vp": 8, "head": 8
}

SERVICES_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree"]

def get_tier(title: str) -> int:
    """Helper to assign a numerical tier to a job title."""
    title_lower = title.lower()
    for key, tier in sorted(TITLE_TIERS.items(), key=lambda x: -len(x[0])):
        if key in title_lower:
            return tier
    return 3 # Default to 3


def process_candidate_record(candidate: dict, jd_reqs: dict, current_date: datetime) -> dict:
    """
    Candidates is the JSON, jd_reqs will need to be created from the job description
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    yoe = profile.get("years_of_experience", 0.0)
    
    row = {
        "candidate_id": candidate.get("candidate_id"),
        "years_of_experience": yoe,
        "current_title": profile.get("current_title", "Unknown"),
    }

    # Semantic skill matching
    text_chunks = [profile.get("headline", ""), profile.get("summary", "")]
    
    skill_statements = [] # To contain all skills
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "beginner")
        months = skill.get("duration_months", 0)
        
        if prof in ["intermediate", "advanced"] or months > 12:
            skill_statements.append(f"Demonstrates {prof} capability in {name} with {months} months of applied experience.")
            
    # Add job history descriptions
    for job in history:
        text_chunks.append(f"As a {job.get('title')}: {job.get('description', '')}")
        
    row["composite_semantic_text"] = " | ".join(skill_statements) + " || " + " | ".join(filter(None, text_chunks)).strip()


    # Velocity
    if history:
        earliest_title = history[-1].get("title", "")
        latest_title = history[0].get("title", "")
        
        earliest_tier = get_tier(earliest_title)
        latest_tier = get_tier(latest_title)

        row["career_velocity_slope"] = (latest_tier - earliest_tier) / max(yoe, 0.5) 
        
        leadership_months = 0
        title_keywords = ["lead", "manager", "director", "head", "principal", "vp"] # Hard coded because of the CPU restriction
        action_verbs = ["led a team", "mentored", "managed", "directed", "supervised", "grew the team"]
        
        for job in history:
            t_lower = job.get("title", "").lower()
            d_lower = job.get("description", "").lower()
            if any(kw in t_lower for kw in title_keywords) or any(verb in d_lower for verb in action_verbs):
                leadership_months += job.get("duration_months", 0)
                
        row["leadership_months"] = leadership_months
        row["is_proven_leader"] = leadership_months >= 12
    else:
        row["career_velocity_slope"] = 0.0
        row["leadership_months"] = 0
        row["is_proven_leader"] = False

    # Builder signal
    row["github_activity_score"] = signals.get("github_activity_score", 0.0)
    
    # Activity signal
    last_active_str = signals.get("last_active_date")
    if last_active_str:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        row["days_since_active"] = (current_date - last_active).days
    else:
        row["days_since_active"] = 999 

    # Negative score
    negative_score = 0
    is_employed = any(job.get("is_current", False) for job in history) # If they're currently employed
    
    if is_employed and signals.get("open_to_work_flag", False):
        negative_score += 1 # Flight risk
    if not is_employed and signals.get("applications_submitted_30d", 0) > 15:
        negative_score += 1 # Likely failing screens
    if signals.get("recruiter_response_rate", 1.0) < 0.40 or signals.get("avg_response_time_hours", 0.0) > 72.0:
        negative_score += 1 # Poor responsiveness
    
    # B. The "Title-Chaser" penalty 
    unique_companies = {job.get("company", "").lower() for job in history if job.get("company")}
    if len(unique_companies) > 1:
        avg_company_tenure = (yoe * 12) / len(unique_companies)
        if avg_company_tenure < 18.0:
            negative_score += 3
            
    # C. The "Services-Only" penalty
    if unique_companies and all(any(sf in c for sf in SERVICES_FIRMS) for c in unique_companies):
        negative_score += 2
        
    # D. The "Theoretical Researcher" penalty
    composite_desc = " ".join([job.get("description", "") for job in history]).lower()
    if yoe > 3.0 and not any(word in composite_desc for word in ["production", "deployed", "shipped", "scaled", "infrastructure"]):
        negative_score += 2

    row["behavioral_penalty_score"] = negative_score

    # JD requirements
    row["meets_notice_period"] = signals.get("notice_period_days", 999) <= jd_reqs.get("max_notice_days", 90)
    
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    exp_max = salary_range.get("max", 999.0)
    row["within_negotiable_budget"] = exp_max <= (jd_reqs.get("max_budget_lpa", 0.0) * 1.10)
    
    pref_mode = signals.get("preferred_work_mode", "").lower()
    row["meets_work_mode"] = pref_mode in jd_reqs.get("allowed_work_modes", [pref_mode])
    
    # Extra fields for final output
    ai_core_set = {"python", "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "transformers", "langchain", "llm", "rag", "embeddings", "vector search", "information retrieval", "generative ai", "data science"}
    ai_core_skills_count = sum(1 for skill in skills if skill.get("name", "").lower() in ai_core_set)
    row["ai_core_skills"] = ai_core_skills_count
    row["response_rate"] = signals.get("recruiter_response_rate", 0.0)

    return row


# Saving
def run_pipeline(input_json_path: str, output_parquet_path: str, jd_reqs: dict):
    print("Loading raw JSON/JSONL data...")
    raw_data = []
    with open(input_json_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            raw_data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        
    current_date = datetime(2026, 6, 4) 
    
    print(f"Processing {len(raw_data)} candidates...")
    processed_records = [
        process_candidate_record(cand, jd_reqs, current_date) 
        for cand in raw_data
    ]
    
    df = pd.DataFrame(processed_records)
    df.to_parquet(output_parquet_path, index=False)
    
    print(f"Success: Saved optimized dataset to {output_parquet_path}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return df

if __name__ == "__main__":
    jd_requirements_criteria = {
        "max_notice_days": 90,
        "max_budget_lpa": 45.0,
        "allowed_work_modes": ["hybrid", "remote", "flexible"]
    }
    
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(parent_dir, "India_runs_data_and_ai_challenge", "candidates.jsonl")
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "processed_candidates_15.parquet")
    
    run_pipeline(input_path, output_path, jd_requirements_criteria)