import csv
import sys

def check_submission(file_path):
    print(f"Validating {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return False

    if not rows:
        print("[ERROR] File is empty.")
        return False

    # 1. Header Check
    expected_header = ['candidate_id', 'rank', 'score', 'reasoning']
    header = rows[0]
    if header != expected_header:
        print(f"[ERROR] Incorrect header.\nExpected: {expected_header}\nGot: {header}")
        return False

    data_rows = rows[1:]

    # 2. Exactly 100 rows
    if len(data_rows) != 100:
        print(f"[ERROR] Expected exactly 100 data rows, got {len(data_rows)}.")
        return False

    ranks = set()
    candidate_ids = set()
    prev_score = float('inf')
    reasonings = set()
    all_valid = True

    for i, row in enumerate(data_rows):
        if len(row) != 4:
            print(f"[ERROR] Row {i+1} does not have exactly 4 columns.")
            all_valid = False
            continue

        cid, rank_str, score_str, reasoning = row
        
        # 3. candidate_id exists and is unique
        if not cid.startswith("CAND_"):
            print(f"[ERROR] Row {i+1}: Invalid candidate_id '{cid}'. Must start with CAND_")
            all_valid = False
        if cid in candidate_ids:
            print(f"[ERROR] Row {i+1}: Duplicate candidate_id '{cid}'.")
            all_valid = False
        candidate_ids.add(cid)

        # 4. rank is int (1-100) and unique
        try:
            rank = int(rank_str)
            if rank < 1 or rank > 100:
                print(f"[ERROR] Row {i+1}: Rank {rank} is out of bounds (1-100).")
                all_valid = False
            if rank in ranks:
                print(f"[ERROR] Row {i+1}: Duplicate rank {rank}.")
                all_valid = False
            ranks.add(rank)
        except ValueError:
            print(f"[ERROR] Row {i+1}: Rank '{rank_str}' is not an integer.")
            all_valid = False

        # 5. score is non-increasing
        try:
            score = float(score_str)
            if score > prev_score:
                print(f"[ERROR] Row {i+1}: Score '{score}' is strictly greater than previous score '{prev_score}'. Scores must be non-increasing.")
                all_valid = False
            prev_score = score
        except ValueError:
            print(f"[ERROR] Row {i+1}: Score '{score_str}' is not a float.")
            all_valid = False

        # 6. Reasoning checks
        if not reasoning.strip():
            print(f"[WARNING] Row {i+1}: Reasoning is empty.")
        reasonings.add(reasoning.strip())

    if len(reasonings) == 1:
        print("[ERROR] All reasonings are identical. Evaluation heavily penalizes templated or identical strings.")
        all_valid = False

    if all_valid:
        print("[SUCCESS] Submission passes all automated structural checks!")
    else:
        print("[FAILED] Submission FAILED one or more automated checks.")

    return all_valid

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checker.py <path_to_submission.csv>")
        sys.exit(1)
    
    check_submission(sys.argv[1])
