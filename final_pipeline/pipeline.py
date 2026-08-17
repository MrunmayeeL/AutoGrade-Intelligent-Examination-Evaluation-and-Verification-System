import os
import json
import re
import hashlib
from ocr_pixtral import extract_text_from_image, client
from evaluator import evaluate_single

INPUT_ROOT = "input_papers"
OUTPUT_ROOT = "output"
ANSWER_KEY_FILE = "answer_key.json"
CACHE_DIR = ".ocr_cache"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# LOAD & PARSE ANSWER KEY
# -----------------------------
def load_answer_key():
    with open(ANSWER_KEY_FILE, "r", encoding="utf-8") as f:
        raw_key = json.load(f)
    
    answer_key = {}
    for q, val in raw_key.items():
        q_clean = normalize_q(q)
        if isinstance(val, dict):
            answer_key[q_clean] = {
                "answer": val.get("answer", ""),
                "max_marks": int(val.get("max_marks", 5))
            }
        else:
            answer_key[q_clean] = {
                "answer": str(val),
                "max_marks": 5
            }
    return answer_key

# -----------------------------
# OCR CACHING BY FILE HASH
# -----------------------------
def get_image_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_text_cached(image_path):
    try:
        img_hash = get_image_hash(image_path)
        cache_file = os.path.join(CACHE_DIR, f"{img_hash}.txt")
        
        if os.path.exists(cache_file):
            print(f"  [Cache Hit] Using cached OCR for {os.path.basename(image_path)}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        
        print(f"  [Cache Miss] Running Pixtral OCR on {os.path.basename(image_path)}...")
        text = extract_text_from_image(image_path)
        
        # Normalize text variations for questions
        text = text.replace("Ql", "Q1").replace("Q l", "Q1")
        
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
            
        return text
    except Exception as e:
        print(f"  [OCR Error] Failed to process {image_path}: {e}")
        return ""

# -----------------------------
# ROLL NUMBER DETECTION & SEGREGATION
# -----------------------------
def detect_roll_number(text):
    # Pattern 1: Explicit separator (colon, hyphen, hash, dot) - accepts any alphanumeric/hyphenated code
    explicit_patterns = [
        r"(?:roll\s*no|roll\s*number|roll|student\s*id|enrollment\s*no|id|candidate\s*no)[:\-#\.]+\s*(\b[\w\-]+\b)"
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    # Pattern 2: Space separator - requires the ID to contain at least one digit to avoid matching dictionary words
    space_numeric_patterns = [
        r"(?:roll\s*no|roll\s*number|roll|student\s*id|enrollment\s*no|id|candidate\s*no)\s+(\b[\w\-]*\d+[\w\-]*\b)"
    ]
    for pattern in space_numeric_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    return None

def normalize_q(q):
    q = q.replace(".", "").replace(")", "")
    q = q.replace(" ", "")
    return q.upper()

def split_answers(text):
    pattern = r'(Q\s*\d+[a-zA-Z]?[\.\)]?)'
    parts = re.split(pattern, text)
    
    qa_pairs = {}
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            question = parts[i].strip()
            answer = parts[i + 1].strip()
            qa_pairs[question] = answer
    return qa_pairs

# (Scoring and LLM evaluation methods refactored out to evaluator.py)

# -----------------------------
# SCORECARD GENERATION
# -----------------------------
def generate_scorecard_md(student_name, results):
    total_score = sum(r["llm_grade"] for r in results if r["llm_grade"] is not None)
    total_max = sum(r["max_marks"] for r in results if r["max_marks"] is not None)
    percentage = (total_score / total_max * 100) if total_max > 0 else 0
    
    md = f"""# Academic Performance Scorecard

**Student Identifier**: `{student_name}`  
**Overall Grade**: **{total_score:.1f} / {total_max}** ({percentage:.1f}%)

---

## Detailed Evaluation Report

"""
    for r in results:
        q_name = r["question"]
        student_ans = r["student_answer"]
        ref_ans = r["reference_answer"]
        bert_sc = r["bert_score"]
        sbert_sc = r["sbert_score"]
        grade = r["llm_grade"]
        max_m = r["max_marks"]
        feedback = r["llm_feedback"]
        
        md += f"""### {q_name} (Max Marks: {max_m})

- **Student Answer**:  
  > *{student_ans}*
- **Reference Answer**:  
  > *{ref_ans}*
- **Semantic Similarity Scores**:
  - **SBERT (Mistral API)**: `{sbert_sc:.4f}`
  - **BERT Score**: `{bert_sc:.4f}` if bert_sc is not None else `N/A`
- **Assigned Grade**: **{grade:.1f} / {max_m}**
- **Evaluator Feedback**:  
  > {feedback}

---

"""
    md += "\n*Generated automatically by Advanced Agentic Grading Pipeline.*\n"
    return md

# -----------------------------
# PROCESS SINGLE STUDENT
# -----------------------------
def process_student(student_name, page_ocr_list, answer_key):
    full_text = "\n".join([ocr for _, ocr in page_ocr_list])
    qa_pairs = split_answers(full_text)
    
    results = []
    
    for q, student_ans in qa_pairs.items():
        q_clean = normalize_q(q)
        ref_info = answer_key.get(q_clean, None)
        
        if ref_info:
            ref_ans = ref_info["answer"]
            max_m = ref_info["max_marks"]
            eval_res = evaluate_single(q_clean, ref_ans, student_ans, max_m)
            bert_sc = eval_res["bert_score"]
            sbert_sc = eval_res["sbert_score"]
            grade = eval_res["llm_grade"]
            feedback = eval_res["llm_feedback"]
        else:
            ref_ans = None
            max_m = 5
            bert_sc = 0.0
            sbert_sc = 0.0
            grade = 0.0
            feedback = f"Question {q_clean} not found in the answer key."
            
        results.append({
            "question": q_clean,
            "student_answer": student_ans,
            "reference_answer": ref_ans,
            "max_marks": max_m,
            "bert_score": bert_sc,
            "sbert_score": sbert_sc,
            "llm_grade": grade,
            "llm_feedback": feedback
        })
        
    return full_text, results

def save_results(student_name, text, results):
    student_out = os.path.join(OUTPUT_ROOT, student_name)
    os.makedirs(student_out, exist_ok=True)
    
    # Save extracted text
    with open(os.path.join(student_out, "extracted.txt"), "w", encoding="utf-8") as f:
        f.write(text)
        
    # Save JSON
    with open(os.path.join(student_out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Save scorecard
    scorecard = generate_scorecard_md(student_name, results)
    with open(os.path.join(student_out, "scorecard.md"), "w", encoding="utf-8") as f:
        f.write(scorecard)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Input folder '{INPUT_ROOT}' does not exist. Creating it.")
        os.makedirs(INPUT_ROOT, exist_ok=True)
        return

    answer_key = load_answer_key()
    
    # Check if we have subdirectories in INPUT_ROOT
    subdirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    student_groups = {}
    
    if subdirs:
        print(f"Detected {len(subdirs)} structured student folder(s) in '{INPUT_ROOT}'.")
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(INPUT_ROOT, subdir)
            files = sorted(os.listdir(subdir_path))
            image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            if image_files:
                student_groups[subdir] = []
                for filename in image_files:
                    img_path = os.path.join(subdir_path, filename)
                    ocr_text = extract_text_cached(img_path)
                    student_groups[subdir].append((img_path, ocr_text))
    else:
        print(f"No student folders found. Analyzing flat scans in '{INPUT_ROOT}'...")
        files = sorted(os.listdir(INPUT_ROOT))
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        
        if not image_files:
            print("No supported images found in input_papers folder.")
            return
            
        current_student = None
        unknown_counter = 1
        
        for filename in image_files:
            path = os.path.join(INPUT_ROOT, filename)
            ocr_text = extract_text_cached(path)
            detected_roll = detect_roll_number(ocr_text)
            
            if detected_roll:
                print(f"Found Roll No '{detected_roll}' in {filename}. Starting new student group.")
                current_student = f"student_{detected_roll}"
                if current_student not in student_groups:
                    student_groups[current_student] = []
            elif current_student is None:
                current_student = f"unknown_student_{unknown_counter}"
                unknown_counter += 1
                student_groups[current_student] = []
                print(f"No Roll No detected in first scan {filename}. Defaulting to '{current_student}'.")
                
            student_groups[current_student].append((path, ocr_text))

    # Process all student groups
    print(f"\nProcessing total of {len(student_groups)} student(s)...")
    for student_name, page_list in student_groups.items():
        print(f"\n========== Grading {student_name} ({len(page_list)} page(s)) ==========")
        text, results = process_student(student_name, page_list, answer_key)
        save_results(student_name, text, results)
        print(f"Completed and saved scorecard for {student_name}")

if __name__ == "__main__":
    main()