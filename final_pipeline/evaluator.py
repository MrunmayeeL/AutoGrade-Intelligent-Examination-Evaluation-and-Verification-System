import os
import json
from openai import OpenAI

api_key = os.getenv(
    "MISTRAL_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "MISTRAL_API_KEY not found"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"
)

# -----------------------------
# SIMILARITY SCORING (BERT)
# -----------------------------
bert_scorer = None

def get_bert_scorer():
    global bert_scorer
    if bert_scorer is None:
        print("Loading local BERTScore model...")
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(lang="en")
    return bert_scorer

def compute_bert_score(candidate, reference):
    if not candidate or not reference:
        return 0.0
    try:
        scorer = get_bert_scorer()
        P, R, F1 = scorer.score([candidate], [reference])
        return float(F1[0])
    except Exception as e:
        print(f"Error computing BERT score: {e}")
        return 0.0

# -----------------------------
# SIMILARITY SCORING (SBERT)
# -----------------------------
sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        print("Loading local SBERT model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model

def compute_sbert_score(candidate, reference):
    if not candidate or not reference:
        return 0.0
    try:
        from sentence_transformers import util
        model = get_sbert_model()
        emb1 = model.encode(candidate, convert_to_tensor=True)
        emb2 = model.encode(reference, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2)
        return float(similarity[0][0])
    except Exception as e:
        print(f"Error computing SBERT score: {e}")
        return 0.0

# -----------------------------
# LLM EVALUATION
# -----------------------------
def grade_answer_with_llm(question, ideal, student_ans, max_marks):
    if not student_ans or str(student_ans).strip() == "":
        return {
            "grade": 0.0,
            "max_marks": max_marks,
            "feedback": "Blank or empty answer."
        }
        
    prompt = f"""
    You are an expert academic evaluator. Grade the student's answer based on the reference (ideal) answer.

    Question: {question}
    Reference Answer: {ideal}
    Student's Answer: {student_ans}
    Maximum Marks Allotted: {max_marks}

    Provide a fair numeric grade (marks) between 0 and {max_marks} (can be a decimal like 4.5 or integer like 5), where:
    - {max_marks}: Perfectly correct, complete, and covers all key points.
    - {max_marks * 0.8}: Mostly correct, covers key points but lacks minor details.
    - {max_marks * 0.6}: Partially correct, misses some key points or has minor errors.
    - {max_marks * 0.4}: Has some relevance but contains major errors or misses core concepts.
    - {max_marks * 0.2}: Very poor, barely relevant.
    - 0: Completely incorrect, irrelevant, or blank.

    Also provide brief, constructive feedback explaining the grade and what was missing or incorrect.

    Return a JSON object with the following fields:
    {{
      "grade": <numeric score between 0 and {max_marks}>,
      "max_marks": {max_marks},
      "feedback": "<brief explanation of the grade>"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are a professional academic grader. You must output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        res_text = response.choices[0].message.content.strip()
        data = json.loads(res_text)
        return {
            "grade": float(data.get("grade", 0)),
            "max_marks": int(data.get("max_marks", max_marks)),
            "feedback": str(data.get("feedback", "No feedback provided."))
        }
    except Exception as e:
        if "429" in str(e):
            raise e
        print(f"Error grading answer with LLM: {e}")
        return {
            "grade": 0.0,
            "max_marks": max_marks,
            "feedback": f"Failed to grade answer via LLM: {e}"
        }

# -----------------------------
# CORE SINGLE-ANSWER EVALUATOR
# -----------------------------
def evaluate_single(question, ideal_ans, student_ans, max_marks=5):
    """
    Orchestrates semantic similarity scores and LLM grading for a single answer.
    """
    bert_sc = compute_bert_score(student_ans, ideal_ans)
    sbert_sc = compute_sbert_score(student_ans, ideal_ans)
    llm_eval = grade_answer_with_llm(question, ideal_ans, student_ans, max_marks)
    
    return {
        "bert_score": bert_sc,
        "sbert_score": sbert_sc,
        "llm_grade": llm_eval["grade"],
        "llm_feedback": llm_eval["feedback"]
    }
