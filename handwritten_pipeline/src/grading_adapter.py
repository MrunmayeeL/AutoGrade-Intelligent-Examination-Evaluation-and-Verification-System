import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger("HandwrittenPipeline.GradingAdapter")

class GradingAdapter:
    """
    Adapter to interface between the new handwritten pipeline and the existing
    read-only grading pipeline (final_pipeline).
    """
    def __init__(self, answer_key_path: str, final_pipeline_dir: Optional[str] = None):
        self.answer_key_path = os.path.abspath(answer_key_path)
        
        if final_pipeline_dir:
            self.final_pipeline_dir = os.path.abspath(final_pipeline_dir)
        else:
            # Check if answer key folder contains evaluator.py
            parent_dir = os.path.dirname(self.answer_key_path)
            if os.path.exists(os.path.join(parent_dir, "evaluator.py")):
                self.final_pipeline_dir = parent_dir
            else:
                # Fallback: ../final_pipeline relative to this script
                self.final_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../final_pipeline"))
        
        # Add final_pipeline to sys.path so we can import its modules
        if self.final_pipeline_dir not in sys.path:
            logger.info(f"Adding final_pipeline directory to sys.path: {self.final_pipeline_dir}")
            sys.path.append(self.final_pipeline_dir)
            
        # Dynamically import the core evaluate_single function from evaluator.py
        try:
            from evaluator import evaluate_single
            self.evaluate_single = evaluate_single
            logger.info("Successfully imported evaluate_single from final_pipeline.evaluator")
        except ImportError as e:
            logger.error(f"Failed to import evaluate_single from final_pipeline: {e}")
            raise ImportError(
                "Could not import final_pipeline modules. Please check the answer_key_file path in the config."
            ) from e

        # Load answer key using clean normalization logic matching final_pipeline/pipeline.py
        self.answer_key = self._load_answer_key()

    def _normalize_q(self, q: str) -> str:
        q = q.replace(".", "").replace(")", "")
        q = q.replace(" ", "")
        return q.upper()

    def _load_answer_key(self) -> Dict[str, Dict[str, Any]]:
        logger.info(f"Loading answer key from {self.answer_key_path}...")
        if not os.path.exists(self.answer_key_path):
            raise FileNotFoundError(f"Answer key file not found at: {self.answer_key_path}")
            
        with open(self.answer_key_path, "r", encoding="utf-8") as f:
            raw_key = json.load(f)
            
        answer_key = {}
        for q, val in raw_key.items():
            q_clean = self._normalize_q(q)
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
        logger.info(f"Loaded {len(answer_key)} questions from the answer key.")
        return answer_key

    def get_expected_questions(self) -> List[str]:
        """
        Returns list of standardized expected questions from the answer key.
        """
        return list(self.answer_key.keys())

    def grade_student(self, student_id: str, answers: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Grades the student's structured answers by calling evaluate_single.

        Args:
            student_id: Unique student ID or roll number.
            answers: Dict mapping question codes to student answer strings.

        Returns:
            A list of dictionary results, one for each question.
        """
        logger.info(f"Grading student: {student_id}...")
        results = []
        
        for q, student_ans in answers.items():
            q_clean = self._normalize_q(q)
            ref_info = self.answer_key.get(q_clean)
            
            if ref_info:
                ref_ans = ref_info["answer"]
                max_m = ref_info["max_marks"]
                
                logger.info(f"  Grading question {q_clean} (Max Marks: {max_m})...")
                # Call original evaluator module
                eval_res = self.evaluate_single(q_clean, ref_ans, student_ans, max_m)
                
                bert_sc = eval_res["bert_score"]
                sbert_sc = eval_res["sbert_score"]
                grade = eval_res["llm_grade"]
                feedback = eval_res["llm_feedback"]
            else:
                # Fallback if question was not in the answer key
                logger.warning(f"  Question {q_clean} not found in the answer key. Assigning 0 marks.")
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
            
        logger.info(f"Grading complete for student: {student_id}.")
        return results

    def save_results(self, output_dir: str, student_id: str, raw_text: str, answers: Dict[str, str], results: List[Dict[str, Any]]) -> str:
        """
        Saves student text, structured answers, results JSON, and Markdown scorecard.
        """
        student_out = os.path.join(output_dir, student_id)
        os.makedirs(student_out, exist_ok=True)
        
        # Save raw transcribed OCR text
        with open(os.path.join(student_out, "ocr_raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
            
        # Save structured answers
        with open(os.path.join(student_out, "student_answers.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4)
            
        # Save grading results JSON
        with open(os.path.join(student_out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        # Generate Markdown Scorecard by importing the logic or replicating the format
        # Since we must keep the look and feel identical to the original project, we replicate
        # the scorecard format from pipeline.py
        scorecard_md = self.generate_scorecard_md(student_id, results)
        with open(os.path.join(student_out, "scorecard.md"), "w", encoding="utf-8") as f:
            f.write(scorecard_md)
            
        logger.info(f"Saved all results for {student_id} to {student_out}")
        return student_out

    def generate_scorecard_md(self, student_id: str, results: List[Dict[str, Any]]) -> str:
        """
        Replicates the exact scorecard formatting from final_pipeline/pipeline.py
        """
        total_score = sum(r["llm_grade"] for r in results if r["llm_grade"] is not None)
        total_max = sum(r["max_marks"] for r in results if r["max_marks"] is not None)
        percentage = (total_score / total_max * 100) if total_max > 0 else 0
        
        md = f"""# Academic Performance Scorecard

**Student Identifier**: `{student_id}`  
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
