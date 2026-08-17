import os
import re
import time
import yaml
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

from src.pdf_processor import pdf_to_images, enhance_image
from src.ocr_engine import get_ocr_engine
from src.question_extractor import QuestionExtractor
from src.grading_adapter import GradingAdapter

logger = logging.getLogger("HandwrittenPipeline.BatchRunner")

def detect_roll_number(text: str) -> Optional[str]:
    """
    Detects a student roll number or ID from the text.
    Duplicated exactly from final_pipeline/pipeline.py to maintain alignment.
    """
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

class BatchRunner:
    """
    Orchestrator for the end-to-end evaluation pipeline of handwritten student PDFs.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        logger.info("Initializing BatchRunner components...")
        self.paths = self.config.get("paths", {})
        self.pdf_settings = self.config.get("pdf_preprocessing", {})
        self.ocr_settings = self.config.get("ocr", {})
        self.extraction_settings = self.config.get("question_extraction", {})
        self.grading_settings = self.config.get("grading", {})
        
        # Instantiate adapters and engines
        self.grading_adapter = GradingAdapter(
            answer_key_path=self.paths.get("answer_key_file"),
            final_pipeline_dir=self.paths.get("final_pipeline_dir")
        )
        
        # Configure expected questions from answer key
        self.expected_questions = self.grading_adapter.get_expected_questions()
        
        # OCR Engine setup (caching included)
        self.ocr_engine = get_ocr_engine(
            engine_type=self.ocr_settings.get("engine_type", "pixtral"),
            cache_dir=self.paths.get("cache_dir", ".ocr_cache"),
            languages=self.ocr_settings.get("languages", ["en"]),
            model=self.ocr_settings.get("model")
        )
        
        # Question extractor setup
        self.question_extractor = QuestionExtractor(
            method=self.extraction_settings.get("method", "hybrid"),
            expected_questions=self.expected_questions,
            llm_model=self.extraction_settings.get("llm_model", "mistral-large-latest"),
            temperature=self.extraction_settings.get("temperature", 0.1)
        )
        
        # Checkpointing file path
        self.checkpoint_path = os.path.join(self.paths.get("output_dir", "output_handwritten"), "batch_results.csv")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        output_dir = self.config.get("paths", {}).get("output_dir", "output_handwritten")
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = os.path.join(output_dir, "pipeline.log")
        
        # Setup clean root formatting
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging initialized. Output file: {log_file}")

    def run(self):
        """
        Executes the batch processing pipeline on all PDFs in the input directory.
        """
        input_dir = self.paths.get("input_dir", "student_pdfs")
        output_dir = self.paths.get("output_dir", "output_handwritten")
        
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory '{input_dir}' does not exist. Creating it.")
            os.makedirs(input_dir, exist_ok=True)
            print(f"\nPlease place student handwritten PDFs in '{input_dir}' and run again.\n")
            return

        # Find all PDF files
        pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")])
        if not pdf_files:
            logger.info("No PDF files found to process.")
            return

        logger.info(f"Found {len(pdf_files)} PDF(s) to process: {pdf_files}")
        
        # Load progress from checkpoint if it exists
        processed_pdfs = set()
        overall_results = []
        if os.path.exists(self.checkpoint_path):
            try:
                progress_df = pd.read_csv(self.checkpoint_path)
                if not progress_df.empty and "pdf_file" in progress_df.columns:
                    processed_pdfs = set(progress_df["pdf_file"].dropna().unique())
                    logger.info(f"Checkpoint loaded. Already processed {len(processed_pdfs)} PDF(s).")
                    overall_results = progress_df.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Error reading checkpoint file: {e}")

        # Processing loop
        for pdf_name in pdf_files:
            if pdf_name in processed_pdfs:
                logger.info(f"Skipping already processed PDF: {pdf_name}")
                continue

            pdf_path = os.path.join(input_dir, pdf_name)
            student_base_name = os.path.splitext(pdf_name)[0]
            student_out_dir = os.path.join(output_dir, student_base_name)
            
            logger.info(f"\n==================================================")
            logger.info(f"Processing PDF: {pdf_name}")
            logger.info(f"==================================================")
            
            try:
                # 1. PDF page extraction to images
                logger.info(f"[{pdf_name}] Starting PDF page extraction...")
                poppler_path = self.paths.get("poppler_path")
                dpi = self.pdf_settings.get("dpi", 300)
                
                page_images = pdf_to_images(
                    pdf_path=pdf_path,
                    output_dir=student_out_dir,
                    dpi=dpi,
                    poppler_path=poppler_path
                )
                logger.info(f"[{pdf_name}] Extracted {len(page_images)} page image(s).")

                # 2. Image enhancement
                if self.pdf_settings.get("enhance_images", True):
                    logger.info(f"[{pdf_name}] Enhancing page images for OCR...")
                    for img_path in page_images:
                        enhance_image(
                            image_path=img_path,
                            output_path=img_path,
                            contrast_factor=self.pdf_settings.get("contrast_factor", 1.5),
                            brightness_factor=self.pdf_settings.get("brightness_factor", 1.2)
                        )
                
                # 3. OCR Transcription with Rate Limit Handling & Caching
                logger.info(f"[{pdf_name}] Running OCR transcription...")
                page_texts = []
                backoffs = [20, 40, 60, 80, 100]
                
                for idx, img_path in enumerate(page_images, start=1):
                    success = False
                    text = ""
                    for attempt in range(6): # retry 6 times
                        try:
                            text = self.ocr_engine.extract_text(img_path)
                            success = True
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 5:
                                delay = backoffs[attempt]
                                logger.warning(f"Rate limit hit during OCR. Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                logger.error(f"OCR failed for page {idx} after retries: {e}")
                                break
                    
                    if not success:
                        logger.error(f"Could not transcribe page {idx} of {pdf_name}. Skipping page.")
                    
                    page_texts.append(text)
                    # Respect rate limits between pages
                    time.sleep(3)

                full_raw_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

                # 4. Roll Number / Student ID detection
                detected_roll = detect_roll_number(full_raw_text)
                if detected_roll:
                    student_id = f"student_{detected_roll}"
                    logger.info(f"[{pdf_name}] Detected roll number '{detected_roll}' in transcription.")
                else:
                    student_id = student_base_name
                    logger.warning(f"[{pdf_name}] No roll number detected. Falling back to ID: {student_id}")

                # 5. Question extraction
                logger.info(f"[{pdf_name}] Extracting answers...")
                student_answers = self.question_extractor.extract(full_raw_text)
                logger.info(f"[{pdf_name}] Answer extraction complete: {list(student_answers.keys())}")

                # 6. Grading with Rate Limit Handling
                logger.info(f"[{pdf_name}] Grading student answers via Adapter...")
                results = []
                
                for q, ans in student_answers.items():
                    success = False
                    eval_res_list = []
                    
                    for attempt in range(6):
                        try:
                            # Grade a dict containing just this single question
                            eval_res_list = self.grading_adapter.grade_student(student_id, {q: ans})
                            success = True
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 5:
                                delay = backoffs[attempt]
                                logger.warning(f"Rate limit hit during grading. Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                logger.error(f"Grading failed for {q} after retries: {e}")
                                break
                    
                    if success and eval_res_list:
                        results.extend(eval_res_list)
                    else:
                        # Dummy entry for failed questions
                        results.append({
                            "question": q,
                            "student_answer": ans,
                            "reference_answer": None,
                            "max_marks": 5,
                            "bert_score": 0.0,
                            "sbert_score": 0.0,
                            "llm_grade": 0.0,
                            "llm_feedback": "Grading failed due to API connection errors."
                        })
                    
                    # Respect rate limits between grading calls
                    time.sleep(5)

                # 7. Save outputs for the student
                self.grading_adapter.save_results(
                    output_dir=output_dir,
                    student_id=student_base_name,  # Save folder keeps filename base structure to avoid conflicts
                    raw_text=full_raw_text,
                    answers=student_answers,
                    results=results
                )
                logger.info(f"[{pdf_name}] Student processing and grading completed successfully.")

                # 8. Checkpoint progress updating
                for res in results:
                    overall_results.append({
                        "pdf_file": pdf_name,
                        "student_id": student_id,
                        "question": res["question"],
                        "student_answer": res["student_answer"],
                        "reference_answer": res["reference_answer"],
                        "max_marks": res["max_marks"],
                        "bert_score": res["bert_score"],
                        "sbert_score": res["sbert_score"],
                        "llm_grade": res["llm_grade"],
                        "llm_feedback": res["llm_feedback"]
                    })
                
                # Write to disk
                checkpoint_df = pd.DataFrame(overall_results)
                checkpoint_df.to_csv(self.checkpoint_path, index=False)
                processed_pdfs.add(pdf_name)
                logger.info(f"Checkpoint saved. Processed files count: {len(processed_pdfs)}")

            except Exception as e:
                logger.error(f"Failed to process student PDF {pdf_name}: {e}", exc_info=True)
                # Keep running other files in the directory even if one fails
                continue
                
        logger.info("\nBatch execution runner completed.")
        return overall_results
