import os
import re
import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger("HandwrittenPipeline.QuestionExtractor")

# Regex to detect common question labels at the start of a line/section
# Matches Q1, Q.1, Q 1, Q1a, Q1.a, Question 1, etc.
QUESTION_PATTERN = re.compile(
    r'(?:^[Qq][\s\.\)]*\s*(\d+)\s*[\.\)\-\>\s]*([a-zA-Z]?))|(?:\n[Qq][\s\.\)]*\s*(\d+)\s*[\.\)\-\>\s]*([a-zA-Z]?))'
)

def normalize_question_id(q_num: str, q_part: Optional[str] = None) -> str:
    """
    Standardizes question codes to uppercase, e.g. '1', 'a' -> 'Q1A'
    """
    part = q_part.strip().upper() if q_part else ""
    return f"Q{q_num}{part}"

def extract_questions_regex(text: str) -> Dict[str, str]:
    """
    Uses regex pattern matching to segment text question-wise.
    
    Args:
        text: Raw OCR transcription.
        
    Returns:
        A dictionary mapping standardized question keys (e.g. 'Q1', 'Q2') to answer text.
    """
    logger.info("Running regex-based question extraction...")
    
    # Split text by matching question patterns
    lines = text.splitlines()
    qa_dict = {}
    current_q = None
    current_ans_lines = []
    
    # Pattern to match question tag at the start of a line
    line_pattern = re.compile(r'^[Qq]\s*[\.\)]*\s*(\d+)\s*[\.\)\-\>\s]*([a-zA-Z]?)(?![a-zA-Z])')
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        match = line_pattern.match(stripped)
        if match:
            # Save the previous question's answer before starting a new one
            if current_q:
                qa_dict[current_q] = "\n".join(current_ans_lines).strip()
                logger.debug(f"Regex extracted {current_q} -> {len(qa_dict[current_q])} chars")
            
            qnum = match.group(1)
            qpart = match.group(2)
            current_q = normalize_question_id(qnum, qpart)
            current_ans_lines = []
            
            # Remove the matched question header from the first line of the answer
            remaining = stripped[match.end():].strip()
            remaining = re.sub(r'^[\.\)\-\>\s]+', '', remaining)
            if remaining:
                current_ans_lines.append(remaining)
        else:
            if current_q:
                current_ans_lines.append(stripped)
                
    # Save the final question
    if current_q:
        qa_dict[current_q] = "\n".join(current_ans_lines).strip()
        logger.debug(f"Regex extracted final question {current_q} -> {len(qa_dict[current_q])} chars")
        
    return qa_dict

def extract_questions_llm(
    text: str,
    expected_questions: List[str],
    api_key: str,
    model: str = "mistral-large-latest",
    temperature: float = 0.1
) -> Dict[str, str]:
    """
    Uses Mistral LLM to intelligently segment raw OCR text into question-wise answers.
    Useful when transcription is messy or regex fails.

    Args:
        text: Raw OCR transcription.
        expected_questions: List of question keys expected (e.g. ['Q1', 'Q2']).
        api_key: Mistral API key.
        model: Mistral model name.
        temperature: Temperature for sampling.

    Returns:
        A dictionary mapping question keys to extracted answer text.
    """
    logger.info(f"Running LLM-based question extraction using {model}...")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1"
    )
    
    prompt = f"""
    You are an expert OCR parser. Your task is to segment a messy handwritten answer sheet transcription into individual student answers for specific questions.

    Expected Question Keys: {expected_questions}

    Raw Transcribed Text:
    ---
    {text}
    ---

    Instructions:
    1. Parse the raw text and map the student's handwritten answer text to the correct expected question keys.
    2. The handwritten text might have typos in question headings (e.g. "Cl" instead of "Q1", "Q. 2." instead of "Q2"), or lack clear numbering. Use contextual clues to align the text with the correct question.
    3. Return a clean JSON object where the keys are the EXACT expected question keys, and the values are the student's answers.
    4. Do not include any explanation or markdown formatting (like ```json ... ```) outside the JSON. Return ONLY the valid JSON object.
    5. If a question is not answered or missing in the text, map its value to an empty string ("").

    Required Output JSON Schema:
    {{
      {", ".join([f'"{q}": "<extracted answer text>"' for q in expected_questions])}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional structured data extractor. You must output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        
        res_text = response.choices[0].message.content.strip()
        data = json.loads(res_text)
        
        # Clean keys and ensure all expected keys exist in the output dict
        result = {}
        for q in expected_questions:
            result[q] = str(data.get(q, "")).strip()
            
        return result
    except Exception as e:
        logger.error(f"LLM question extraction failed: {e}", exc_info=True)
        raise

class QuestionExtractor:
    """
    Orchestrates regex and/or LLM question extraction based on config.
    """
    def __init__(
        self,
        method: str = "hybrid",
        expected_questions: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        llm_model: str = "mistral-large-latest",
        temperature: float = 0.1
    ):
        self.method = method.lower()
        self.expected_questions = expected_questions or []
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.llm_model = llm_model
        self.temperature = temperature

    def extract(self, text: str) -> Dict[str, str]:
        """
        Extracts questions from OCR text using the configured method.
        """
        if self.method == "regex":
            return extract_questions_regex(text)
            
        if self.method == "llm":
            if not self.api_key:
                raise ValueError("Mistral API key is required for LLM question extraction.")
            return extract_questions_llm(
                text, self.expected_questions, self.api_key, self.llm_model, self.temperature
            )
            
        if self.method == "hybrid":
            # Hybrid mode: Try regex first.
            logger.info("Hybrid Mode: Attempting regex extraction first...")
            regex_results = extract_questions_regex(text)
            
            # Check if we successfully found all expected questions and they are not empty.
            # If we miss expected questions, or get empty answers for all, we fallback to LLM.
            has_all_expected = all(q in regex_results and regex_results[q].strip() for q in self.expected_questions)
            
            if has_all_expected:
                logger.info("Regex extraction successful for all expected questions.")
                return {q: regex_results[q] for q in self.expected_questions}
            
            # Fallback to LLM if regex missed some expected answers
            logger.info("Regex missed some expected questions or yielded empty answers. Falling back to LLM extraction...")
            if not self.api_key:
                logger.warning("Mistral API key not found. Proceeding with partial regex results.")
                # Return whatever regex found, filling missing with empty string
                return {q: regex_results.get(q, "") for q in self.expected_questions}
                
            try:
                llm_results = extract_questions_llm(
                    text, self.expected_questions, self.api_key, self.llm_model, self.temperature
                )
                return llm_results
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}. Falling back to regex results.", exc_info=True)
                return {q: regex_results.get(q, "") for q in self.expected_questions}
                
        raise ValueError(f"Unknown extraction method: {self.method}")
