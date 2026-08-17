import os
import unittest
from unittest.mock import patch, MagicMock
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.question_extractor import (
    extract_questions_regex,
    extract_questions_llm,
    QuestionExtractor,
    normalize_question_id
)

class TestQuestionExtractor(unittest.TestCase):
    def test_normalize_question_id(self):
        self.assertEqual(normalize_question_id("1"), "Q1")
        self.assertEqual(normalize_question_id("2", "a"), "Q2A")
        self.assertEqual(normalize_question_id("12", " B "), "Q12B")

    def test_extract_questions_regex_basic(self):
        text = """
        Q1.
        This is the answer to question 1.
        It has multiple lines.
        
        Q2)
        This is the answer to question 2.
        """
        results = extract_questions_regex(text)
        self.assertEqual(results.get("Q1"), "This is the answer to question 1.\nIt has multiple lines.")
        self.assertEqual(results.get("Q2"), "This is the answer to question 2.")

    def test_extract_questions_regex_subparts(self):
        text = """
        Q1a. Answers part A
        Q1b. Answers part B
        """
        results = extract_questions_regex(text)
        self.assertEqual(results.get("Q1A"), "Answers part A")
        self.assertEqual(results.get("Q1B"), "Answers part B")

    @patch('src.question_extractor.OpenAI')
    def test_extract_questions_llm(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"Q1": "Answer for Q1 extracted by LLM", "Q2": "Answer for Q2 extracted by LLM"}'
        mock_client.chat.completions.create.return_value = mock_response

        text = "Some messy text"
        expected = ["Q1", "Q2"]
        results = extract_questions_llm(text, expected, api_key="dummy", model="dummy_model")

        self.assertEqual(results["Q1"], "Answer for Q1 extracted by LLM")
        self.assertEqual(results["Q2"], "Answer for Q2 extracted by LLM")
        mock_client.chat.completions.create.assert_called_once()

    def test_extractor_hybrid_regex_success(self):
        text = "Q1. Ans 1\nQ2. Ans 2"
        extractor = QuestionExtractor(method="hybrid", expected_questions=["Q1", "Q2"])
        results = extractor.extract(text)
        
        self.assertEqual(results["Q1"], "Ans 1")
        self.assertEqual(results["Q2"], "Ans 2")

    @patch('src.question_extractor.extract_questions_llm')
    def test_extractor_hybrid_fallback(self, mock_llm_extract):
        mock_llm_extract.return_value = {"Q1": "LLM 1", "Q2": "LLM 2"}
        
        # Regex only finds Q1, but Q2 is expected
        text = "Q1. Ans 1\nSome unrecognized text here"
        
        extractor = QuestionExtractor(
            method="hybrid", 
            expected_questions=["Q1", "Q2"],
            api_key="dummy_key"
        )
        results = extractor.extract(text)
        
        self.assertEqual(results["Q1"], "LLM 1")
        self.assertEqual(results["Q2"], "LLM 2")
        mock_llm_extract.assert_called_once()

if __name__ == '__main__':
    unittest.main()
