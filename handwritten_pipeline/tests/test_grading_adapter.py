import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock evaluate_single before importing GradingAdapter to prevent importing the actual final_pipeline models
sys.modules['evaluator'] = MagicMock()
from src.grading_adapter import GradingAdapter

class TestGradingAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock answer key file
        self.answer_key_data = {
            "Q1": {
                "answer": "AI is human intelligence in machines.",
                "max_marks": 7
            },
            "Q2": "Machine Learning learns from data."
        }
        self.answer_key_path = os.path.join(self.temp_dir.name, "answer_key.json")
        with open(self.answer_key_path, "w") as f:
            json.dump(self.answer_key_data, f)
            
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('src.grading_adapter.GradingAdapter._load_answer_key')
    def test_adapter_init(self, mock_load):
        mock_load.return_value = {"Q1": {"answer": "A", "max_marks": 5}}
        adapter = GradingAdapter(self.answer_key_path)
        self.assertEqual(adapter.answer_key_path, os.path.abspath(self.answer_key_path))
        mock_load.assert_called_once()

    def test_load_answer_key_parsing(self):
        adapter = GradingAdapter(self.answer_key_path)
        
        # Test normalized keys
        self.assertIn("Q1", adapter.answer_key)
        self.assertIn("Q2", adapter.answer_key)
        
        # Check details
        self.assertEqual(adapter.answer_key["Q1"]["max_marks"], 7)
        self.assertEqual(adapter.answer_key["Q2"]["max_marks"], 5) # Default value
        self.assertEqual(adapter.answer_key["Q2"]["answer"], "Machine Learning learns from data.")

    def test_grade_student_mocked(self):
        adapter = GradingAdapter(self.answer_key_path)
        
        # Mock evaluate_single behavior
        adapter.evaluate_single = MagicMock(return_value={
            "bert_score": 0.85,
            "sbert_score": 0.90,
            "llm_grade": 4.5,
            "llm_feedback": "Excellent answer."
        })
        
        student_answers = {
            "Q1": "Artificial Intelligence is human intelligence replication in software.",
            "Q2": "ML learns from data directly."
        }
        
        results = adapter.grade_student("student123", student_answers)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["question"], "Q1")
        self.assertEqual(results[0]["bert_score"], 0.85)
        self.assertEqual(results[0]["llm_grade"], 4.5)
        
        # Verify evaluate_single was called twice
        self.assertEqual(adapter.evaluate_single.call_count, 2)

if __name__ == '__main__':
    unittest.main()
