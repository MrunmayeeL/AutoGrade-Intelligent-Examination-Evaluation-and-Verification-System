import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sys
import yaml
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to prevent executing slow external network calls and imports
sys.modules['pdf2image'] = MagicMock()
sys.modules['evaluator'] = MagicMock()
from src.batch_runner import BatchRunner, detect_roll_number

class TestBatchRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock configurations
        self.config_data = {
            "paths": {
                "input_dir": os.path.join(self.temp_dir.name, "inputs"),
                "output_dir": os.path.join(self.temp_dir.name, "outputs"),
                "answer_key_file": os.path.join(self.temp_dir.name, "answer_key.json"),
                "cache_dir": os.path.join(self.temp_dir.name, "cache"),
                "poppler_path": None
            },
            "pdf_preprocessing": {
                "dpi": 150,
                "enhance_images": True,
                "contrast_factor": 1.2,
                "brightness_factor": 1.1
            },
            "ocr": {
                "engine_type": "pixtral",
                "languages": ["en"]
            },
            "question_extraction": {
                "method": "regex",
                "llm_model": "mistral-large-latest",
                "temperature": 0.1
            },
            "grading": {
                "run_evaluation": False,
                "ground_truth_csv": None
            }
        }
        
        # Write config YAML file
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)
            
        # Write mock answer key JSON file
        answer_key_data = {"Q1": "Ideal answer 1", "Q2": "Ideal answer 2"}
        with open(self.config_data["paths"]["answer_key_file"], "w") as f:
            import json
            json.dump(answer_key_data, f)
            
        # Create input dir and a mock student PDF
        os.makedirs(self.config_data["paths"]["input_dir"], exist_ok=True)
        self.mock_pdf_path = os.path.join(self.config_data["paths"]["input_dir"], "student1.pdf")
        with open(self.mock_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 mock pdf content")

    def tearDown(self):
        # Close and remove any active logging handlers to release file locks on Windows
        import logging
        for logger_name in logging.root.manager.loggerDict.keys():
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
        self.temp_dir.cleanup()

    def test_detect_roll_number(self):
        # Test basic patterns
        self.assertEqual(detect_roll_number("Roll No: 12345"), "12345")
        self.assertEqual(detect_roll_number("Candidate No-ABC-99"), "ABC-99")
        self.assertEqual(detect_roll_number("Roll number: BT101"), "BT101")
        self.assertEqual(detect_roll_number("student id 789"), "789")
        self.assertIsNone(detect_roll_number("This is just an answer without ID"))

    @patch('src.batch_runner.pdf_to_images')
    @patch('src.batch_runner.enhance_image')
    @patch('src.batch_runner.get_ocr_engine')
    @patch('src.batch_runner.QuestionExtractor')
    @patch('src.batch_runner.GradingAdapter')
    def test_batch_runner_execution_flow(
        self, mock_adapter_class, mock_extractor_class, mock_get_ocr, mock_enhance, mock_pdf_to_images
    ):
        # 1. Setup Mock Instances and Behaviors
        mock_pdf_to_images.return_value = ["/mock/path/page_1.png"]
        
        # OCR Mock
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = "Roll No: BT123\nQ1. Answer 1\nQ2. Answer 2"
        mock_get_ocr.return_value = mock_ocr
        
        # Extractor Mock
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"Q1": "Answer 1", "Q2": "Answer 2"}
        mock_extractor_class.return_value = mock_extractor
        
        # Adapter Mock
        mock_adapter = MagicMock()
        mock_adapter.get_expected_questions.return_value = ["Q1", "Q2"]
        
        def mock_grade_student(student_id, answers):
            res_map = {
                "Q1": {"question": "Q1", "student_answer": "Answer 1", "reference_answer": "Ideal 1", "max_marks": 5, "bert_score": 0.8, "sbert_score": 0.9, "llm_grade": 4.0, "llm_feedback": "Good"},
                "Q2": {"question": "Q2", "student_answer": "Answer 2", "reference_answer": "Ideal 2", "max_marks": 5, "bert_score": 0.7, "sbert_score": 0.8, "llm_grade": 3.5, "llm_feedback": "Fair"}
            }
            return [res_map[q] for q in answers if q in res_map]
            
        mock_adapter.grade_student.side_effect = mock_grade_student
        mock_adapter_class.return_value = mock_adapter

        # 2. Run BatchRunner
        runner = BatchRunner(self.config_path)
        results = runner.run()

        # 3. Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["student_id"], "student_BT123")
        self.assertEqual(results[0]["question"], "Q1")
        self.assertEqual(results[1]["llm_grade"], 3.5)

        # Check that checkpoint CSV is generated
        checkpoint_file = runner.checkpoint_path
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Read checkpoint and verify
        checkpoint_df = pd.read_csv(checkpoint_file)
        self.assertEqual(len(checkpoint_df), 2)
        self.assertEqual(checkpoint_df.iloc[0]["student_id"], "student_BT123")
        self.assertEqual(checkpoint_df.iloc[0]["pdf_file"], "student1.pdf")

if __name__ == '__main__':
    unittest.main()
