import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ocr_engine import get_ocr_engine, OCREngineWithCache, PixtralOCREngine, MistralOCREngine

class TestOCREngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        # Close any open file handlers to release file lock on Windows
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

    @patch('src.ocr_engine.OpenAI')
    def test_pixtral_engine_init(self, mock_openai):
        os.environ["MISTRAL_API_KEY"] = "dummy_key"
        engine = get_ocr_engine("pixtral")
        self.assertIsInstance(engine, PixtralOCREngine)
        mock_openai.assert_called_once()

    @patch('src.ocr_engine.OpenAI')
    def test_cached_ocr_engine(self, mock_openai):
        # Set up mock client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Transcribed handwritten Q1 answer."
        mock_client.chat.completions.create.return_value = mock_response

        os.environ["MISTRAL_API_KEY"] = "dummy_key"
        
        # Test image path
        img_path = os.path.join(self.temp_dir.name, "test_img.png")
        with open(img_path, 'wb') as f:
            f.write(b"fake image data")

        # Instantiate engine with cache
        engine = get_ocr_engine("pixtral", cache_dir=self.temp_dir.name)
        self.assertIsInstance(engine, OCREngineWithCache)

        # First run: should be a Cache Miss
        text1 = engine.extract_text(img_path)
        self.assertEqual(text1, "Transcribed handwritten Q1 answer.")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

        # Second run: should be a Cache Hit (no API call)
        text2 = engine.extract_text(img_path)
        self.assertEqual(text2, "Transcribed handwritten Q1 answer.")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1) # Still 1

    @patch('src.ocr_engine.Mistral')
    def test_mistral_ocr_engine_execution(self, mock_mistral_class):
        mock_client = MagicMock()
        mock_mistral_class.return_value = mock_client
        
        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file_12345"
        mock_client.files.upload.return_value = mock_file
        
        # Mock OCR processing response
        mock_page = MagicMock()
        mock_page.markdown = "Transcribed handwritten OS End Sem answer."
        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_client.ocr.process.return_value = mock_response
        
        os.environ["MISTRAL_API_KEY"] = "dummy_key"
        
        img_path = os.path.join(self.temp_dir.name, "test_img.png")
        with open(img_path, 'wb') as f:
            f.write(b"fake image data")
            
        engine = get_ocr_engine("mistral-ocr")
        self.assertIsInstance(engine, MistralOCREngine)
        text = engine.extract_text(img_path)
        
        self.assertEqual(text, "Transcribed handwritten OS End Sem answer.")
        mock_client.files.upload.assert_called_once()
        mock_client.ocr.process.assert_called_once_with(
            model="mistral-ocr-latest",
            document={"file_id": "file_12345"}
        )
        mock_client.files.delete.assert_called_once_with(file_id="file_12345")

if __name__ == '__main__':
    unittest.main()
