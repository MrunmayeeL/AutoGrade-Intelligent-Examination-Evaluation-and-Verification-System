import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pdf_processor import pdf_to_images, enhance_image

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('src.pdf_processor.convert_from_path')
    def test_pdf_to_images_mocked(self, mock_convert):
        # Create a mock PIL image
        mock_img = Image.new('RGB', (100, 100), color='white')
        mock_convert.return_value = [mock_img, mock_img]
        
        pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        # Touch mock PDF file
        with open(pdf_path, 'w') as f:
            f.write("mock pdf content")
            
        output_dir = os.path.join(self.temp_dir.name, "output_pages")
        
        image_paths = pdf_to_images(pdf_path, output_dir, dpi=300)
        
        # Assertions
        self.assertEqual(len(image_paths), 2)
        self.assertTrue(os.path.exists(image_paths[0]))
        self.assertTrue(os.path.exists(image_paths[1]))
        self.assertTrue(image_paths[0].endswith("page_1.png"))
        
        mock_convert.assert_called_once_with(pdf_path, dpi=300, poppler_path=None)

    def test_enhance_image(self):
        # Create a test image
        img_path = os.path.join(self.temp_dir.name, "test_img.png")
        img = Image.new('RGB', (50, 50), color='red')
        img.save(img_path)
        
        output_path = os.path.join(self.temp_dir.name, "enhanced_img.png")
        result_path = enhance_image(img_path, output_path, contrast_factor=1.2, brightness_factor=1.1)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify it can be loaded
        with Image.open(output_path) as enhanced_img:
            self.assertEqual(enhanced_img.size, (50, 50))

if __name__ == '__main__':
    unittest.main()
