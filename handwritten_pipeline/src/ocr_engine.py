import os
import base64
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from openai import OpenAI

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

logger = logging.getLogger("HandwrittenPipeline.OCREngine")

class OCREngine(ABC):
    """
    Abstract Base Class for OCR engines.
    """
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """
        Transcribes handwritten text from a given image.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            The transcribed text as a string.
        """
        pass


class PixtralOCREngine(OCREngine):
    """
    OCR Engine using Mistral's Pixtral cloud vision model.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable or argument is required for Pixtral OCR.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.mistral.ai/v1"
        )
        self.model = model

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    def extract_text(self, image_path: str) -> str:
        logger.info(f"Running Pixtral OCR on {os.path.basename(image_path)} using model {self.model}...")
        try:
            image_base64 = self._image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract all handwritten text from this image. "
                                    "Preserve line breaks and spacing as closely as possible. "
                                    "Do not add explanations."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            text = response.choices[0].message.content.strip()
            # Normalize common question variations (same as original pipeline)
            text = text.replace("Ql", "Q1").replace("Q l", "Q1")
            return text
        except Exception as e:
            logger.error(f"Pixtral OCR failed: {e}", exc_info=True)
            raise


class MistralOCREngine(OCREngine):
    """
    OCR Engine using Mistral's dedicated Document OCR API (mistral-ocr-latest).
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-ocr-latest"):
        if Mistral is None:
            raise ImportError(
                "mistralai is not installed. "
                "Run 'pip install mistralai' to use the Mistral OCR API."
            )
            
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable or argument is required for Mistral OCR.")
            
        self.client = Mistral(api_key=self.api_key)
        self.model = model

    def extract_text(self, image_path: str) -> str:
        logger.info(f"Running Mistral Document OCR on {os.path.basename(image_path)} using model {self.model}...")
        uploaded_file = None
        try:
            # 1. Upload local file to Mistral storage
            with open(image_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(image_path),
                        "content": f,
                    },
                    purpose="ocr"
                )
            logger.debug(f"Uploaded file ID for OCR: {uploaded_file.id}")

            # 2. Call OCR process
            response = self.client.ocr.process(
                model=self.model,
                document={"file_id": uploaded_file.id}
            )
            
            # 3. Concatenate markdown pages
            pages_text = [page.markdown for page in response.pages]
            text = "\n".join(pages_text).strip()
            
            # Normalize common question variations
            text = text.replace("Ql", "Q1").replace("Q l", "Q1")
            return text
            
        except Exception as e:
            logger.error(f"Mistral Document OCR failed: {e}", exc_info=True)
            raise
        finally:
            # 4. Clean up uploaded file
            if uploaded_file:
                try:
                    logger.debug(f"Deleting uploaded OCR file {uploaded_file.id}...")
                    self.client.files.delete(file_id=uploaded_file.id)
                except Exception as del_err:
                    logger.warning(f"Failed to delete uploaded OCR file {uploaded_file.id}: {del_err}")


class EasyOCREngine(OCREngine):
    """
    Offline OCR Engine using EasyOCR.
    """
    def __init__(self, languages: Optional[List[str]] = None):
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr is not installed. Run 'pip install easyocr' to use EasyOCR.")
            
        langs = languages or ['en']
        logger.info(f"Initializing EasyOCR reader for languages: {langs}...")
        self.reader = easyocr.Reader(langs)

    def extract_text(self, image_path: str) -> str:
        logger.info(f"Running EasyOCR on {os.path.basename(image_path)}...")
        try:
            results = self.reader.readtext(image_path, detail=0)
            return "\n".join(results)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}", exc_info=True)
            raise


class PaddleOCREngine(OCREngine):
    """
    Offline OCR Engine using PaddleOCR.
    """
    def __init__(self, languages: Optional[List[str]] = None):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr is not installed. "
                "Run 'pip install paddleocr paddlepaddle' (or paddlepaddle-gpu) to use PaddleOCR."
            )
            
        lang = languages[0] if languages else 'en'
        logger.info(f"Initializing PaddleOCR for language: {lang}...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang, show_log=False)

    def extract_text(self, image_path: str) -> str:
        logger.info(f"Running PaddleOCR on {os.path.basename(image_path)}...")
        try:
            results = self.ocr.ocr(image_path)
            all_text = []
            if results and results[0]:
                for line in results[0]:
                    text = line[1][0]
                    all_text.append(text)
            return "\n".join(all_text)
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}", exc_info=True)
            raise


class OCREngineWithCache(OCREngine):
    """
    Decorator class that adds file-hash-based caching to any OCREngine.
    """
    def __init__(self, engine: OCREngine, cache_dir: str = ".ocr_cache"):
        self.engine = engine
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_image_hash(self, image_path: str) -> str:
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def extract_text(self, image_path: str) -> str:
        try:
            img_hash = self._get_image_hash(image_path)
            cache_file = os.path.join(self.cache_dir, f"{img_hash}.txt")
            
            if os.path.exists(cache_file):
                logger.info(f"[Cache Hit] Using cached OCR for {os.path.basename(image_path)}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
                    
            logger.info(f"[Cache Miss] Running OCR engine on {os.path.basename(image_path)}...")
            text = self.engine.extract_text(image_path)
            
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
                
            return text
        except Exception as e:
            logger.error(f"Cached OCR wrapper encountered error: {e}", exc_info=True)
            # Fallback to direct extraction if caching fails
            return self.engine.extract_text(image_path)


def get_ocr_engine(engine_type: str, cache_dir: Optional[str] = None, **kwargs) -> OCREngine:
    """
    Factory function to retrieve a configured OCR engine.
    """
    engine_type = engine_type.lower()
    
    if engine_type == "pixtral":
        engine = PixtralOCREngine(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "mistral-large-latest")
        )
    elif engine_type == "mistral-ocr":
        engine = MistralOCREngine(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "mistral-ocr-latest")
        )
    elif engine_type == "easyocr":
        engine = EasyOCREngine(languages=kwargs.get("languages"))
    elif engine_type == "paddleocr":
        engine = PaddleOCREngine(languages=kwargs.get("languages"))
    else:
        raise ValueError(f"Unsupported OCR engine type: {engine_type}")
        
    if cache_dir:
        return OCREngineWithCache(engine, cache_dir=cache_dir)
        
    return engine
