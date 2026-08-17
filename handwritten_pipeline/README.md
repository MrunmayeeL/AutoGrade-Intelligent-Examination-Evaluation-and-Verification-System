# Handwritten Answer Sheet Evaluation Pipeline

This is an end-to-end preprocessing extension to the smart answer evaluation system. It takes scanned student answer sheet PDFs, performs page preprocessing, applies pluggable OCR to transcribe handwriting, extracts questions into a structured format, passes them through the validated grading pipeline, and generates scorecards and evaluation metrics.

> [!IMPORTANT]
> This pipeline interacts with the existing grading pipeline in `final_pipeline` entirely as a read-only external library. **No files in the original grading project are modified.**

---

## Features
- **PDF-to-Image Processing**: Converts multi-page handwritten student PDFs into high-resolution images.
- **Image Enhancement**: Adjusts contrast and brightness to optimize OCR performance.
- **Pluggable OCR**: Supports Mistral's **Pixtral** (default cloud model), **PaddleOCR**, and **EasyOCR** (local/offline).
- **Hybrid Question Extraction**: Splits raw OCR transcripts question-wise using regex with an intelligent LLM fallback parser.
- **Grading Adapter**: Seamlessly feeds structured answers to the untouched core grading library.
- **Batch Processing & Checkpointing**: Runs evaluations across large directories of PDFs with progress backup and interruption recovery.
- **Analytics & Report Generation**: Calculates MAE, MSE, correlation metrics and outputs visual reports.

---

## Installation

### 1. Python Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. System Dependency: Poppler
The `pdf2image` library requires **poppler** to convert PDFs to images.

- **Linux (Debian/Ubuntu)**:
  ```bash
  sudo apt-get install poppler-utils
  ```
- **macOS**:
  ```bash
  brew install poppler
  ```
- **Windows**:
  1. Download the latest poppler binary release (e.g. from `@oschwartz10612` on GitHub).
  2. Extract the ZIP folder.
  3. Either add the `bin/` directory path to your system's environment variable `PATH` OR configure `poppler_path` inside `configs/pipeline_config.yaml` to point to the extracted `bin/` directory.

### 3. Environment Variables
Ensure your Mistral API key is set up in your terminal environment:
- **Linux/macOS**:
  ```bash
  export MISTRAL_API_KEY="your-api-key-here"
  ```
- **Windows (PowerShell)**:
  ```powershell
  $env:MISTRAL_API_KEY="your-api-key-here"
  ```

---

## Configuration

All parameters are configured via `configs/pipeline_config.yaml`:
- **paths**: Input/output folder locations, answer key JSON, and caching paths.
- **pdf_preprocessing**: Image enhancement properties (DPI, contrast, brightness).
- **ocr**: Option to choose `pixtral` (cloud) or `paddleocr` / `easyocr` (local/offline).
- **question_extraction**: Segmentation strategy (`regex`, `llm`, or `hybrid`).

---

## Usage

Place the scanned student answer sheet PDFs in the configured input folder (default: `student_pdfs/`).
Place the answer key in the configured answer key path (default: `../final_pipeline/answer_key.json`).

Run the pipeline:
```bash
python run_pipeline.py
```

### Output Files
For each processed student, the pipeline generates inside the `output_handwritten/<student_id>/` directory:
- `page_X.png`: Preprocessed images of PDF pages.
- `ocr_raw.txt`: The raw transcribed text.
- `student_answers.json`: Segmented question-wise responses.
- `result.json`: Detailed grading results (BERTScore, SBERT, LLM grade).
- `scorecard.md`: Beautiful formatted scorecard markdown report.

It also generates overall metric plots and correlation logs under `output_handwritten/` if validation is enabled.
