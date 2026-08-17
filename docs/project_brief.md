# Project Brief

## 1. Project Summary

### What is this project?

This is an AI-powered Automated Answer Sheet Evaluation System that grades handwritten student answers automatically.

**The pipeline:**
1. Takes scanned handwritten answer sheets
2. Uses OCR (Pixtral Vision Model) to extract text
3. Splits answers question-wise
4. Compares student answers with model answers
5. Computes semantic similarity scores using BERT/SBERT
6. Uses an LLM (Mistral Large) to assign marks and feedback
7. Generates evaluation reports and analytics plots
8. Benchmarks grading performance against teacher-assigned marks

### Problem It Solves

Manual evaluation of descriptive answer sheets is:
- Time consuming
- Inconsistent across evaluators
- Difficult to scale

This project automates descriptive answer grading using modern NLP and LLMs.

### Main Features
- Handwritten answer sheet OCR
- Automatic answer key extraction from images
- Semantic answer comparison
- BERTScore evaluation
- SBERT similarity scoring
- LLM-based grading
- Detailed feedback generation
- Evaluation benchmarking against teacher marks
- Checkpointing and resume support
- OCR result caching
- Analytics and visualization generation

---

## 2. Technical Details

### Languages
- Python

### Libraries Used

**AI / NLP:**
- bert-score
- sentence-transformers
- OpenAI SDK (used with Mistral API)

**Data Processing:**
- pandas
- numpy

**Statistics:**
- scipy

**ML Metrics:**
- scikit-learn

**Visualization:**
- matplotlib
- seaborn

### External AI Models

| Component | Model |
|---|---|
| OCR | Pixtral Large |
| LLM Grader | Mistral Large |
| Embedding Model | all-MiniLM-L6-v2 |
| Similarity Metrics | BERTScore |

### Storage
- CSV, JSON, File-based storage (no database)

### Hardware
- No special hardware required. Can run on laptop, workstation, cloud VM, or Kaggle.

### APIs
- **Mistral API** — Used for OCR and LLM grading

---

## 3. Architecture

### High-Level Flow

```
Handwritten Answer Sheet
            │
            ▼
     Pixtral OCR
            │
            ▼
 Extracted Text
            │
            ▼
 Question Detection
            │
            ▼
 Student Answers
            │
            ├─────────────┐
            ▼             ▼
      BERTScore      SBERT Similarity
            │             │
            └──────┬──────┘
                   ▼
             Mistral LLM
                   │
                   ▼
      Marks + Feedback
                   │
                   ▼
         Scorecard Report
                   │
                   ▼
      Evaluation Analytics
```

### Important Modules

| Module | Responsibility |
|---|---|
| `ocr_pixtral.py` | Converts image → text using Pixtral OCR |
| `create_answer_key.py` | Reads answer key image, generates structured JSON |
| `evaluator.py` | Core grading engine: BERTScore, SBERT similarity, LLM grading, feedback generation |
| `pipeline.py` | Main orchestration: OCR caching, roll number detection, answer segmentation, scorecard generation |
| `test_dataset.py` | Research/benchmarking: large-scale evaluation, teacher marks comparison, correlation analysis, plot generation |

---

## 4. Benchmarking

The system was benchmarked on a dataset of **4,274 evaluated answers**, comparing AI-assigned grades against teacher-assigned marks.

Metrics computed:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE) / RMSE
- Pearson Correlation
- Spearman Correlation
- SBERT Score Correlation
- BERTScore Correlation
