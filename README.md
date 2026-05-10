# Legal Text Chunking Pipeline

A hybrid legal document chunking pipeline designed to segment long-form legal and legislative documents into context-preserving chunks for downstream NLP and LLM applications.

---

## Overview

This project focuses on intelligent segmentation of legal and parliamentary documents using a hybrid approach combining:
- BERT-based boundary detection
- Rule-based segmentation
- Customized tokenization
- Sliding-window chunking

The pipeline is designed to preserve semantic and legal context while preparing documents for:
- Retrieval-Augmented Generation (RAG)
- Long-context LLMs
- Legal search systems
- Legal summarization

---

## Features

### Hybrid Chunking Architecture
- BERT-based chunk boundary detection
- Rule-based segmentation refinement
- Context-aware chunk generation
- Sliding-window overlap support

### Legal Document Processing
- Parliamentary bill processing
- Legislative text handling
- Legal judgment segmentation
- Long-form legal document support

### Text Processing
- Customized NLTK tokenization
- Regex-based title extraction
- Bill number recognition
- Semantic context preservation

### Structured Outputs
- JSONL chunk outputs
- Metadata-rich chunk records
- Reproducible preprocessing pipeline

---

## Pipeline Architecture

1. Load legal documents
2. Detect structural boundaries
3. Apply BERT segmentation
4. Perform rule-based refinement
5. Generate overlapping chunks
6. Export structured JSONL outputs

---

## Dataset Schema

Example chunk output:

```json
{
  "document_id": "bill_001",
  "chunk_id": 5,
  "text": "An Act to amend...",
  "start_position": 1200,
  "end_position": 1800
}
```

---

## Technologies Used

- Python
- Transformers
- BERT
- NLTK
- Regex
- JSONL

---

## Usage

```bash
python main.py
```

---

## Project Structure

```text
text_chunking/
├── data/
├── outputs/
├── models/
├── scripts/
├── main.py
└── README.md
```

---

## Applications

- Retrieval-Augmented Generation (RAG)
- Legal search systems
- Long-context LLMs
- Legislative document analysis
- Legal summarization

---

## Design Goals

- Preserve legal context
- Reduce semantic fragmentation
- Improve retrieval quality
- Support scalable preprocessing

---

## Future Improvements

- Hierarchical chunking
- Citation-aware segmentation
- Adaptive chunk sizing
- Semantic overlap optimization
