# Legal Document Semantic Chunking Pipeline

A hybrid legal document chunking pipeline designed to segment lengthy parliamentary bills into context-preserving semantic chunks and proposition-level sub-chunks for downstream retrieval, search, and legal AI applications.

---

## Overview

This project focuses on intelligent segmentation of parliamentary bills using a hybrid approach combining:

- BERT-based semantic boundary detection
- Sliding-window chunking with overlap support
- Rule-based segmentation refinement
- Customised tokenization for legal text

The pipeline is designed to preserve semantic and legal context while preparing structured outputs for:

- Retrieval-Augmented Generation (RAG)
- Legal search systems
- Long-context LLMs
- Legislative document analysis

---

## Pipeline Architecture

1. Load parliamentary bills stored in markdown format
2. Extract bill metadata using regex (bill title, bill number, section information)
3. Apply sentence segmentation using NLTK Punkt tokenizer with custom legal abbreviation handling
4. Detect semantic chunk boundaries using BERT-based chunking model (tim1900/bert-chunker-3)
5. Evaluate and select chunking threshold (evaluated 0.3, 0.5, 0.7 — selected 0.5)
6. Apply sliding-window chunking with overlap support
7. Merge very small chunks using chunk merging logic
8. Generate fine-grained proposition-level sub-chunks using dependency parsing and propositional extraction
9. Export structured JSON and JSONL outputs with bill metadata and chunk information

---

## Features

### Hybrid Chunking Architecture
- BERT-based semantic chunk boundary detection (tim1900/bert-chunker-3)
- Sliding-window chunking with overlap support
- Rule-based segmentation refinement
- Chunk merging logic for very small chunks
- Proposition-level sub-chunk generation via dependency parsing

### Legal Document Processing
- Parliamentary bill processing from markdown format
- Regex-based bill title and bill number extraction
- Section-level metadata extraction
- Legislative writing style handling

### Text Processing
- Customised NLTK Punkt tokenizer with legal abbreviation support
- SpaCy (en_core_web_sm) for NLP processing
- Dependency parsing for propositional extraction
- Regex-based title and bill number recognition

### Structured Outputs
- JSON and JSONL chunk outputs
- Metadata-rich chunk records
- Reproducible preprocessing pipeline

---

## Dataset Schema

Each output record contains:

```json
{
  "bill_title": "Example Parliamentary Bill",
  "bill_number": "Bill No. 1 of 2025",
  "section": "Part I — Preliminary",
  "chunk_id": 1,
  "chunk_text": "An Act to amend...",
  "sub_chunks": [
    "An Act",
    "to amend the relevant legislation"
  ]
}
```

---

## Chunking Threshold Evaluation

Three chunking thresholds were evaluated using the BERT-based chunking model:

| Threshold | Behaviour |
|---|---|
| 0.3 | More aggressive chunking, smaller chunks |
| 0.5 | Balanced — selected as final threshold |
| 0.7 | More conservative chunking, larger chunks |

Threshold 0.5 was selected as it produced the most semantically coherent chunk boundaries for parliamentary bill content.

---

## Technologies Used

- **Language:** Python
- **Chunking Model:** BERT (tim1900/bert-chunker-3)
- **NLP:** SpaCy (en_core_web_sm), NLTK Punkt Tokenizer
- **Parsing:** Dependency Parsing, Propositional Extraction
- **Output Format:** JSON, JSONL

---

## Applications

- Retrieval-Augmented Generation (RAG)
- Legal search and retrieval systems
- Long-context LLM document processing
- Legislative document analysis
- Legal AI dataset preparation

---

## Notes

This pipeline was built during an AI Engineer internship at the Home Team Science and Technology Agency (HTX), Singapore. Source documents processed are parliamentary bills in markdown format. Document outputs are not publicly available.
