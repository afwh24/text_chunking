
---

# Text Chunking Pipeline

```markdown
# Legal Text Chunking Pipeline

A hybrid legal document chunking pipeline designed for segmenting long legislative and legal documents into context-preserving chunks for downstream NLP and LLM applications.

---

## Overview

This project focuses on intelligent segmentation of legal and parliamentary documents using a hybrid approach combining:

- BERT-based boundary detection
- Rule-based processing
- Customized tokenization
- Sliding-window chunking

The goal is to preserve semantic coherence and legal context while preparing documents for retrieval and long-context AI systems.

---

## Features

### Hybrid Chunking Architecture
- BERT-based segmentation
- Rule-based boundary detection
- Context-aware chunking
- Sliding-window overlap support

### Legal Document Support
- Parliamentary bills
- Legislative documents
- Legal judgments
- Long-form legal text

### Text Processing
- Customized NLTK tokenization
- Regex-based title extraction
- Bill number recognition
- Context preservation

### Structured Outputs
- JSONL export
- Metadata-rich chunk records
- Reproducible preprocessing pipeline

---

## Chunking Workflow

1. Load legal documents
2. Detect structural boundaries
3. Apply BERT segmentation
4. Perform rule-based adjustments
5. Generate overlapping chunks
6. Export structured JSONL

---

## Example Output

```json
{
  "document_id": "bill_001",
  "chunk_id": 5,
  "text": "An Act to amend...",
  "start_position": 1200,
  "end_position": 1800
}
