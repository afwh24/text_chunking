# Legal Document Semantic Chunker

A Python pipeline that transforms Singapore parliamentary bills into structured semantic chunks and word-count-based sub-chunks, suitable for downstream retrieval and legal AI applications. Built during a six-month AI Engineer internship at HTX, Singapore's national defence-tech agency.

## Scale

Over **2,000 parliamentary bills** processed.

## Why this exists

Legal documents are structurally difficult for general-purpose NLP tools. They contain nested sections, numbered clauses, defined terms, and abbreviations (e.g. "s." for section, "Art." for Article) that standard sentence tokenisers routinely misinterpret, splitting sentences in the wrong places or losing structural context. This pipeline combines a pretrained neural chunking model with customised traditional NLP techniques to handle these cases correctly.

## Pipeline

### 1. Metadata extraction
A single regular expression extracts the bill title and bill number from each document, matching text ending in "Bill" followed on the next line by a "Bill No. X/Y." pattern.

### 2. Chunk boundary detection (BERT)
Chunk boundaries are detected using a pretrained BERT-based token classification model, [`tim1900/bert-chunker-3`](https://huggingface.co/tim1900/bert-chunker-3), run on CPU. The tokenizer is configured with a maximum sequence length of 255 tokens.

Three confidence thresholds were evaluated during development — 0.3, 0.5, and 0.7 — and **0.7** was selected as the final configuration based on inspection of output chunk quality.

The model processes text in sequential, non-overlapping windows, advancing to the position of the last detected chunk boundary (or to the end of the window if no boundary was found), rather than using a classic overlapping sliding window.

### 3. Sentence-level sub-chunking (NLTK)
Within each BERT-detected chunk, sentences are further split using a customised NLTK Punkt tokenizer. Two custom mechanisms improve accuracy on legal text:

- **Custom abbreviation list** (`chap`, `cap`, `no`, `sec`, `art`, `pt`, `vol`, `ann`) prevents the tokenizer from incorrectly treating these as sentence-ending periods
- **Number-period protection**: numbered list markers (e.g. "1.", "3.") are temporarily masked before tokenization and restored afterward, preventing them from being misread as sentence boundaries

Sub-chunks shorter than ten words are merged into the following sub-chunk(s) until the combined length reaches at least ten words, reducing fragmentation in the final output.

### 4. Output
Each bill produces a single JSON file containing:

```json
{
  "bill_title": "...",
  "bill_number": "...",
  "chunks": [
    {
      "chunk_index": 0,
      "token_position": 0,
      "chunk_text": "...",
      "sub_chunks": [
        { "sub_chunk_id": "0.0", "text": "..." },
        { "sub_chunk_id": "0.1", "text": "..." }
      ]
    }
  ]
}
```

## Resume behaviour

The pipeline checks whether an output file already exists for a given input bill before processing, allowing a long-running batch of 2,000+ bills to be safely interrupted and resumed without redoing completed work. Resumability is at the level of whole bills, not individual chunks.

## Tech stack

Python, Hugging Face Transformers, PyTorch, NLTK
