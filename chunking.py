
import re
import json
import math
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

#FINAL WORKING VERSION OF TEXT CHUNKING

#BERT chunking + CUSTOMIZED NLTK tokenizing

# ========== Setup ==========
#BERT CONFIGURATION
MODEL_PATH = "tim1900/bert-chunker-3"
MAX_TOKENS = 255
PROB_THRESHOLD = 0.7
DEVICE = "cpu"  # or 'cuda'

#FILE PATH
current_dir = Path("/workspace/alfred/text_chunking/parliament_bills/md/")
OUT_PATH = Path("/workspace/alfred/text_chunking/parliament_bills/jsonl/")

#sort file names
files = sorted([f for f in current_dir.iterdir() if f.is_file() and f.suffix == '.md'])

#set the rule for (customized) sentence detection boundaries -> to prevent false split
punkt_params = PunktParameters()
punkt_params.abbrev_types.update([
        'chap', 'cap', 'no', 'sec', 'art', 'pt', 'vol', 'ann'  # all MUST BE lowercase
    ])

# Create the custom sentence tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(punkt_params)

def protect_number_periods(text):
    # Matches numbers like 1. or 3. (not decimals like 3.14)
    return re.sub(r'\b(\d+)\.(?=\s*[A-Z0-9])', r'\1<PERIOD>', text)

def restore_number_periods(text):
    return text.replace('<PERIOD>', '.')

def custom_sent_tokenize(text): 
    protected = protect_number_periods(text)
    
    # split on \n\n to simulate hard sentence/paragraph boundaries
    paragraph_chunks = protected.split('\n\n')

    # Tokenize each paragraph separately using the custom sentence tokenizer 
    sentences = []
    for para in paragraph_chunks:
        if para.strip():
            para_sents = custom_sent_tokenizer.tokenize(para)
            sentences.extend(para_sents)
    
    return [restore_number_periods(sent) for sent in sentences]

# ========== Load BERT Model ==========
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, padding_side="right", model_max_length=255, trust_remote_code=True
)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)


# ========== BERT Chunking Function ==========
def chunk_text(model, text, tokenizer, prob_threshold=0.5):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"]
    CLS = input_ids[:, 0].unsqueeze(0)
    SEP = input_ids[:, -1].unsqueeze(0)
    input_ids = input_ids[:, 1:-1]

    model.eval()
    split_positions = []
    token_positions = []
    start = 0
    threshold_logit = math.log(1 / prob_threshold - 1)

    print(f">>>>> BERT chunking on {input_ids.shape[1]} tokens...")

    while start < input_ids.shape[1]:
        end = start + MAX_TOKENS - 2
        window_ids = torch.cat((CLS, input_ids[:, start:end], SEP), dim=1).to(DEVICE)

        output = model(input_ids=window_ids, attention_mask=torch.ones_like(window_ids))
        logits = output.logits[:, 1:-1, :]  # remove CLS/SEP
        chunk_probs = logits[:, :, 1] > (logits[:, :, 0] - threshold_logit)
        split_idxs = torch.where(chunk_probs)[1].tolist()

        if split_idxs and not (split_idxs[0] == 0 and len(split_idxs) == 1):
            for idx in split_idxs:
                if idx > 0:
                    char_start = tokens.token_to_chars(idx + start + 1).start
                    split_positions.append(char_start)
                    token_positions.append(idx + start + 1)
            start += split_idxs[-1]
        else:
            start = end

    # Create substrings based on split points
    chunk_texts = [
        text[i:j] for i, j in zip([0] + split_positions, split_positions + [len(text)])
    ]
    token_positions = [0] + token_positions

    return chunk_texts, token_positions


#file counter
counter = 0

# ========== load text for each md files ==========
for item in files:
    jsonl_item = item.name.replace('.md','.jsonl')
    jsonl_item_path = OUT_PATH/jsonl_item
    
    print("Processing file name: " + item.name)

    #skipped md files that has already been converted; act as check point
    if(jsonl_item_path.exists()): 
        counter +=1
        print(f"{item.name} skipped!")
        print("File Completed: " + str(counter))
        continue

    #md files has not been converted to JSONL 
    text = item.read_text(encoding="utf-8")

    # ========== Extract Bill Name and Number ==========
    bill_info_pattern = re.compile(
        r"^(?P<title>.+?Bill)\s*\n(?P<bill_no>Bill No\.\s*\d+/\d+\.)", re.MULTILINE
    )
    bill_match = bill_info_pattern.search(text)
    bill_title = bill_match.group("title").strip() if bill_match else "Unknown"
    bill_number = bill_match.group("bill_no").replace("Bill No.", "").strip() if bill_match else "Unknown"

    # ========== Run BERT Chunking ==========
    chunks, token_positions = chunk_text(model, text, tokenizer, prob_threshold=PROB_THRESHOLD)

    # ========== NLTK Sub-chunking ==========
    chunk_data = []
    for i, (chunkText, token_pos) in enumerate(zip(chunks, token_positions)):
        sub_chunks = custom_sent_tokenize(chunkText)

        merged_chunks = []
        #index to keep track of the sub_chunks
        j = 0 

        #merge the sub chunks that are < 10 into the subsequent sub chunks
        while j<len(sub_chunks):
            buffer = sub_chunks[j].strip()
            word_count = len(buffer.split())
            k = j+1

            while word_count < 10 and k < len(sub_chunks):
                buffer += " " + sub_chunks[k].strip()
                word_count+= len(sub_chunks[k].split())
                k+=1
            
            merged_chunks.append(buffer)
            j = k

        sub_chunk_data = [
            {"sub_chunk_id": f"{i}.{j}", "text": sent.strip()}
            for j, sent in enumerate(merged_chunks)
        ]
        chunk_data.append({
            "chunk_index": i,
            "token_position": token_pos,
            "chunk_text": chunkText.strip(),
            "sub_chunks": sub_chunk_data
        })
    # ========== Save as JSON ==========
    output_json = {
        "bill_title": bill_title,
        "bill_number": bill_number,
        "chunks": chunk_data
    }

    with open(jsonl_item_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    

    print(item.name + " text chunking done!")
    counter+=1
    print("File Completed: " + str(counter))



print("All text chunking done!")