import json
import os
import tqdm
dataset = json.load(open("processed_clean_data/msmarco-triplets_embeddings_600k.json"))
if not os.path.exists("train_phase1_msmarco"):
    os.makedirs("train_phase1_msmarco")
with open("train_phase1_msmarco/train.jsonl", 'w') as writer:
    for data in tqdm.tqdm(dataset):
        writer.write(json.dumps(data, ensure_ascii=False))
        writer.write("\n")
        

