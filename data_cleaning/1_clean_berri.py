import os, json


origin_path = "berri_corpus_data"
output_path = "processed_clean_data"

# agnews_embeddings_100k
data_name = "agnews_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# altlex_embeddings_100k
data_name = "altlex_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# ccnews_title_text_corpus
data_name = "ccnews_title_text_corpus"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# cnn_dailymail_embeddings_100k
data_name = "cnn_dailymail_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# coco_captions_embeddings_100k
data_name = "coco_captions_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# eli5_question_answer_embeddings_100k
data_name = "eli5_question_answer_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# medical_sim_preprocessed
data_name = "medical_sim_preprocessed"
folder_path = os.path.join(origin_path, data_name)
dataset = [json.loads(d) for d in open(os.path.join(folder_path, "medical_sim_train_dpr.jsonl"))]
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text']['text'] for d in data['negative_ctxs']]
    })

with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)

print(f"done {data_name}")

# msmarco-triplets_embeddings_600k
data_name = "msmarco-triplets_embeddings_600k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# multilexsum_preprocessed
data_name = "multilexsum_preprocessed"
folder_path = os.path.join(origin_path, data_name)
dataset = [json.loads(d) for d in open(os.path.join(folder_path, "multilexsum_train_dpr.jsonl"))]
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# npr_embeddings_100k
data_name = "npr_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# oqa_corpus
# data_name = "oqa_corpus"
# folder_path = os.path.join(origin_path, data_name)
# dataset = [json.loads(d) for d in open(os.path.join(folder_path, "oqa_qq_train.jsonl"))]
# new_dataset = []
# for data in dataset:
#     new_dataset.append({
#         "query": data['question'],
#         "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
#         "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
#     })
# with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
#     json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
# print(f"done {data_name}")

# paq_corpus
data_name = "paq_corpus"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# qrecc_preprocessed
data_name = "qrecc_preprocessed"
folder_path = os.path.join(origin_path, data_name)
dataset = [json.loads(d) for d in open(os.path.join(folder_path, "qrecc_train_dpr.jsonl"))]
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# quora_duplicates_triplets_embeddings
data_name = "quora_duplicates_triplets_embeddings"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# scitldr_preprocessed
data_name = "scitldr_preprocessed"
folder_path = os.path.join(origin_path, data_name)
dataset = [json.loads(d) for d in open(os.path.join(folder_path, "scitldr_train_dpr.jsonl"))]
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# searchQA_top5_snippets_embeddings_100k
data_name = "searchQA_top5_snippets_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# sentence-compression_embeddings_10k
data_name = "sentence-compression_embeddings_10k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# squad_pairs_embeddings_100k
data_name = "squad_pairs_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# stackexchange_duplicate_questions_title_title_embeddings_100k
data_name = "stackexchange_duplicate_questions_title_title_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# stackexchange_duplicate_questions_title-body_title-body_embeddings_100k
data_name = "stackexchange_duplicate_questions_title-body_title-body_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# wikihow_embeddings_100k
data_name = "wikihow_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# xsum_embeddings_100k
data_name = "xsum_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# yahoo_answers_title_answer_embeddings_100k
data_name = "yahoo_answers_title_answer_embeddings_100k"
folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(folder_path, "qa_data.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# # mdmcqa_corpus_qa_train
# data_name = "mdmcqa_corpus_qa_train"
# # folder_path = os.path.join(origin_path, data_name)
# dataset = json.load(open(os.path.join(origin_path, "mdmcqa_corpus_qa_train.json")))
# new_dataset = []
# for data in dataset:
#     new_dataset.append({
#         "query": data['target'],
#         "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
#         "neg_ctxs": [d['text'] for d in data['negative_ctxs']]
#     })
# with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
#     json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
# print(f"done {data_name}")

# pubmed_hard_negatives_added
data_name = "pubmed_hard_negatives_added"
dataset = json.load(open(os.path.join(origin_path, "pubmed_hard_negatives_added.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['hard_negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")

# record_hard_negatives_added
data_name = "record_hard_negatives_added"
# folder_path = os.path.join(origin_path, data_name)
dataset = json.load(open(os.path.join(origin_path, "record_hard_negatives_added.json")))
new_dataset = []
for data in dataset:
    new_dataset.append({
        "query": data['question'],
        "pos_ctxs": [d['text'] for d in data['positive_ctxs']],
        "neg_ctxs": [d['text'] for d in data['hard_negative_ctxs']]
    })
with open(os.path.join(output_path, f"{data_name}.json"), 'w', encoding='utf-8') as writer:
    json.dump(new_dataset, writer, indent=4, ensure_ascii=False)
print(f"done {data_name}")