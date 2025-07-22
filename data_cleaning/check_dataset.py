import os, json, random
from mapping_berri import berri_mapping

datasets = ['cnn_dailymail_embeddings_100k', 'coco_captions_embeddings_100k', 'eli5_question_answer_embeddings_100k', 'medical_sim_preprocessed', 'msmarco-triplets_embeddings_600k', 'multilexsum_preprocessed', 'record_hard_negatives_added', 'pubmed_hard_negatives_added', 'paq_corpus', 'ccnews_title_text_corpus', 'agnews_embeddings_100k', 'qrecc_preprocessed', 'scitldr_preprocessed', 'searchQA_top5_snippets_embeddings_100k', 'yahoo_answers_title_answer_embeddings_100k', 'sentence-compression_embeddings_10k', 'npr_embeddings_100k', 'squad_pairs_embeddings_100k', 'stackexchange_duplicate_questions_title-body_title-body_embeddings_100k', 'stackexchange_duplicate_questions_title_title_embeddings_100k', 'wikihow_embeddings_100k', 'xsum_embeddings_100k', 'quora_duplicates_triplets_embeddings']
root_path = "remop_data/clean_data"


check_dataset = []
for dataset, attributes in berri_mapping.items():
    print(f"processing {dataset}-{attributes}")
    data = json.load(open(os.path.join(root_path, f"{dataset}.json")))
    check_dataset.append({f"{dataset}-{attributes}": random.sample(data, 2)})

with open("check_dataset.json", 'w') as writer:
    json.dump(check_dataset, writer, indent=4, ensure_ascii=False)