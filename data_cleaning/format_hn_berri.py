import os, json
import tqdm


berri_mapping = {
    "altlex_embeddings_100k": ["wikipedia", "sentence-paraphrase"], # low quality
	"cnn_dailymail_embeddings_100k": ["news", "summarization"], 
	"coco_captions_embeddings_100k": ["caption"], 
	"eli5_question_answer_embeddings_100k": ["qa", "wikipedia"], 
	"mdmcqa_corpus_qa_train": ["medical"],  # incomplete data
	"medical_sim_preprocessed": ["medical", "summarization"], 
	"msmarco-triplets_embeddings_600k": [], 
	"oqa_corpus": ["qa", "wikipedia"], # incomplete data
	# "berri_corpus_data": [], # duplicated data
	"multilexsum_preprocessed": ["legal", "summarization"], 
	"record_hard_negatives_added": ["news", "qa"], 
	"pubmed_hard_negatives_added": ["medical", "science"], 
	"paq_corpus": ["wikipedia", "qa"], 
	"ccnews_title_text_corpus": ["news", "summarization"], 
	"agnews_embeddings_100k": ["news", "summarization"], 
	"qrecc_preprocessed": ["wikipedia", "qa"], 
	"scitldr_preprocessed": ["science", "summarization"], 
	"searchQA_top5_snippets_embeddings_100k": ["qa"], 
	"yahoo_answers_title_answer_embeddings_100k": ["qa"], 
	"sentence-compression_embeddings_10k": ["summarization"], 
	"npr_embeddings_100k": ["news", "summarization"], 
	"squad_pairs_embeddings_100k": ["wikipedia", "qa"], 
	"stackexchange_duplicate_questions_title-body_title-body_embeddings_100k": ["qa"], 
	"stackexchange_duplicate_questions_title_title_embeddings_100k": ["sentence-paraphrase"], 
	"wikihow_embeddings_100k": ["qa"], 
	"xsum_embeddings_100k": ["news", "summarization"], 
	"quora_duplicates_triplets_embeddings": ["sentence-paraphrase"]
}

hn_dir = "remop_data/hn_contriever/hn_results"
hn_datasets = os.listdir(hn_dir)
# hn_datasets = ['scitldr_preprocessed'] # ['medical_sim_preprocessed']
output_dir = "remop_data/clean_berri_hn"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dataset in hn_datasets:
    datafile = os.path.join(hn_dir, dataset, "qa_data.json")
    data = [json.loads(d) for d in open(datafile)]
    attributes = berri_mapping[dataset]
    with open(os.path.join(output_dir, f"{dataset}.json"), 'w') as writer:
        for d in tqdm.tqdm(data, total=len(data), desc=dataset):
            if 'positive_ctxs' in d:
                pos_ctxs = [c['text'] for c in d['positive_ctxs']]
            else:
                pos_ctxs = d['answers']
            # neg_ctxs = [c['text']['text'] for c in d.get('negative_ctxs', [])]
            neg_ctxs = [c['text'] for c in d.get('negative_ctxs', [])]
            hn_ctxs = [c['text'] for c in d['ctxs'] if c['text'] not in pos_ctxs and c['text'] not in neg_ctxs]
            # print(d['negative_ctxs'])
            # print(d['ctxs'])
            # break
            writer.write(json.dumps({
                "query": d['question'],
                "pos_ctxs": pos_ctxs,
                "neg_ctxs": neg_ctxs + hn_ctxs,
                "attributes": attributes
            }, ensure_ascii=False))
            writer.write("\n")