attributes = [
    "caption",
    "dialogue",
    "fact-verification",
    "legal",
    "medical",
    "news",
    "qa",
    "science",
    "sentence-paraphrase",
    "summarization",
    "wikipedia",
]

berri_mapping = {
    # "altlex_embeddings_100k": ["wikipedia", "sentence-paraphrase"], # low quality
	"cnn_dailymail_embeddings_100k": ["news", "summarization"], 
	"coco_captions_embeddings_100k": ["caption"], 
	"eli5_question_answer_embeddings_100k": ["qa", "wikipedia"], 
	# "mdmcqa_corpus_qa_train": ["medical"],  # incomplete data
	"medical_sim_preprocessed": ["medical", "summarization"], 
	"msmarco-triplets_embeddings_600k": [], 
	# "oqa_corpus": ["qa", "wikipedia"], # incomplete data
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