datasets=(
    cnn_dailymail_embeddings_100k
    coco_captions_embeddings_100k
    eli5_question_answer_embeddings_100k
    paq_corpus
    ccnews_title_text_corpus
    agnews_embeddings_100k
    qrecc_preprocessed
    medical_sim_preprocessed
    mdmcqa_corpus_qa_train
    scitldr_preprocessed
    searchQA_top5_snippets_embeddings_100k
    yahoo_answers_title_answer_embeddings_100k
    sentence-compression_embeddings_10k
    npr_embeddings_100k
    squad_pairs_embeddings_100k
    stackexchange_duplicate_questions_title-body_title-body_embeddings_100k
    stackexchange_duplicate_questions_title_title_embeddings_100k
    wikihow_embeddings_100k
    xsum_embeddings_100k
    quora_duplicates_triplets_embeddings
)

for dataset in "${datasets[@]}"
do
    echo "processing ${dataset}"
    bash data_cleaning/hn_generation/passage_encode.sh $dataset 0
    bash data_cleaning/hn_generation/passage_retrieve.sh $dataset 0
done

wait