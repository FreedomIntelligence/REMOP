MODEL_PATH=facebook/contriever-msmarco
DATA_NAME=$1
PASSAGE_FILE=./berri_corpus_data/${DATA_NAME}/corpus.tsv
PASSAGE_EMBEDS=./remop_data/hn_contriever/passage_embeds/${DATA_NAME}
QUERY_FILE=./berri_corpus_data/${DATA_NAME}/qa_data.json
HARDNEG_RESULT=./remop_data/hn_contriever/hn_results/${DATA_NAME}

# encode passage
# CUDA_VISIBLE_DEVICES=$2 python generate_passage_embeddings.py \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir $PASSAGE_EMBEDS  \
#     --passages $PASSAGE_FILE \
#     --shard_id 0 --num_shards 1

# retrieve hard negative
CUDA_VISIBLE_DEVICES=$2 python ./data_cleaning/contriever/passage_retrieval.py \
    --model_name_or_path $MODEL_PATH \
    --passages ${PASSAGE_FILE} \
    --n_docs 10 \
    --passages_embeddings "${PASSAGE_EMBEDS}/*" \
    --data $QUERY_FILE \
    --output_dir $HARDNEG_RESULT \