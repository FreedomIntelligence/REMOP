
DATA_NAME='arguana'

python -m remop.faiss_retriever \
--query_reps ./encoded/$DATA_NAME.query \
--passage_reps ./encoded/$DATA_NAME.corpus/'*.pt' \
--depth 1000 \
--batch_size -1 \
--save_text \
--save_ranking_to encoded/$DATA_NAME.rank.tsv 