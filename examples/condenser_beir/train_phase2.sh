
python -m torch.distributed.launch --nproc_per_node 4 -m remop.driver.train \
  --output_dir ./experiment_stage2 \
  --overwrite_output_dir \
  --model_type bert \
  --model_name_or_path Luyu/co-condenser-marco \
  --prefix \
  --per_device_train_batch_size 8 \
  --num_train_epochs 5 \
  --grad_cache \
  --negatives_x_device \
  --gc_q_chunk_size 8 \
  --gc_p_chunk_size 120 \
  --save_steps 6000 \
  --train_dir ./data/full_berri \
  --train_n_passages 6 \
  --fp16 \
  --general_lr 7e-6 \
  --attribute_lr 7e-3 \
  --general_prompt_path ./experiment_stage1/trained_prompts/general_state_dict.prompt