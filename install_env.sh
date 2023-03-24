
# please ensure python>=3.7

pip install sentencepiece && \
pip install -U scikit-learn && \
conda install -c conda-forge faiss-gpu==1.7.1 && \
pip install transformers==4.26.0 && \
pip install datasets==1.17.0 && \
pip install deepspeed==0.4.0 && \
pip install tensorflow-gpu==2.5.0 \
pip install beir \
pip install pyserini

git clone https://github.com/NVIDIA/apex ~/.apex && \
cd ~/.apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

git clone https://github.com/luyug/GradCache ~/GradCache && \
cd ~/GradCache; pip install -e .

