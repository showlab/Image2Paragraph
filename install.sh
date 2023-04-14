conda create --name i2p python=3.8 -y
conda activate i2p
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout cc87e7ec
pip install -e .
cd ..
pip install -r requirements.txt
pip install -U transformers
pip install openai
pip install --upgrade diffusers[torch]
pip install setuptools==59.5.0
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -U openmim
mim install mmcv
pip install spacy
python -m spacy download en_core_web_sm