# 1. Installment

### 1.1 Download Pretrained Model
First download the pretrained model from Segment Anything.

```bash
mkdir pretrained_models
cd pretrained_models
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
cd ..
```

### 1.2 Install Enviornment

Simply run

```
bash install.sh
```

or 
```bash
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
```

