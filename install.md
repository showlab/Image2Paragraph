# 1. Installment

### 1.1 Download Pretrained Model
First cd in the project path, then download the pretrained model from Segment Anything and grit.

```bash
cd [YOUR_PATH_TO_THIS_PROJECT]
mkdir pretrained_models
cd pretrained_models
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
cd ..
```

### 1.2 Install Enviornment

Simply run

```
pip install -r requirements.txt
```