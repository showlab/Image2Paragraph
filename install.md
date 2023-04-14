# 1. Installment

I recommend you create two envs **ssa,grit** for this project, as below:

## 1. ssa env

### 1.1 Download Pretrained Model
First download the pretrained model from Segment Anything.

```bash
cd pretrained_models;
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth;
```

### 1.2 Install Dependencies of Semantic Segment Anything
The semantic segment anything rely multiple bigger models from Transformers.
Follow instruction in [semantic segment anything](https://github.com/fudan-zvg/Semantic-Segment-Anything). This will create a conda env named ssa.

Then
```bash
conda activate ssa
```

### 1.3 Install Dependencies of BLIP2, ChatGPT
```bash
pip install transformers
pip install openai
```

### 1.4 Install Dependencies of Segment Anything

Follow instruction in [segment anything](https://github.com/facebookresearch/segment-anything).


### 1.5 Install Dependencies of ControlNet from diffuser

Follow instruction in [diffuser](https://github.com/huggingface/diffusers).


## 2. grit env


### GRIT

Follow instruction in [GRIT](https://github.com/JialianW/GRiT/blob/master/docs/INSTALL.md). This will create a conda env named grit.

__I suggest to create a new env for GRIT since it depends on complex Detectron2.__

Then
```
cp utils/image_dense_captions.py [YOUR_GRIT_DIRECTORY]
```

modify line10-line12 in __/models/grit_model.py__ accordingly.

The main process will call this code with subprocess for dense caption.

**Make sure you can run GRIT code first and then check the environment name carefully.**
