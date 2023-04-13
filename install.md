## 1. Installment

### BLIP2, ChatGPT,
```bash
pip install transformers
pip install openai
```

### ControlNet from diffuser

Follow instruction in [diffuser](https://github.com/huggingface/diffusers).

### GRIT

Follow instruction in [GRIT](https://github.com/JialianW/GRiT/blob/master/docs/INSTALL.md).

I suggest to create a new env for GRIT since it depends on Detectron2.

Then
```
cp utils/image_dense_captions.py [YOUR_GRIT_DIRECTORY]
```

modify line10-line12 in __/models/grit_model.py__ accordingly.

### Segment Anything
The segment anything rely multiple bigger models from Transformers.

Follow instruction in [segment anything](https://github.com/facebookresearch/segment-anything).

```bash
cd pretrained_models;
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth;
```

### Semantic Segment Anything

Follow instruction in [semantic segment anything](https://github.com/fudan-zvg/Semantic-Segment-Anything).

We recommend first install this dependence first.