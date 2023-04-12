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
Follow instruction in [diffuser](https://github.com/huggingface/diffusers).
