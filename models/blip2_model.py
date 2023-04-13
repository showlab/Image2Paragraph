from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


class ImageCaptioning:
    def __init__(self) -> None:
        # self.processor, self.model = None, None
        self.processor, self.model = self.initialize_model()

    def initialize_model(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu" # for low gpu memory devices
        if device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=self.data_type
        )
        model.to(device)
        return processor, model

    def image_caption(self, image_src):
        image = Image.open(image_src)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu" # for low gpu memory devices
        inputs = self.processor(images=image, return_tensors="pt").to(device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print('*'*100 + '\nStep1, BLIP2 caption:')
        print(generated_text)
        print('\n' + '*'*100)
        return generated_text
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."