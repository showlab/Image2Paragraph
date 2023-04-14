import os
from models.grit.image_dense_captions import image_caption_api

class DenseCaptioning():
    def __init__(self) -> None:
        self.model = None


    def initialize_model(self):
        pass

    def image_dense_caption_debug(self, image_src):
        dense_caption = """
        1. the broccoli is green, [0, 0, 333, 325]; 
        2. a piece of broccoli, [0, 147, 143, 324]; 
        3. silver fork on plate, [4, 547, 252, 612];
        """
        return dense_caption
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_api(image_src)
        print("Step2, Dense Caption:\n")
        print(dense_caption)
        print('\n'+'*'*100)
        return dense_caption
    