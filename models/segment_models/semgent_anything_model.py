import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

class SegmentAnything:
    def __init__(self, arch="vit_h", pretrained_weights="pretrained_models/sam_vit_h_4b8939.pth"):
        # self.model = None
        self.model = self.initialize_model(arch, pretrained_weights)
    
    def initialize_model(self, arch, pretrained_weights):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[arch](checkpoint=pretrained_weights)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    def generate_mask(self, img_src):
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = self.model.generate(image)
        return anns