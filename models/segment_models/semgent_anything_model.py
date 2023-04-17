import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils.util import resize_long_edge_cv2

class SegmentAnything:
    def __init__(self, device, arch="vit_b"):
        self.device = device
        if arch=='vit_b':
            pretrained_weights="pretrained_models/sam_vit_b_01ec64.pth"
        elif arch=='vit_l':
            pretrained_weights="pretrained_models/sam_vit_l_0e2f7b.pth"
        elif arch=='vit_h':
            pretrained_weights="pretrained_models/sam_vit_h_0e2f7b.pth"
        else:
            raise ValueError(f"arch {arch} not supported")
        self.model = self.initialize_model(arch, pretrained_weights)
    
    def initialize_model(self, arch, pretrained_weights):
        sam = sam_model_registry[arch](checkpoint=pretrained_weights)
        sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    def generate_mask(self, img_src):
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_long_edge_cv2(image, 384)
        anns = self.model.generate(image)
        return anns