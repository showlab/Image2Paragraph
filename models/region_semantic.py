from models.segment_models.semgent_anything_model import SegmentAnything
from models.segment_models.semantic_segment_anything_model import SemanticSegment
from models.segment_models.edit_anything_model import EditAnything


class RegionSemantic():
    def __init__(self, device, image_caption_model, region_classify_model='edit_anything', sam_arch='vit_b'):
        self.device = device
        self.sam_arch = sam_arch
        self.image_caption_model = image_caption_model
        self.region_classify_model = region_classify_model
        self.init_models()

    def init_models(self):
        self.segment_model = SegmentAnything(self.device, arch=self.sam_arch)
        if self.region_classify_model == 'ssa':
            self.semantic_segment_model = SemanticSegment(self.device)
        elif self.region_classify_model == 'edit_anything':
            self.edit_anything_model = EditAnything(self.image_caption_model)
            print('initalize edit anything model')
        else:
            raise ValueError("semantic_class_model must be 'ssa' or 'edit_anything'")
        
    def semantic_prompt_gen(self, anns, topk=5):
        """
        fliter too small objects and objects with low stability score
        anns: [{'class_name': 'person', 'bbox': [0.0, 0.0, 0.0, 0.0], 'size': [0, 0], 'stability_score': 0.0}, ...]
        semantic_prompt: "person: [0.0, 0.0, 0.0, 0.0]; ..."
        """
        # Sort annotations by area in descending order
        sorted_annotations = sorted(anns, key=lambda x: x['area'], reverse=True)
        anns_len = len(sorted_annotations)
        # Select the top 10 largest regions
        top_10_largest_regions = sorted_annotations[:min(anns_len, topk)]
        semantic_prompt = ""
        for region in top_10_largest_regions:
            semantic_prompt += region['class_name'] + ': ' + str(region['bbox']) + "; "
        print(semantic_prompt)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return semantic_prompt

    def region_semantic(self, img_src, region_classify_model='edit_anything'):
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("\nStep3, Semantic Prompt:")
        print('extract region segmentation with SAM model....\n')
        anns = self.segment_model.generate_mask(img_src)
        print('finished...\n')
        if region_classify_model == 'ssa':
            print('generate region supervision with blip2 model....\n')
            anns_w_class = self.semantic_segment_model.semantic_class_w_mask(img_src, anns)
            print('finished...\n')
        elif region_classify_model == 'edit_anything':
            print('generate region supervision with edit anything model....\n')
            anns_w_class = self.edit_anything_model.semantic_class_w_mask(img_src, anns)
            print('finished...\n')
        else:
            raise ValueError("semantic_class_model must be 'ssa' or 'edit_anything'")
        return self.semantic_prompt_gen(anns_w_class)
    
    def region_semantic_debug(self, img_src):
        return "region_semantic_debug"