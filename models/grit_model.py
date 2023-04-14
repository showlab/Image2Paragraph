"""
since grit install is complex, we call subprocess to run grit demo.py
"""
import subprocess
import os

class DenseCaptioning():
    def __init__(self) -> None:
        self.grit_working_directory = "../GRiT/"
        self.grit_env_python = '/home/aiops/wangjp/anaconda3/envs/grit/bin/python'
        self.grit_script = 'image_dense_captions.py'
        self.model_weights = 'models/grit_b_densecap_objectdet.pth'

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
        result = subprocess.run([self.grit_env_python, self.grit_script, '--image_src', image_src, '--test-task', 'DenseCap', '--config-file', 'configs/GRiT_B_DenseCap_ObjectDet.yaml', '--opts', 'MODEL.WEIGHTS', self.model_weights], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,  cwd=self.grit_working_directory)
        print("Subprocess finished, continuing main.py...\n")
        # output = result.stdout
        output_file = os.path.expanduser("~/grit_output.txt")
        with open(output_file, 'r') as f:
            output = f.read()
        # print('*'*100)
        # print("Output:", output)
        print("Step2, Dense Caption:\n")
        # print(output.split('[START]'[-1]))
        print(output)
        print('\n'+'*'*100)
        return output
    