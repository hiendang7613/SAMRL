import os
import sys
import urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAM:
    def __init__(self, 
    checkpoint_url=None, 
    model_type="vit_h", 
    device="cuda"):
      if not checkpoint_url:
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
      checkpoint_filename = os.path.basename(checkpoint_url)
      if not os.path.exists(checkpoint_filename):
          print(f"Downloading checkpoint from {checkpoint_url}...")
          urllib.request.urlretrieve(checkpoint_url, checkpoint_filename)
      self.sam = sam_model_registry[model_type](checkpoint=checkpoint_filename).to(device=device)
      self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    def segment(self, images):
      masks = []
      for image in images:
        masks.append(self.mask_generator.generate(image))
      return masks
