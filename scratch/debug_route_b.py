import cv2
import numpy as np
from src.vision.pipelines import PipelineRegistry

img = cv2.imread('data/workspaces/20260916_145246_Home_real/production/raw_images/view_0016.png')
pB = PipelineRegistry.create('edge_centerline')
resB = pB.run(img, None)
print("Extra metrics:", resB.extra_metrics)
print("Targets:", len(resB.targets))
