from torchvision.models.segmentation import deeplabv3_resnet101
import torch
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np

class SegmentationModelOutputWrapper(torch.nn.Module):
  def __init__(self, model): 
    super(SegmentationModelOutputWrapper, self).__init__()
    self.model = model
        
  def forward(self, x):
    return self.model(x)["out"]

mask_model = deeplabv3_resnet101(pretrained=True, progress=False)
mask_model = mask_model.eval()
mask_model = SegmentationModelOutputWrapper(mask_model)


sem_classes = [
  '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
desired_category = sem_class_to_idx["cat"]

def get_mask(image):
  rgb_img = np.float32(image) / 255
  image_tensor = preprocess_image(rgb_img,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
  out = mask_model(image_tensor)
  
  normalized_masks = torch.nn.functional.softmax(out, dim=1).cpu()
  
  # Image.fromarray(both_images).save(f'output/{index}_mask.jpg')
  
  desired_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
  desired_mask_uint8 = 255 * np.uint8(desired_mask == desired_category)
  # desired_mask_float = np.float32(desired_mask == desired_category)

  return desired_mask_uint8


