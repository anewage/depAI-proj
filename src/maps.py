import warnings
warnings.filterwarnings('ignore')
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import torch
import cv2
import math
from typing import List, Callable, Optional
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
  def __init__(self, model):
    super(HuggingfaceToTensorModelWrapper, self).__init__()
    self.model = model

  def forward(self, x):
    return self.model(x).logits

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
  name_to_index = dict((v, k) for k, v in model.config.id2label.items())
  return name_to_index[category_name]
    
""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          method: Callable=GradCAM):
  with method(model=HuggingfaceToTensorModelWrapper(model), target_layers=[target_layer], reshape_transform=reshape_transform) as cam:

    # Replicate the tensor for each of the categories we want to create Grad-CAM for:
    repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

    batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)
    results = []
    for grayscale_cam in batch_results:
      visualization = show_cam_on_image(np.float32(input_image)/255, grayscale_cam, use_rgb=True)
      visualization = cv2.resize(visualization, (visualization.shape[1], visualization.shape[0]))
      results.append(visualization)
    return [batch_results, results]


def reshape_transform_vit_huggingface(x):
  print(x.shape)
  activations = x[:, 1:, :]
  magic_number = int(math.sqrt(activations.shape[1]))
  print(magic_number)
  activations = activations.view(activations.shape[0], magic_number, magic_number, activations.shape[2])
  activations = activations.transpose(2, 3).transpose(1, 2)
  return activations

## dive into reshape transform
def get_map(model, predicted_labels, inputs_tensor, image_resized):
  target_layer_gradcam = model.vit.encoder.layer[-2].output
  targets_for_gradcam = [ClassifierOutputTarget(i) for i in predicted_labels]
  return run_grad_cam_on_image(model=model,
                      target_layer=target_layer_gradcam,
                      targets_for_gradcam=targets_for_gradcam,
                      input_tensor=inputs_tensor.pixel_values[0],
                      input_image=image_resized,
                      reshape_transform=reshape_transform_vit_huggingface)