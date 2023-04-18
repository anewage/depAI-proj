import argparse
import cv2
import numpy as np
import torch
import os

from pytorch_grad_cam import GradCAM, \
  ScoreCAM, \
  GradCAMPlusPlus, \
  AblationCAM, \
  XGradCAM, \
  EigenCAM, \
  EigenGradCAM, \
  LayerCAM, \
  FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
  preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from datasets import load_dataset
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models.segmentation import deeplabv3_resnet101


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--use-cuda', action='store_true', default=False,
                      help='Use NVIDIA GPU acceleration')
  parser.add_argument(
      '--directory',
      type=str,
      default='./data',
      help='Input dataset')
  parser.add_argument(
      '--output',
      type=str,
      default='./output',
      help='Output directory')
  parser.add_argument('--aug_smooth', action='store_true',
                      help='Apply test time augmentation to smooth the CAM')
  parser.add_argument(
      '--eigen_smooth',
      action='store_true',
      help='Reduce noise by taking the first principle componenet'
      'of cam_weights*activations')

  parser.add_argument(
      '--method',
      type=str,
      default='gradcam',
      help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

  args = parser.parse_args()
  args.use_cuda = args.use_cuda and torch.cuda.is_available()
  if args.use_cuda:
    print('Using GPU for acceleration')
  else:
    print('Using CPU for computation')

  return args


def reshape_transform(tensor, height=14, width=14):
  result = tensor[:, 1:, :].reshape(tensor.size(0),
                                    height, width, tensor.size(2))

  # Bring the channels to the first dimension,
  # like in CNNs.
  result = result.transpose(2, 3).transpose(1, 2)
  return result


def category_name_to_index(model, category_name):
  name_to_index = dict((v, k) for k, v in model.config.id2label.items())
  return name_to_index[category_name]


class SegmentationModelOutputWrapper(torch.nn.Module):
  def __init__(self, model): 
    super(SegmentationModelOutputWrapper, self).__init__()
    self.model = model
        
  def forward(self, x):
    return self.model(x)["out"]

if __name__ == '__main__':
  """ python generate_masks.py --directory <path_to_directory>
  Example usage of using cam-methods on a VIT network.

  """

  args = get_args()
  methods = \
      {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad}

  if args.method not in list(methods.keys()):
    raise Exception(f"method should be one of {list(methods.keys())}")

  # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
  # model.eval()
  model = deeplabv3_resnet101(pretrained=True, progress=False)
  model = model.eval()
  model = SegmentationModelOutputWrapper(model)
  if torch.cuda.is_available():
    model = model.cuda()

  if args.use_cuda:
    model = model.cuda()

  target_layers = [model.model.backbone.layer4]

  if args.method not in methods:
    raise Exception(f"Method {args.method} not implemented")

  if args.method == "ablationcam":
    cam = methods[args.method](model=model,
                                  target_layers=target_layers,
                                  use_cuda=args.use_cuda,
                                  reshape_transform=reshape_transform,
                                  ablation_layer=AblationLayerVit())
  else:
    cam = methods[args.method](model=model,
                                  target_layers=target_layers,
                                  use_cuda=args.use_cuda,
                                  reshape_transform=reshape_transform)

  dataset = load_dataset("cats_vs_dogs", revision="main")
  
  
  
  
  sem_classes = [
      '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
  ]
  sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
  desired_category = sem_class_to_idx["cat"]
  
  
  
  
  
  
  for index in range(4):
    print(f'Processing {index}...')
    image = np.array(dataset["train"]["image"][index])
    rgb_img = np.float32(image) / 255
    image_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    
    if torch.cuda.is_available():
      image_tensor = image_tensor.cuda()
      
    output = model(image_tensor)
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    desired_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    desired_mask_uint8 = 255 * np.uint8(desired_mask == desired_category)
    desired_mask_float = np.float32(desired_mask == desired_category)

    both_images = np.hstack((image, np.repeat(desired_mask_uint8[:, :, None], 3, axis=-1)))
    # print(output.shape)
    cv2.imwrite(f'{args.output}/{index}_mask.jpg', both_images)