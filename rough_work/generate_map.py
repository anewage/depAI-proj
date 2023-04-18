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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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

  model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
  model.eval()

  if args.use_cuda:
    model = model.cuda()

  target_layers = [model.blocks[-1].norm1]

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

  entries = os.listdir(args.directory)
  for f in entries:
    print(f'Processing {f}...')
    rgb_img = cv2.imread(f'{args.directory}/{f}', 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    
    
    # We will show GradCAM for the "Egyptian Cat" and the 'Remote Control" categories:
    targets = [ClassifierOutputTarget(category_name_to_index(model, "Egyptian cat")),
                       ClassifierOutputTarget(category_name_to_index(model, "remote control, remote"))]
    
    
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.output}/{f}_{args.method}_cam.jpg', cam_image)