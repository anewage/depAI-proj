import numpy as np
import argparse
import cv2
from PIL import Image
from datasets import load_dataset
from labels import get_top_categories
from masks import get_mask
from maps import get_map
from transformers import AutoImageProcessor, ViTForImageClassification


def process(model_to_certify, processor, dimension, dataset, threshold=0.8, partial=None):
  length = dataset['train'].num_rows
  if partial:
    length = partial
  sturdiness = [0] * length
  for index in range(partial):
    print('Processing ', index, '...')
    
    ## Process image
    raw_image = dataset["train"]["image"][index]
    image = np.array(raw_image)
    inputs_tensor = processor(images=image, return_tensors="pt")
    inputs_np = processor(images=image, return_tensors="np")
    
    ## Get predictions
    predicted_label = get_top_categories(model=model_to_certify, img_tensor=inputs_tensor.pixel_values[0])[0]
    true_label = dataset["train"]["labels"][index]
    print('True label: ', true_label, ', Predicted: ', predicted_label)
    
    ## Get mask & saliency map only if prediction is correct
    if (true_label == predicted_label):
      resized_image = cv2.resize(np.array(image), (dimension, dimension))

      mask = get_mask(resized_image)
      Image.fromarray(mask).save(f'output/{index}_mask.jpg')
      
      map = get_map(model=model_to_certify, predicted_labels=[predicted_label], inputs_tensor=inputs_tensor, image_resized=resized_image)
      grayscale = np.uint8(map[0][0] * 255)
      Image.fromarray(grayscale).save(f'output/{index}_map.jpg')
      
      ## Element-wise multiplication of mask and map
      ints = np.multiply(mask/255, grayscale*100/255)
      
      ## Compute the area of the intersection
      area = ints.sum()/(mask/255).sum()
      
      ## If the area is greater than 5%, the image is sturdy
      if (area > 0.05):
        sturdiness[index] = 1
      else:
        sturdiness[index] = 0
  
  response = sum(sturdiness) / len(sturdiness)
  if(response > threshold):
    return True
  else:
    return False



def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='google/vit-base-patch32-384', help='Huggingface transformer repo/name')
  parser.add_argument('--dimension', type=int, default=384, help='Transformer input dimension')
  parser.add_argument('--dataset', type=str, default='cats_vs_dogs', help='Huggingface dataset identifier')
  parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for robustness')
  parser.add_argument('--partial', type=int, default=2, help='Partial processing range')
  args = parser.parse_args()
  return args


  """
  the main function of the script

  example: python certify.py --model google/vit-base-patch32-384 --dimension 384 --dataset cats_vs_dogs --threshold 0.8 --partial 4
  """
if __name__ == "__main__":
  # model_to_certify = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384')
  # processor = AutoImageProcessor.from_pretrained('google/vit-base-patch32-384')
  # dataset = load_dataset("cats_vs_dogs", revision="main")
  # dimension = 384
  # threshold = 0.8
  args = get_args()
  model_to_certify = ViTForImageClassification.from_pretrained(args.model)
  processor = AutoImageProcessor.from_pretrained(args.model)
  dataset = load_dataset(args.dataset, revision="main")
  dimension = args.dimension
  threshold = args.threshold
  partial = args.partial
  if process(model_to_certify, processor, dimension, dataset, threshold, partial):
    print("Model is robust")
  else:
    print("Model is not robust")