from transformers import ViTForImageClassification
from torch import Tensor
def get_top_categories(model: ViTForImageClassification, img_tensor: Tensor, top_k=5):
  logits = model(img_tensor.unsqueeze(0)).logits
  indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
  return indices

# model.config.id2label[i]