import math
import numpy as np
from PIL import ImageOps, Image, ImageEnhance
import random
import torch
import torchvision.transforms as transforms
import augmentations

def aug(image, preprocess, aug_severity=3, mixture_width=3, mixture_depth=-1, all_ops=False):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    # depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
    #     1, 4)
    depth = np.random.randint(1, mixture_depth + 1 if mixture_depth > 0 else 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed



class AugMix(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess,aug_severity=1, no_jsd=True, IMAGE_SIZE=32):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.aug_severity = aug_severity
    augmentations.IMAGE_SIZE = IMAGE_SIZE

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess, aug_severity=self.aug_severity), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess, aug_severity=self.aug_severity),
                  aug(x, self.preprocess, aug_severity=self.aug_severity))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)
