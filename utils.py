import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import cv2
import torch

def show_masks(images, masks):
  ncols=3
  nrows=(len(images)+ncols-1)//ncols
  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6*len(images)))
  for i, (image, anns) in enumerate(zip(images, masks)):
    n_row, n_col = i//3, i%3
    if nrows==1:
      ax = axs[n_col]
    else:
      ax = axs[n_row, n_col]
    ax.imshow(image)
    show_anns(ax, anns)
    ax.axis('off')
  plt.show()
  

def show_anns(ax, anns):
    if not anns:
        return

    # Find the largest annotation
    max_area = max(ann['area'] for ann in anns)
    largest_ann = [ann for ann in anns if ann['area'] == max_area][0]

    img = np.zeros((largest_ann['segmentation'].shape[0], largest_ann['segmentation'].shape[1], 4))
    
    for ann in anns:
        m = ann['segmentation']
        color_mask = np.random.rand(3)
        img[m] = np.append(color_mask, 0.75)
    
    ax.imshow(img)

def save_segments(data_dir, images, img_masks):
  max_k = sum([ len(masks) for masks in img_masks])
  with tqdm(total=max_k, leave=False) as step:
    for image, masks in zip(images, img_masks):
      for k, mask in enumerate(masks):
        new_image = np.expand_dims(mask['segmentation'],-1)*image
        coord_x = np.sum(masks[k]['segmentation'],axis=1)
        x_min, x_max = np.where(coord_x>0)[0][0], np.where(coord_x>0)[0][-1]
        coord_y = np.sum(masks[k]['segmentation'],axis=0)
        y_min, y_max = np.where(coord_y>0)[0][0], np.where(coord_y>0)[0][-1]
        new_image = new_image[x_min: x_max, y_min: y_max]
        output_path = data_dir + 'unknow/' + str(step.n) + '.png'
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(output_path, new_image)
        step.update(1)
  print(f"Saved {max_k} images to {data_dir}.")

def get_embeddings(dataset, clip_model):
  embeddings = []
  labels = []
  paths = []
  for image, label, path in tqdm(dataset):
      with torch.no_grad():
        outputs = clip_model.encode_image(image.unsqueeze(0).to(device))
      embeddings.append(outputs)
      labels.append(label)
      paths.append(path)
  embeddings = torch.cat(embeddings, dim=0)
  labels = torch.tensor(labels, dtype=torch.int32)
  return embeddings, labels, paths

