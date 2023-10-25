import matplotlib.pyplot as plt
import numpy as np

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



