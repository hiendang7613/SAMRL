from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, data_dir,transform):
      super().__init__(data_dir,transform)
      new_indexes = sorted(range(len(self.imgs)), key=lambda k: int(os.path.basename(self.imgs[k][0]).split('.')[0]) )
      self.imgs = [self.imgs[i] for i in new_indexes]
      self.samples = [self.samples[i] for i in new_indexes]
      self.targets = [self.targets[i] for i in new_indexes]
      pass
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

