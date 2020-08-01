from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class DirDataSet(Dataset):
    def __init__(self, main_dir, transform=transforms.ToTensor):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        small = image.copy()
        small.thumbnail((320,320), Image.ANTIALIAS)
        tensor_image = self.transform()(image) * 255
        tensor_small = self.transform()(small) * 255
        return tensor_small, tensor_image