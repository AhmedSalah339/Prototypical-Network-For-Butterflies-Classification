from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import os
from torchvision import transforms
from PIL import Image
from pathlib import Path

import pickle
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
class Butterfly200DataSet(Dataset):
    # butterfly 200 dataset
    # uses the paths of preprocessed images saved as tensors
    def __init__(self,split_dict_path=None,paths=None,precache=True,mode='train'):
        super(Butterfly200DataSet, self).__init__()
        self.mode = mode
        
        # raise error if no split dictionary or paths are supplied to use in the training 
        if ( split_dict_path is None) and ( paths is None):
            raise Exception('a split dict or paths should be passed to the Butterfly200DataSet')
        #initializations
        self.y = []
        # load the split dictionary and get the split paths
        if split_dict_path is not None:
            split_dict = load_dict(split_dict_path)
            self.paths = split_dict[mode]
        elif paths is not None:
            self.paths = paths
        # get the labels in self.y
        for path in self.paths:
            try:
                # use the naming in the butterfly dataset to get the class number
                cls = int(os.path.basename(path.split('.')[0]))-1
                self.y.append(cls)
            except:
                # no class number supplied (mode has to be not training)
                if self.mode == 'train':
                    raise Exception('No classes can be inferred\n The mode is "train" but the naming is wrong ')
                self.y.append(-1)
        
        # prepare the cashing
        self.imgs_cashe = {}
        # cashe the examples
        if precache:
            for i in tqdm(range(len(self.paths))):
                img = self.load_img(self.paths[i])
                label = self.y[i]
                self.imgs_cashe[i] = (img,label)
    def load_img(self,path):
        # load the torch tensor
        suf = Path(path).suffix

        if suf =='.pt':
            x = torch.load(path)
        else:
            x = self.process_image(path)
            
        return x
    def process_image(self,path):
        x = Image.open(path)
        if self.mode == 'train':
                aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomErasing(p=0.5,scale=(0.05, 0.05)),
                    transforms.RandomAffine(90,(0.1,0.1))])
                
                x = aug(x)
        preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        if len(x.mode)<3:
            raise Exception("An image with no colored channels at "+path)
        x = preprocess(x)
        # torch.save(x, path[:-4]+'sub.pt')
        return x
        
    def __getitem__(self, index):
        # if the example is cashed return it
        if  index in self.imgs_cashe:
            return self.imgs_cashe[index][0].cuda(non_blocking=True),self.imgs_cashe[index][1]
        else:
            # if not cashed load it and return it
            img = self.load_img(self.paths[index])
            img = img.cuda(non_blocking=True)
            label = self.y[index]
        return img,label
    def __len__(self):
        return len(self.paths)
