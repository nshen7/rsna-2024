import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
import os

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


# Load image as tensor
def load_image_from_path(path):
    img = pydicom.dcmread(path)
    img = img.pixel_array
    img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255 # Pixel value between 0-255
    img = tv_tensors.Image(img) # [CHANNEL, HEIGHT, WIDTH]
    return img.double()

# ----------- Functions for disc detection -----------

# Tranform image
def get_transform_disc_detection():
    transforms = []
    transforms.append(T.Resize((250, 250), antialias=True))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

## Reminder of encoding rule
# LEVEL_LABELS = {
#     "L1/L2": 1,
#     "L2/L3": 2,
#     "L3/L4": 3,
#     "L4/L5": 4,
#     "L5/S1": 5
# }

# Prepare dataset
class RSNAMultipleBBoxesDataset(torch.utils.data.Dataset):
    def __init__(self, df, w, h_l1_l4, h_l5, transforms=get_transform_disc_detection(), limit=None):
        # Limit for debugging
        if limit:
            df = df.iloc[0:limit]
        
        self.df = df
        
        # Unique image_paths
        self.images_df = df[
          ['study_id', 'series_id', 'instance_number', 'image_path']
        ].drop_duplicates().reset_index(drop=True)

        self.transforms = transforms
        self.w, self.h_l1_l4, self.h_l5 = w, h_l1_l4, h_l5

    def __getitem__(self, idx):
        row = self.images_df.iloc[idx]
        target = {}

        # Image
        img = load_image_from_path(row['image_path'])
        w_orig, h_orig = img.shape[-1], img.shape[-2]
        target['img'] = img
        target['image_id'] = idx
        target['series_id'] = row['series_id']
        target['study_id'] = row['study_id']
        target['instance_number'] = row['instance_number']
        
        # Transform
        if self.transforms:
            img = self.transforms(img)

        w_resize, h_resize = img.shape[-1], img.shape[-2]
        w_ratio = w_resize / w_orig
        h_ratio = h_resize / h_orig
            
        target['boxes'] = []
        target['area'] = []
        target['labels'] = []
        target['iscrowd'] = []
        series_df = self.df[self.df['series_id'] == row['series_id']]

        for i,row in series_df.iterrows():
            # Label
            target['labels'].append(row['level_code'])

            # Bounding Boxes
            if row['level_code'] == 5:
                w = self.w
                h = self.h_l5
            else:
                w = self.w
                h = self.h_l1_l4
            # Here, I'm dislocating the box in such a way
            # that upper level discs are closer to the 
            # bottom of it's box and lower level discs 
            # are closer to the top of it's box.
            level = int(row['level'][1])
            x0 = (row['x'])*w_ratio - w*5/6
            x1 = (row['x'])*w_ratio + w*1/6

            if row['level_code'] == 5:
                y0 = (row['y'])*h_ratio - h*1/6
                y1 = (row['y'])*h_ratio + h*5/6
            else:
                y0 = (row['y'])*h_ratio - h*1/3
                y1 = (row['y'])*h_ratio + h*2/3

            target['boxes'].append(
              [x0, y0, x1, y1]
            )
            # Box area
            target['area'].append((x1-x0) * (y1-y0))

            # Instances with iscrowd=True will be ignored during evaluation.
            target['iscrowd'].append(False)
        
        target['area'] = torch.tensor(target['area'])
        target['labels'] = torch.tensor(target['labels'])#.squeeze(dim=-1)
        target['iscrowd'] = torch.tensor(target['iscrowd'])
        # print(target['labels'].shape)
        target['boxes'] = BoundingBoxes(
            target['boxes'],
            format='XYXY',
            dtype=torch.float32,
            canvas_size=img.shape[-2:]
        )
        return img, target

    def __len__(self):
        return len(self.images_df)

    
# ----------- Functions for severity classification -----------
    
## The swin transformer model (i.e., torchvision.models.swin_v2_t) does not contain a transform layer, 
## so we'd have to normalize it ourselves.
def get_transform_severity_classification(img_size = 224):
    transforms = []
    transforms.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)
    
class RSNACroppedImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=get_transform_severity_classification(), limit=None):
        # Limit for debugging
        if limit:
            df = df.iloc[0:limit]
        
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        file = row['cropped_image_path']
        img = torch.load(file).double()

        # Transform
        if self.transforms:
            img = self.transforms(img)
        
        # Label
        label = row['severity_code']

        return img, label

    def __len__(self):
        return len(self.df)
    

class RSNAUncroppedImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=get_transform_severity_classification(), limit=None):
        # Limit for debugging
        if limit:
            df = df.iloc[0:limit]
        
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        file = row['image_path']
        img = load_image_from_path(file)

        # Transform
        if self.transforms:
            img = self.transforms(img)
        
        # Label
        label = row['severity_code']

        return img, label

    def __len__(self):
        return len(self.df)