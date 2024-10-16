# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================

'''
    Preprocess Pix2Pix dataset.
'''

import os
import numpy as np
from PIL import Image
import mindspore
from mindspore import dataset as de
import mindspore.dataset.vision.c_transforms as C
from ..utils.config import get_args

args = get_args()

class pix2pixDataset():
    '''
        Define train dataset.
    '''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(f"这是读取前{self.list_files}")
        # 过滤出图片文件
        self.image_files = [file for file in self.list_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 排序文件名
        self.image_files.sort(key=lambda x: int(x.split('.')[0]))
        self.list_files =self.image_files


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        # 确保路径是一个文件而不是目录
        if not os.path.isfile(img_path):
            raise ValueError(f"Invalid file path: {img_path}")

        AB = Image.open(img_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((w2, 0, w, h))
        B = AB.crop((0, 0, w2, h))

        A = A.resize((args.load_size, args.load_size))
        B = B.resize((args.load_size, args.load_size))

        transform_params = get_params(A.size)
        A_crop = crop(A, transform_params, size=256)
        B_crop = crop(B, transform_params, size=256)

        return A_crop, B_crop

def get_params(size=(256, 256)):
    '''
        Get parameters from images.
    '''
    w, h = size
    new_h = h
    new_w = w
    new_h = new_w = args.load_size      # args.load_size

    x = np.random.randint(0, np.maximum(0, new_w - 256))
    y = np.random.randint(0, np.maximum(0, new_h - 256))

    return (x, y)

def crop(img, pos, size=256):
    '''
        Crop the images.
    '''
    ow = oh = args.load_size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        return img
    return img

def sync_random_Horizontal_Flip(input_images, target_images):
    '''
      Randomly flip the input images and the target images.
    '''
    seed = np.random.randint(0, 2000000000)
    mindspore.set_seed(seed)
    op = C.RandomHorizontalFlip(prob=0.5)
    out_input = op(input_images)
    mindspore.set_seed(seed)
    op = C.RandomHorizontalFlip(prob=0.5)
    out_target = op(target_images)
    return out_input, out_target

def create_train_dataset(dataset):
    '''
      Create train dataset.
    '''

    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3

    trans = [
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    train_ds = de.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False)

    train_ds = train_ds.map(operations=[sync_random_Horizontal_Flip], input_columns=["input_images", "target_images"])

    train_ds = train_ds.map(operations=trans, input_columns=["input_images"])
    train_ds = train_ds.map(operations=trans, input_columns=["target_images"])

    train_ds = train_ds.batch(1, drop_remainder=True)
    train_ds = train_ds.repeat(1)

    return train_ds

class pix2pixDataset_val():
    '''
       Define val dataset.
    '''

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(f"这是读取前{self.list_files}")
        # 过滤出图片文件
        self.image_files = [file for file in self.list_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 排序文件名
        self.image_files.sort(key=lambda x: int(x.split('.')[0]))
        self.list_files =self.image_files
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        AB = Image.open(img_path).convert('RGB')
        w, h = AB.size

        w2 = int(w / 2)
        A = AB.crop((w2, 0, w, h))
        B = AB.crop((0, 0, w2, h))

        return A, B

def create_val_dataset(dataset):
    '''
      Create val dataset.
    '''

    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3

    trans = [
        C.Resize((256, 256)),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    val_ds = de.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False)

    val_ds = val_ds.map(operations=trans, input_columns=["input_images"])
    val_ds = val_ds.map(operations=trans, input_columns=["target_images"])
    val_ds = val_ds.batch(1, drop_remainder=True)
    val_ds = val_ds.repeat(1)

    return val_ds
import os
from PIL import Image

class SingleImageDataset():
    '''
       Define a dataset for a single image.
    '''

    def __init__(self, root_dir):
        self.img_path = root_dir

    def __len__(self):
        return 1

    def __getitem__(self, index):
        img = Image.open(self.img_path).convert('RGB')
        w, h = img.size

        # 创建一个512x512的纯白图片
        white_img = Image.new('RGB', (512, 512), (255, 255, 255))

        # 将原图和纯白图片拼接在一起，形成1024x512的图片
        combined_img = Image.new('RGB', (1024, 512))
        combined_img.paste(white_img, (0, 0))
        combined_img.paste(img, (512, 0))

        # 裁剪图片
        A = combined_img.crop((512, 0, 1024, 512))
        B = combined_img.crop((0, 0, 512, 512))

        return A, B
