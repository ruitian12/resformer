# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import torch.distributed as dist
from samplers import RASampler, SubsetSampler
import os.path as op
from PIL import Image
import numpy as np
import base64
import io


def img_from_base64(imagestring, color=True):
    img_str = base64.b64decode(imagestring)
    try:
        if color:
            r = Image.open(io.BytesIO(img_str)).convert('RGB')
            return r
        else:
            r = Image.open(io.BytesIO(img_str)).convert('L')
            return r
    except:
        return None


class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
    

    def _ensure_lineidx_loaded(self):
        # if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
        #     generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
    


class TSVInstance(Dataset):
    def __init__(self, tsv_file, transform, ftype='.tsv', post_transform_list = None):
        self.tsv = TSVFile(tsv_file + ftype)
        self.transform = transform
        self.post_transform_list = post_transform_list
       
    def __getitem__(self, index):
        row = self.tsv.seek(index)
        img = img_from_base64(row[-1])
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.post_transform_list is None:
            trans_img = [img]
        else:
            trans_img = []
            for tf in self.post_transform_list:
                trans_img.append(tf(img))
        idx = int(row[1])
        label = torch.from_numpy(np.array(idx, dtype=np.int64))
        return trans_img, label

    def __len__(self):
        return self.tsv.num_rows()

    def switch_class(self, class_name):
        self.tsv.set_class(class_name)


class MultiResDataset(ImageFolder):
    def __init__(self, root, transform, post_transform):
        super().__init__(root, transform = transform)
        self.post_transform = post_transform

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        trans_img = []
        if self.post_transform is not None:
            for tf in self.post_transform:
                trans_img.append(tf(sample))
        else:
            trans_img = [sample]
        return trans_img, target


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder



def build_transform(is_train, args):
    assert torch.all(torch.sort(torch.Tensor(args.input_size), descending = True).values == torch.Tensor(args.input_size))
    resize_im = args.input_size[0] > 32
    input_size = args.input_size[0]
    
    if is_train:
        if args.multi_res and args.sep_aug:
            post_t_list = []
            for i, res_sz in enumerate(args.input_size):
                transform = create_transform(
                    input_size=res_sz,
                    is_training=True,
                    color_jitter=args.color_jitter,
                    auto_augment=None if args.aa == 'none' else args.aa,
                    interpolation=args.train_interpolation,
                    re_prob=args.reprob,
                    re_mode=args.remode,
                    re_count=args.recount,
                )
                if not resize_im:
                    transform.transforms[0] = transforms.RandomCrop(
                        res_sz, padding=4)
                post_t_list.append(transform)

            return None, post_t_list        
        else:
            transform = create_transform(
                input_size=input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=None if args.aa == 'none' else args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    input_size, padding=4)
            
            if args.multi_res:
                post_t_list = []
                if args.reprob > 0:
                    t1 = transform.transforms.pop(-2)
                    t2 = transform.transforms.pop(-1)
                else:
                    t1 = transform.transforms.pop(-1)
                    t2 = None
                
                for i, res_sz in enumerate(args.input_size):
                    t = []
                    if i > 0:
                        t.append(transforms.Resize(res_sz, interpolation=InterpolationMode.BICUBIC),)
                    t.append(t1)
                    if t2 is not None:
                        t.append(t2)
                    post_t_list.append( transforms.Compose(t))
                return transform, post_t_list
            
            else:
                return transform, None
    
    t_list = []

    if args.multi_res:
        for res_sz in args.input_size:
            t = []
            if resize_im:
                size = int((256 / 224) * res_sz)
                t.append(
                    transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(res_sz))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            t_list.append(transforms.Compose(t))
        return None, t_list
    
    else:
        t = []
        if resize_im:
            size = int((256 / 224) * input_size)
            t.append(
                transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t), None


def build_imagenet_dataloader(is_train, args):
    transform, post_transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    
    if args.load_tsv_data:
        dataset = TSVInstance(root,  transform=transform, post_transform_list= post_transform)
    else:
        dataset = MultiResDataset(root,  transform=transform, post_transform=post_transform)
        
    if is_train:
        if args.distributed:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()
            if args.repeated_aug:
                sampler_train = RASampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            # elif args.subset_aug and args.multi_res:
                # sampler_train = SubsetSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, num_subset=len(args.input_size))
            else:
                sampler_train = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True )
            data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=args.pin_mem,drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader( dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    else:
        if args.distributed:
            if args.dist_eval:
                num_tasks = dist.get_world_size()
                global_rank = dist.get_rank()
                sampler_val = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset)  
            # data_loader = torch.utils.data.DataLoader(dataset, sampler= sampler_val, batch_size=int(args.batch_size), shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_mem, drop_last=False)
            data_loader = torch.utils.data.DataLoader(dataset, sampler= sampler_val, batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_mem, drop_last=False)
        else:
            # data_loader = torch.utils.data.DataLoader( dataset, batch_size= int(args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
            data_loader = torch.utils.data.DataLoader( dataset, batch_size= int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
            

    return dataset, data_loader

