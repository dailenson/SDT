import random
from utils.util import normalize_xys
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from utils.util import corrds2xys

transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5), std = (0.5))
])

script={"CHINESE":['CASIA_CHINESE', 'Chinese_content.pkl'],
        'JANPANESE':['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
        "ENGLISH":['CASIA_ENGLISH', 'English_content.pkl']
        }

class ScriptDataset(Dataset):
    def __init__(self, root='data', dataset='CHINESE', is_train=True, num_img = 15):
        data_path = os.path.join(root, script[dataset][0])
        self.content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb')) #content samples
        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.is_train = is_train
        if self.is_train:
            lmdb_path = os.path.join(data_path, 'train') # online characters
            self.img_path = os.path.join(data_path, 'train_style_samples') # style samples
            self.num_img = num_img*2
            self.writer_dict = self.all_writer['train_writer']
        else:
            lmdb_path = os.path.join(data_path, 'test') # online characters
            self.img_path = os.path.join(data_path, 'test_style_samples') # style samples
            self.num_img = num_img
            self.writer_dict = self.all_writer['test_writer']
        if not os.path.exists(lmdb_path):
            raise IOError("input the correct lmdb path")
        
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        if script[dataset][0] == "CASIA_CHINESE" :
            self.max_len = -1  # Do not filter characters with many trajectory points
        else: # Japanese, Indic, English
            self.max_len = 150

        self.all_path = {}
        for pkl in os.listdir(self.img_path):
            writer = pkl.split('.')[0]
            self.all_path[writer] = os.path.join(self.img_path, pkl)

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            if self.max_len <= 0:
                self.indexes = list(range(0, self.num_sample))
            else:
                print('Filter the characters containing more than max_len points')
                self.indexes = []
                for i in range(self.num_sample):
                    data_id = str(i).encode('utf-8')
                    data_byte = txn.get(data_id)
                    coords = pickle.loads(data_byte)['coordinates']
                    if len(coords) < self.max_len:
                        self.indexes.append(i)
                    else:
                        pass

    def __getitem__(self, index):
        index = self.indexes[index]
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname']
        char_img = self.content[tag_char] # content samples
        char_img = char_img/255. # Normalize pixel values between 0.0 and 1.0
        writer = data['fname'].split('.')[0]
        img_path_list = self.all_path[writer]
        with open(img_path_list, 'rb') as f:
            style_samples = pickle.load(f)
        img_list = []
        img_label = []
        random_indexs = random.sample(range(len(style_samples)), self.num_img)
        for idx in random_indexs:
            tmp_img = style_samples[idx]['img']
            tmp_img = tmp_img/255.
            tmp_label = style_samples[idx]['label']
            img_list.append(tmp_img)
            img_label.append(tmp_label)
            
        img_list = np.expand_dims(np.array(img_list), 1) # [N, C, H, W], C=1
        coords = normalize_xys(coords) # Coordinate Normalization

        #### Convert absolute coordinate values into relative ones
        coords[1:, :2] = coords[1:, :2] - coords[:-1, :2]

        writer_id = self.writer_dict[fname]
        character_id = self.char_dict.find(tag_char)
        label_id = []
        for i in range(self.num_img):
            label_id.append(self.char_dict.find(img_label[i]))
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id]),
                'img_list': torch.Tensor(img_list),
                'char_img': torch.Tensor(char_img),
                'img_label': torch.Tensor([label_id])}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['coords'].shape[0] for s in batch_data]) + 1
        output = {'coords': torch.zeros((bs, max_len, 5)), # (x, y, state_1, state_2, state_3)
                  'coords_len': torch.zeros((bs, )),
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,)),
                  'img_list': [],
                  'char_img': [],
                  'img_label': []}
        output['coords'][:,:,-1] = 1 # pad to a fixed length with pen-end state
        
        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords']
            output['coords'][i, 0, :2] = 0 ### put pen-down state in the first token
            output['coords_len'][i] = s
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']
            output['img_list'].append(batch_data[i]['img_list'])
            output['char_img'].append(batch_data[i]['char_img'])
            output['img_label'].append(batch_data[i]['img_label'])
        output['img_list'] = torch.stack(output['img_list'], 0) # -> (B, num_img, 1, H, W)
        temp = torch.stack(output['char_img'], 0)
        output['char_img'] = temp.unsqueeze(1)
        output['img_label'] = torch.cat(output['img_label'], 0)
        output['img_label'] = output['img_label'].view(-1, 1).squeeze()
        return output

"""
 loading generated online characters for evaluating the generation quality
"""
class Online_Dataset(Dataset):
    def __init__(self, data_path):
        lmdb_path = os.path.join(data_path, 'test')
        print("loading characters from", lmdb_path)
        if not os.path.exists(lmdb_path):
            raise IOError("input the correct lmdb path")

        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.writer_dict = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            self.indexes = list(range(0, self.num_sample))

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            character_id, coords, writer_id, coords_gt = data['character_id'], \
                data['coordinates'], data['writer_id'], data['coords_gt']
        try:
            coords, coords_gt = corrds2xys(coords), corrds2xys(coords_gt)
        except:
            print('Error in character format conversion')
            return self[index+1]
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id]),
                'coords_gt': torch.Tensor(coords_gt)}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['coords'].shape[0] for s in batch_data])
        max_len_gt = max([h['coords_gt'].shape[0] for h in batch_data])
        output = {'coords': torch.zeros((bs, max_len, 5)),  # preds -> (x,y,state) 
                  'coords_gt':torch.zeros((bs, max_len_gt, 5)), # gt -> (x,y,state) 
                  'coords_len': torch.zeros((bs, )),
                  'len_gt': torch.zeros((bs, )),
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,))}

        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords']
            h =  batch_data[i]['coords_gt'].shape[0]
            output['coords_gt'][i, :h] = batch_data[i]['coords_gt']
            output['coords_len'][i], output['len_gt'][i] = s, h
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']
        return output