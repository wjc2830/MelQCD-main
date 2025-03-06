import torch.utils.data as data
import os
import torch
import numpy as np
import json
from einops import rearrange
import torch.nn.functional as F


def code2emb_qmel(code, num_quan=3, num_win=8):
    # Code: Pytorch Tensor with shape = L * (NumWin//4)
    # Emb: Pytorch Tensor with shape = L * NumWin
    assert num_win == code.shape[1] * 4
    assert num_quan % 2 == 1
    num_side = (num_quan - 1)//2
    code = code.unsqueeze(2)
    codes = []
    for i in range(4):
        codes.append(code % num_quan)
        code = code // num_quan
    emb = torch.cat(codes, dim=2) - num_side
    emb = rearrange(emb, 'w a b -> w (a b)')
    return emb


class VGGSoundQmelDataset(data.Dataset):
    
    def __init__(self, mode, root_path, num_quan=3, num_win=8):
        
        self.mode = mode
        self.root_path = root_path
        self.video_index = []
        if self.mode == 'train':
            f = open(os.path.join(self.root_path, 'train.txt'), 'r')
            lines = f.readlines()
            f.close()
            
        else:
            f = open(os.path.join(self.root_path, 'test.txt'), 'r')
            lines = f.readlines()
            f.close()

        for line in lines:
            self.video_index.append(line.strip())
        self.num_samples = len(self.video_index)
        
        self.num_quan = num_quan
        self.num_win = num_win
                
        self.vfeat_root = os.path.join(self.root_path, 'syncformer')
        
        self.mean_root = os.path.join(self.root_path, 'Qmel_{}_{}'.format(num_quan, num_win), 'Mean')
        self.std_root = os.path.join(self.root_path, 'Qmel_{}_{}'.format(num_quan, num_win), 'Std')
        self.emb_root = os.path.join(self.root_path, 'Qmel_{}_{}'.format(num_quan, num_win), 'Emb')
        self.code_root = os.path.join(self.root_path, 'Qmel_{}_{}'.format(num_quan, num_win), 'Code')

        print(f'[Mode {mode}]: {self.num_samples} videos.')

        with open(os.path.join(self.root_path, 'length_info.json'), 'r') as f:
            self.length_info = json.load(f)

    def __getitem__(self, index):
        video_index = self.video_index[index]

        # 25fps video feature
        vfeat = np.load(os.path.join(self.vfeat_root, '{}.npy'.format(video_index)))
        vfeat = torch.from_numpy(vfeat)
        
        # 100fps audio feature
        mean = np.load(os.path.join(self.mean_root, '{}.npy'.format(video_index)))
        mean = torch.from_numpy(mean)

        std = np.load(os.path.join(self.std_root, '{}.npy'.format(video_index)))
        std = torch.from_numpy(std)

        emb = np.load(os.path.join(self.emb_root, '{}.npy'.format(video_index)))
        emb = torch.from_numpy(emb)
        if emb.shape[0] != mean.shape[0]:
            emb = F.pad(emb, (0, 0, 0, mean.shape[0]-emb.shape[0]), value=0)
        
        code = np.load(os.path.join(self.code_root, '{}.npy'.format(video_index)))
        code = torch.from_numpy(code)
        
        # 25fps signal effect length
        effect_length = min(vfeat.shape[0], code.shape[0]//4, 250)
        
        effect_length_audio = effect_length * 4

        vfeat = F.pad(vfeat[:effect_length], (0, 0, 0, 250-effect_length), value=0)
        mean = F.pad(mean[:effect_length_audio], (0, 250 * (effect_length_audio//effect_length) - effect_length_audio), value=0)
        std = F.pad(std[:effect_length_audio], (0, 250 * (effect_length_audio//effect_length) - effect_length_audio), value=0)
        emb = F.pad(emb[:effect_length_audio], (0, 0, 0, 250 * (effect_length_audio//effect_length) - effect_length_audio), value=0)
        code = F.pad(code[:effect_length_audio], (0, 0, 0, 250 * (effect_length_audio//effect_length) - effect_length_audio), value=0)

        assert emb.size() == (1000, self.num_win), ('emb', emb.size())
        assert code.size() == (1000, self.num_win//4), ('code', code.size())
        assert mean.size() == (1000,), ('mean', mean.size())
        assert std.size() == (1000,), ('std', std.size())
                
        return effect_length, vfeat, mean, std, emb, code
        
                
    def __len__(self):
        return self.num_samples


class VGGSoundQmelDataset_test(data.Dataset):
    
    def __init__(self, video_path, num_quan=3, num_win=8):
        
        self.root_path = os.path.dirname(video_path)
        self.video_index = []
        
        lines = sorted(os.listdir(video_path))
        for line in lines:
            self.video_index.append(line[:-4])
        print(self.video_index)
        self.num_samples = len(self.video_index)
        
        self.num_quan = num_quan
        self.num_win = num_win
                
        # feat_type = 4: SyncFormer/pad
        self.vfeat_root = os.path.join(self.root_path, 'syncformer')

        print(f'{self.num_samples} videos.')

    def __getitem__(self, index):
        video_index = self.video_index[index]
        
        # 25fps video feature
        vfeat = np.load(os.path.join(self.vfeat_root, '{}.npy'.format(video_index)))
        vfeat = torch.from_numpy(vfeat)
        
        # 25fps signal effect length
        effect_length = min(vfeat.shape[0], 250)
        
        # crop and pad
        vfeat = F.pad(vfeat[:effect_length], (0, 0, 0, 250-effect_length), value=0)
                
        return effect_length, vfeat, video_index
        
                
    def __len__(self):
        return self.num_samples