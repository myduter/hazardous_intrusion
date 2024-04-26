import math

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.utils.data as tordata

from .network import SetNet



class Model:
    def __init__(self,
                 hidden_dim,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 frame_num,
                 model_name,
                 data_source,
                 img_size=64):
        self.save_name = save_name
        self.data_source = data_source
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size
        self.sample_type = 'all'
        
        self.encoder = SetNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)

        

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        gpu_num = min(1, batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
                            len(frame_sets[i])
                            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                            if i < batch_size
                            ] for _ in range(gpu_num)]
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
        seqs = [[
                    np.concatenate([
                                        seqs[i][j]
                                        for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                        if i < batch_size
                                        ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
        seqs = [np.asarray([
                                np.pad(seqs[j][_],
                                        ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                        'constant',
                                        constant_values=0)
                                for _ in range(gpu_num)])
                for j in range(feature_num)]
        batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def ts2var(self, x):
        return autograd.Variable(x)
    
    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.data_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, _ = self.encoder(*seq, batch_frame)
            feature_list.append(feature.data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

