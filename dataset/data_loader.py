import os, copy
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import tqdm


def reset_tensor(list_tensor):
    for tensor in list_tensor:
        try:
            del tensor
        except:
            pass
    torch.cuda.empty_cache()


def load_data(partition):
    # download()
    DATA_DIR = 'data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def _load_data_file(name):
    f = h5py.File(name, 'r')
    data = f["data"][:]
    label = f["label"][:]
    return data, label

class UnpairedDataset(Dataset):
    def __init__(self, num_points=1024, train=True, io=None, is_debug=False,
                 n_sample=2, is_uniform=False): #minor edit to adjust with our previous loader
        log_str = ''
        if train:
            partition = 'train'
        else:
            partition = 'test'
        if is_debug:
            n_data = 128
            # self.points = np.zeros((64, 1024, 3), dtype=np.float)
            temp = np.arange(n_data).astype(int)
            temp = np.reshape(temp,(n_data,1,1))
            self.points = np.tile(temp,(1,1024,3))
            self.labels = np.random.randint(low=0, high=4, size=(n_data,1), dtype=int)
            log_str = f"N samples: {len(self.labels)}"
        else:
            self.points, self.labels = load_data(partition)
            log_str = f"N samples: {len(self.labels)}"
        self.num_points = num_points
        self.partition = partition
        self.attentions = np.zeros((len(self.points),self.num_points,n_sample))
        self.already_mixing = False
        print(f"Load data: {partition}")
        if io != None:
            io.cprint(log_str)
        else:
            print(log_str)

    def init_mixing(self, model):
        model = model.eval()
        if not self.already_mixing:
            for i in tqdm.tqdm(range(len(self.points)), desc='Init mixing'):
                sample_i = torch.from_numpy(self.points[i][:self.num_points]).cuda()
                sample_i = sample_i.unsqueeze(dim=0).type(torch.float)
                sample_i = sample_i.permute(0, 2, 1)
                logits, attention = model(sample_i, is_return_attn=True)
                self.attentions[i] = attention.squeeze().detach().cpu().numpy()
            self.already_mixing = True
        else:
            raise Exception("Mixing is already initialized before!")
        reset_tensor([sample_i, logits, attention, model])

    def __getitem__(self, item):
        pointcloud = self.points[item][:self.num_points]
        label = self.labels[item]
        if self.partition == 'train':
            idc = np.arange(len(pointcloud))
            np.random.shuffle(idc)
            pointcloud = pointcloud[idc]
            if self.already_mixing:
                attention = self.attentions[item]
                attention = attention[idc]
                attention = torch.from_numpy(attention).float()
        label = torch.from_numpy(label).type(torch.LongTensor)
        pointcloud = torch.from_numpy(pointcloud).float()
        if self.already_mixing:
            return pointcloud, label, attention
        else:
            return pointcloud, label

    def __len__(self):
        return self.points.shape[0]

class PairedDataset(UnpairedDataset):
    def __init__(self, num_points=1024, n_sample=2, train=True,
                 io=None, is_debug=False, is_uniform=False): #mixup_mode ops: 'intra', 'inter', 'full'
        super(PairedDataset, self).__init__(num_points=num_points, train=train, io=io,
                                            is_debug=is_debug, n_sample=n_sample, is_uniform=is_uniform)
        self.intra_class = True
        self.n_sample = n_sample
        self.unique_cls = np.unique(self.labels).tolist()
        self.unique_cls_temp = copy.deepcopy(self.unique_cls)
        self.per_class_idx = {}
        for cls in (self.unique_cls):
            self.per_class_idx[cls] = []
        for idx, cls in enumerate(self.labels.reshape((-1))):
            self.per_class_idx[cls].append(idx)
        self.per_class_idx_ori = copy.deepcopy(self.per_class_idx)
        if n_sample > len(self.unique_cls):
            print(f"n_sample: {n_sample}, n_class: {len(self.unique_cls)}")
            raise Exception("n_sample cannot larger than n_class")
        self.io = io

    def __getitem__(self, idx0):
        point_list = []
        label_list = []
        attn_list = []

        #get the first sample
        data = super(PairedDataset, self).__getitem__(idx0)
        if self.already_mixing:
            point, label, attn = data
            attn_list.append(attn)
        else:
            point, label = data
        point_list.append(point)
        label_list.append(label)
        label = label.item()
        ## remove the used idx from next pair candidate
        # try:
        #     self.per_class_idx[label].remove(idx0)
        # except:
        #     pass

        #get next sample class label in the pair
        current_label = {}; current_idx = [idx0]
        if self.intra_class: #avoid self-pair
            try:
                self.per_class_idx[label].remove(idx0)
            except: #the idx0 can be used in the previous pair and deleted
                pass

        for i in range(self.n_sample-1):
            if self.intra_class: #get intra-class label
                pass
            else: # get the inter-class pair label
                cand_class = copy.deepcopy(self.unique_cls)
                for item in current_label:
                    cand_class.remove(item)
                label = np.random.choice(cand_class)

            #get next sample's index in the pair
            try:
                idx = np.random.choice(self.per_class_idx[label])
            except: #in case the idx list is already empty, clone from the original list
                self.per_class_idx[label] = copy.deepcopy(self.per_class_idx_ori[label])
                if self.intra_class: #avoid self-pair
                    for item in current_idx:
                        try:
                            self.per_class_idx[label].remove(item)
                        except: #safety (pass) and debug purpose (write the related variable values)
                            str1 = " "
                            temp_list_str = [str(item) for item in self.per_class_idx[label]]
                            self.io.cprint(f"Issue intra class - class candidate: {str1.join(temp_list_str)}")
                            str1 = " "
                            temp_list_str = [str(item) for item in current_idx]
                            self.io.cprint(f"Issue intra class - item to be removed: {str1.join(temp_list_str)}")
                            self.io.cprint(f"Issue intra class - class label: {label}")
                idx = np.random.choice(self.per_class_idx[label])
            current_idx.append(idx)

            #get next sample in the pair
            data = super(PairedDataset, self).__getitem__(idx)
            if self.already_mixing:
                point, label, attn = data
                attn_list.append(attn)
            else:
                point, label = data
            point_list.append(point)
            label_list.append(label)
            label = label.item()

            try: #exclude the 'idx' for the next pair candidate
                self.per_class_idx[label].remove(idx)
            except:
                pass

        if self.already_mixing:
            return point_list, label_list, attn_list
        else:
            return point_list, label_list

    def __len__(self):
        return self.points.shape[0]