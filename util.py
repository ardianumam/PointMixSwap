import numpy as np
import torch, os, json
import torch.nn.functional as F
import yaml
import sklearn.metrics as metrics
import tqdm
from itertools import chain
import open3d as o3d
import time
from dataset import provider


def dumb_config(cfg, args):
    # read yaml config file to python dict
    with open(args.config) as file:
        yaml_from_file = yaml.load(file, Loader=yaml.FullLoader)
    # delete cfg member that don't exist in yaml config file(1st parent)
    cfg_copy = (cfg.dump())
    cfg_copy = dict(yaml.safe_load(cfg_copy))  # parse yaml string to yaml to dict
    keys = list(cfg_copy.keys()).copy()
    for key_ in keys:  # parent key
        if key_ not in yaml_from_file.keys():
            del cfg_copy[key_]
        else:
            keys_child = list(cfg_copy[key_].keys()).copy()
            for key_child_ in keys_child:  # child key
                if key_child_ not in yaml_from_file[key_].keys():
                    del cfg_copy[key_][key_child_]

    # dump to file
    with open(f"{cfg_copy['EXP']['WORKING_DIR']}/{cfg_copy['EXP']['NAME']}/config.yaml", "w") as outfile:
        yaml.dump(cfg_copy, outfile, default_flow_style=False)
    return yaml.dump(cfg_copy)

def reset_tensor(list_tensor):
    for tensor in list_tensor:
        try:
            del tensor
        except:
            pass
    torch.cuda.empty_cache()

def get_color_with_idx(idx):
    '''
    give scalar integer index, gives rgb color with 0-1 range
    '''
    # color_array = np.array([[0,77,155],  #blue
    #                         [155, 8, 81],  # maroon red
    #                         [237, 128, 14],  # orange
    #                         [12,229,135],  #magenta
    #                         [152,60,255]]).astype(float)/255.0 #purple
    from matplotlib import pyplot as plt
    cmap = plt.get_cmap(plt.cm.Set2)
    color_array = cmap.colors
    if idx >= len(color_array):
        raise Exception("Requested color index is more than the available color index!")
    return color_array[idx]

def get_class_mapping_modelnet40():
    # read modelnet dataset info to dictionary = {'class_number': ['class_string', n_train, n_test]}
    class_mapping_file_path = 'assets/class_mapping_modelnet.json'
    with open(class_mapping_file_path, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping

debug_purpose_idx = 0
def swap_pair_n(data, attn, is_visualize=False, batch_idx=None, label=None, args=None, cfg=None, dir=''):
    """
    swap batch of "input data" based on the attention, with n_div (last attention dimension size) division.
    data: list of data, can be [data1] only or [data1,data2]
        data1 : [BS, n_pts, n_feature or emb_dim] -> data that will be mixed up based on attention division
        data2: [BS, n_pts, n_feature or emb_dim] -> another data that will be mixed up, use case: for 'original_feature'
                                                    before attention block, that will be used in residual attention addition ops.
    attn : [BS, n_pts, n_div]
    return:
        new_data: [BS, n_pts, 3]
    """
    data = [data]
    bs, n_pts, n_col = data[0].size()
    n_div = attn.size(-1)
    if bs%n_div != 0: raise Exception("Number of sample must be divisible by n_div!")
    n_pair = (int)(bs / n_div)
    #define the composition of new samples
    idx_perm = np.arange(n_div) if is_visualize else np.random.permutation(n_div)
    comp_idx = np.stack([np.roll(idx_perm,shift=i) for i in range(n_div)], axis=0)
    attn = torch.argmax(attn,dim=-1) #[BS, n_pts]
    attn = attn.detach().cpu().numpy()
    idx_helper_list = []
    if is_visualize:
        class_map = get_class_mapping_modelnet40()
        path = f"{cfg.EXP.WORKING_DIR}/{args.model_path}/{dir}"
        if not os.path.exists(path):
            os.makedirs(path)

    # build new sample indices
    t0 = time.time()
    new_sample_indices = []
    for pair_idx in range(n_pair):
        idx_helper = [sample_pair_idx*n_pair+pair_idx for sample_pair_idx in range(n_div)]
        temp_attn = np.stack([(attn[sample_pair_idx*n_pair+pair_idx]) for sample_pair_idx in range(n_div)], axis=0) #[n_div,n_point]
        for sample_pair_idx in range(n_div):
            ## compose a new sample indices from every samples in the pair
            temp_list = list(chain(*[list(np.argwhere(temp_attn[i]==comp_idx[sample_pair_idx][i]).reshape((-1))+(i*n_pts))
                                     for i in range(n_div)]))
            if len(temp_list) < 50: #if number of new sample indices is under threshold, use the original instead!
                temp_list = np.arange(n_pts).tolist()

            ## sample n_pts and resuffle the points order
            n_pts_temp_list = len(temp_list)
            if n_pts_temp_list >= n_pts:  # if the new sample has more points than or == n_pts
                idc = np.random.choice(temp_list, n_pts, replace=False)
            else:
                delta = n_pts - n_pts_temp_list
                idc_add = np.random.choice(temp_list, delta, replace=True if delta>n_pts_temp_list else False)
                idc = np.concatenate((np.asarray(temp_list), idc_add), axis=0)
                np.random.shuffle(idc)
            new_sample_indices.append(idc) #[BS as list, n_pts as 1D array]

        idx_helper_list.append(idx_helper) #[n_pair as list, n_div as list]
    t1 = time.time()
    # print("Swap index time:", t1 - t0)
    t0 = time.time()
    # define re-ordering new sample indices based on original input data
    idx_helper_list = list(chain(*idx_helper_list)) #[BS as list]
    idx_helper_list_invert = np.argsort(idx_helper_list) #[BS as 1D array]

    # gather new samples based on the constructed indices
    result = []
    new_sample_indices = np.stack(new_sample_indices, axis=0) # [BS, n_pts] --> indices of new samples
    ## re-oder based on the invert indices and gathered tensor
    new_sample_indices = new_sample_indices[idx_helper_list_invert] # [BS, n_pts] --> indices of new samples
    new_sample_indices = torch.from_numpy(new_sample_indices).cuda().unsqueeze(dim=2).repeat(1,1,n_col) #[BS, n_pts=1024, n_col]
    ## create helper indices for gathered data
    idx_helper_list = np.reshape(np.asarray(idx_helper_list),(n_pair,n_div)) #[n_pair, n_div]
    repeat_indices = [np.full((n_div),i) for i in range(n_pair)] #[n_pair, n_div]
    repeat_indices = np.asarray(repeat_indices).reshape((-1)) #[BS]
    idx_helper_list = idx_helper_list[repeat_indices] #[BS, n_div]
    for data_idx in range (len(data)): #repeat to all data in the data list
        temp_data = torch.cat([data[data_idx][idx_helper_list[:,i]] for i in range(n_div)], dim=1) # [BS, n_pts*n_div, n_col]
        ## re-oder based on the invert indices
        temp_data = temp_data[idx_helper_list_invert] # [BS, n_pts*n_div, n_col]
        ## gather the new samples
        result.append(torch.gather(temp_data,dim=1,index=new_sample_indices)) # list of [BS, n_pts, n_col]
        if is_visualize:
            attn = torch.from_numpy(attn)
            temp_attn = torch.cat([attn[idx_helper_list[:,i]] for i in range(n_div)], dim=1)
            temp_attn = temp_attn[idx_helper_list_invert]
            temp_attn = torch.gather(temp_attn, dim=1, index=new_sample_indices[:,:,0].cpu())

    t1 = time.time()
    # print("Swap gathering time:", t1 - t0)
    if is_visualize:
        new_sample_indices = new_sample_indices[:,:,0].cpu().numpy()
        # partition = np.floor(new_sample_indices / n_pts)  # this is a partition by sample, not by attention
        partition = temp_attn
        pc_all = result[0].cpu().numpy()
        for sample_idx in range(bs):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_all[sample_idx]) #only resulf of data[0] will be visualized!
            color = np.zeros((n_pts,3), dtype=float)
            for color_idx in range(n_div):
                mask = partition[sample_idx] == color_idx  # [n_pts]
                gen_color = get_color_with_idx(color_idx)
                color[mask] = gen_color
            pcd.colors = o3d.utility.Vector3dVector(color)
            class_string = class_map[str(label[sample_idx].item())][0]
            path_file = os.path.join(f"{cfg.EXP.WORKING_DIR}/{args.model_path}/{dir}",
                                     f"MIXED{n_div}_{class_string}_{batch_idx}_{str(sample_idx).zfill(4)}.ply")
            print(f"path_file_mixed-{sample_idx}: {path_file}")
            o3d.io.write_point_cloud(path_file, pcd)

    return result[0]


def one_epoch_train(cfg, model, train_loader, opt, criterion, device, epoch, is_mix_start):
    train_loss = 0.0
    count = 0.0
    acc_score = 0.0
    model.train()
    train_pred = []
    train_true = []
    train_pred_mixup = []
    train_true_mixup = []
    attn = None # declaration purpose
    for i, DATA in enumerate(tqdm.tqdm(train_loader, desc=f"Train {epoch}/{cfg.TRAIN.N_EPOCHS}")):
        for j in range (2 if is_mix_start else 1):
            if isinstance(DATA[-1], list):  # using pair loader
                data = torch.cat(DATA[0], dim=0).to(device)
                label = torch.cat(DATA[1], dim=0).view(-1).to(device)
            else:  # using standard loader
                data, label = DATA[0].to(device), DATA[1].to(device).squeeze()

            batch_size = data.size()[0]
            is_return_attn = True
            if j == 1: #mixup turn
                attn = torch.cat(DATA[2],dim=0).to(device)
                if cfg.TRAIN.MIXUP_LEVEL == 'input':
                    mixup_mode = 'input'
                elif cfg.TRAIN.MIXUP_LEVEL == 'feature':
                    mixup_mode = 'feature'
                elif cfg.TRAIN.MIXUP_LEVEL == 'both':
                    mixup_mode = np.random.choice(['feature','input'],1)[0]
                else:
                    raise Exception("Choose a correct MIXUP_LEVEL!")
                if mixup_mode == 'input':
                    data = swap_pair_n(data, attn) #[BS, n_pts, 3]
                data = translate_pointcloud_torch(data, dataset=cfg.EXP.DATASET)  # standard augmentations
                data = data.permute(0, 2, 1)  # [BS, 3, n_pts]
                logits, attn = model(data, is_feature_mixup_turn = (mixup_mode == 'feature'),
                                     given_attention = attn if (mixup_mode == 'feature') else None,
                                     is_return_attn = is_return_attn)

                preds = logits.max(dim=1)[1]
                train_true_mixup.append(label.cpu().numpy())
                train_pred_mixup.append(preds.detach().cpu().numpy())
            else: #non-mixup turn
                data = translate_pointcloud_torch(data, dataset=cfg.EXP.DATASET)  # standard augmentations
                data = data.permute(0, 2, 1)  # [BS, 3, n_pts]
                logits, attn = model(data, is_return_attn = is_return_attn)
                preds = logits.max(dim=1)[1]
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

            loss = criterion(logits, label, smoothing=cfg.TRAIN.LABEL_SMOOTH)
            opt.zero_grad()
            loss.backward()
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
    loss = train_loss*1.0/count
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc_score = metrics.accuracy_score(train_true, train_pred)
    balanced_acc_score = metrics.balanced_accuracy_score(train_true, train_pred)
    if is_mix_start:
        train_true_mixup = np.concatenate(train_true_mixup)
        train_pred_mixup = np.concatenate(train_pred_mixup)
        acc_score_mixup = metrics.accuracy_score(train_true_mixup, train_pred_mixup)
        balanced_acc_score_mixup = metrics.balanced_accuracy_score(train_true, train_pred)
    else:
        acc_score_mixup, balanced_acc_score_mixup = -1, -1 #undefined

    reset_tensor([data, label, logits, attn, preds, DATA[0], DATA[1]])
    try:
        reset_tensor([DATA[2]])
    except:
        pass
    return np.around(acc_score, decimals=4), np.around(balanced_acc_score, decimals=4), np.around(acc_score_mixup, decimals=4), \
           np.around(balanced_acc_score_mixup, decimals=4), np.around(loss, decimals=4)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def translate_pointcloud_torch(pointcloud, dataset='modelnet40'):
    # pointcloud = [BS,_pts,3]
    # print("pointcloud aug torch size: ", pointcloud.size())
    if dataset=='modelnet40':
        bs,n_pts,n_feat = pointcloud.size()
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=(bs,1,n_feat))
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=(bs,1,n_feat))
        xyz1, xyz2 = torch.from_numpy(xyz1).type(torch.float).cuda(), torch.from_numpy(xyz2).type(torch.float).cuda()
        translated_pointcloud = torch.add(torch.mul(pointcloud, xyz1), xyz2)
    elif dataset=='modelnet10':
        pointcloud = pointcloud.cpu().numpy()
        translated_pointcloud = provider.random_point_dropout(pointcloud)
        translated_pointcloud[:, :, 0:3] = provider.random_scale_point_cloud(translated_pointcloud[:, :, 0:3])
        translated_pointcloud[:, :, 0:3] = provider.shift_point_cloud(translated_pointcloud[:, :, 0:3])
        translated_pointcloud = torch.from_numpy(translated_pointcloud).cuda()
    else:
        raise Exception("Undefined dataset!")
    return translated_pointcloud

def save_model(cfg,epoch,model,opt,scheduler,loss_train,loss_test,is_mix_start, never_start_mixing, suffix_name=''):
    model_path = f'{cfg.EXP.WORKING_DIR}/{cfg.EXP.NAME}/models/model{suffix_name}.t7'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss_train,
                'test_loss': loss_test,
                'is_mix_start': is_mix_start,
                'never_start_mixing': never_start_mixing}, model_path)