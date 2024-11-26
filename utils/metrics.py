import argparse
from data_loader.loader import Online_Dataset
import torch
import numpy as np
import tqdm
from fastdtw import fastdtw
from models.eval_model import *

def fast_norm_len_dtw(test_loader):
    """start test iterations"""
    euclidean = lambda x, y: np.sqrt(sum((x - y) ** 2))
    fast_norm_dtw_len, total_num = 0, 0

    for data in tqdm.tqdm(test_loader):
        preds, preds_len, character_id, writer_id, coords_gts, len_gts = data['coords'], \
            data['coords_len'].long(), \
            data['character_id'].long(), \
            data['writer_id'].long(), \
            data['coords_gt'], \
            data['len_gt'].long()
        for i, pred in enumerate(preds):
            pred_len,  gt_len= preds_len[i], len_gts[i]
            pred_valid, gt_valid = pred[:pred_len], coords_gts[i][:gt_len]

            # Convert relative coordinates into absolute coordinates
            seq_1 = torch.cumsum(gt_valid[:, :2], dim=0)
            seq_2 = torch.cumsum(pred_valid[:, :2], dim=0)
            
            # DTW between paired real and fake online characters
            fast_d, _ = fastdtw(seq_1, seq_2, dist= euclidean)
            fast_norm_dtw_len += (fast_d/gt_len)
        total_num += len(preds)
    avg_fast_norm_dtw_len = fast_norm_dtw_len/total_num
    return avg_fast_norm_dtw_len

def get_style_score(test_loader,pretrained_model):
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    print('calculate the acc for the testset')
    print('loading testset...')

    model = offline_style(num_class=test_loader.dataset.num_class).cuda()

    if len(pretrained_model) > 0:
        model.load_state_dict(torch.load(pretrained_model))
        print('load pretrained model from {}'.format(pretrained_model))

    model.eval()
    with torch.no_grad():
        for data, labels in tqdm.tqdm(test_loader):
            data, labels = data.cuda(), labels.cuda()
            test_preds = model(data)
            prediction = torch.argmax(test_preds, 1)
            correct += (prediction == labels).sum().float()
            total += len(labels)
        acc_str = (correct/total).cpu().numpy()
    return acc_str

def get_content_score(test_loader,pretrained_model):
    """ set model, criterion and optimizer"""
    Net = Character_Net(nclass=len(test_loader.dataset.char_dict)).cuda().eval()
    if len(pretrained_model) > 0:
        Net.load_state_dict(torch.load(pretrained_model))
        print('load pretrained model from {}'.format(pretrained_model))

    """start test iterations"""

    Net.eval()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for data in tqdm.tqdm(test_loader):
        coords, coords_len, character_id, writer_id = data['coords'].cuda(), \
                                                      data['coords_len'].cuda(), \
                                                      data['character_id'].long().cuda(), \
                                                      data['writer_id'].long().cuda()

        with torch.no_grad():   
            coords = torch.transpose(coords, 1, 2)
            logits = Net(coords, coords_len)
            prediction = torch.argmax(logits, 1)
            correct += (prediction == character_id.long()).sum().float()
            total += len(coords)
    acc = (correct/total).cpu().numpy()
    return acc