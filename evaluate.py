import argparse
from data_loader.loader import Online_Dataset
import torch
import numpy as np
import tqdm
from fastdtw import fastdtw

def main(opt):
    """ set dataloader"""
    test_dataset = Online_Dataset(opt.data_path)
    print('loading generated samples, the total amount of samples is', len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=8)
    
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
    print(f"the avg fast_norm_len_dtw is {avg_fast_norm_dtw_len}")

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='Generated/Chinese',
                        help='dataset path for evaluating the DTW distance between real and fake characters')
    opt =  parser.parse_args() 
    main(opt)