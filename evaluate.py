import argparse
from data_loader.loader import Online_Dataset, test_offline_Style_Dataset, Online_Gen_Dataset
import torch
import numpy as np
import tqdm
from fastdtw import fastdtw
from utils.metrics import *
def main(opt):
    if opt.metric == 'DTW':
        """ set dataloader"""
        test_dataset = Online_Dataset(opt.data_path)
        print('loading generated samples, the total amount of samples is', len(test_dataset))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt.batchsize,
                                                shuffle=True,
                                                sampler=None,
                                                drop_last=False,
                                                collate_fn=test_dataset.collate_fn_,
                                                num_workers=8)
        DTW = fast_norm_len_dtw(test_loader)
        print(f"the avg fast_norm_len_dtw is {DTW}")
    if opt.metric == 'Style_score':
        test_dataset = test_offline_Style_Dataset(opt.data_path,False)
        print('num testing images:', len(test_dataset))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=opt.batchsize,
                                    shuffle=True,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=8)
        style_score = get_style_score(test_loader, opt.pretrained_style_model)
        print(f"the style_score is {style_score}")
    if opt.metric == 'Content_score':
        test_dataset = Online_Gen_Dataset(opt.data_path, False)
        print('num test images: ', len(test_dataset))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt.batchsize,
                                                shuffle=True,
                                                sampler=None,
                                                drop_last=False,
                                                collate_fn=test_dataset.collate_fn_,
                                                num_workers=8)
        content_score = get_content_score(test_loader, opt.pretrained_content_model)
        print(f"the content_score is {content_score}")
if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, dest='data_path', default='Generated/Chinese',
                        help='dataset path for evaluating the metrics')
    parser.add_argument('--metric', type=str, default='Content_score', help='the metric to evaluate the generated data, DTW, Style_score or Content_score')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--pretrained_style_model', type=str, default='/home/HDD/qiangya/SDT/SDT/eval/style_model/chinese_iter60k_acc999.pth', help='pre-trained model for calculating Style Score')
    parser.add_argument('--pretrained_content_model', type=str, default='/home/HDD/qiangya/SDT/SDT/eval/content_model/chinese_iter30k_acc95.pth',  help='pre-trained model for calculating Content Score')
    opt =  parser.parse_args() 
    main(opt)