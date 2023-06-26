import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from models.loss import SupConLoss, get_pen_loss
from models.model import SDT_Generator
from utils.logger import set_log
from data_loader.loader import ScriptDataset
import torch
from trainer.trainer import Trainer

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)
    """ set dataset"""
    train_dataset = ScriptDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TRAIN.ISTRAIN, cfg.MODEL.NUM_IMGS)
    print('number of training images: ', len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               shuffle=True,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS)
    test_dataset = ScriptDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    char_dict = test_dataset.char_dict
    """ build model, criterion and optimizer"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    ### load checkpoint
    if len(opt.pretrained_model) > 0:
        model.load_state_dict(torch.load(opt.pretrained_model))
        print('load pretrained model from {}'.format(opt.pretrained_model))
    elif len(opt.content_pretrained) > 0:
        model_dict = load_specific_dict(model.content_encoder, opt.content_pretrained, "feature_ext")
        model.content_encoder.load_state_dict(model_dict)
        print('load content pretrained model from {}'.format(opt.content_pretrained))
    else:
        pass
    criterion = dict(NCE=SupConLoss(contrast_mode='all'), PEN=get_pen_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    """start training iterations"""
    trainer = Trainer(model, criterion, optimizer, train_loader, logs, char_dict, test_loader)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='',
                        dest='pretrained_model', required=False, help='continue to train model')
    parser.add_argument('--content_pretrained', default='model_zoo/position_layer2_dim512_iter138k_test_acc0.9443.pth',
                        dest='content_pretrained', required=False, help='continue to train content encoder')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    opt = parser.parse_args()
    main(opt)