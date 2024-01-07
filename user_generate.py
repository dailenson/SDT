import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import UserDataset
import pickle
from models.model import SDT_Generator
import tqdm
from utils.util import writeCache, dxdynp_to_list, coords_render
import lmdb

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """setup data_loader instances"""
    test_dataset = UserDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, opt.style_path)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)

    os.makedirs(os.path.join(opt.save_dir), exist_ok=True)

    """build model architecture"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    if len(opt.pretrained_model) > 0:
        model_weight = torch.load(opt.pretrained_model)
        model.load_state_dict(model_weight)
        print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        raise IOError('input the correct checkpoint path')
    model.eval()

    """setup the dataloader"""
    batch_samples = len(test_loader)
    data_iter = iter(test_loader)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):

            data = next(data_iter)
            # prepare input
            img_list, char_img, char = data['img_list'].cuda(), \
                data['char_img'].cuda(), data['char'] 
            preds = model.inference(img_list, char_img, 120)
            bs = char_img.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
            preds = preds.detach().cpu().numpy()

            for i, pred in enumerate(preds):
                """Render the character images by connecting the coordinates"""
                sk_pil = coords_render(preds[i], split=True, width=256, height=256, thickness=8, board=1)
                
                save_path = os.path.join(opt.save_dir, char[i] +'.png')
                try:
                    sk_pil.save(save_path)
                except:
                    print('error. %s, %s' % (save_path, char[i]))


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_USER.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/Chinese_User', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True, help='continue train model')
    parser.add_argument('--style_path', dest='style_path', default='style_samples', help='dir of style samples')
    opt = parser.parse_args()
    main(opt)