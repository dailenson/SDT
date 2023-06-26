import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import ScriptDataset
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
    writer_dict = test_dataset.writer_dict

    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

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

    """calculate the total batches of generated samples"""
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        batch_samples = int(opt.sample_size)*len(writer_dict)/cfg.TRAIN.IMS_PER_BATCH

    batch_num, num_count= 0, 0
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            batch_num += 1
            if batch_num > batch_samples:
                break
            else:
                # prepare input
                coords, coords_len, character_id, writer_id, img_list, char_img = data['coords'].cuda(), \
                    data['coords_len'].cuda(), \
                    data['character_id'].long().cuda(), \
                    data['writer_id'].long().cuda(), \
                    data['img_list'].cuda(), \
                    data['char_img'].cuda()
                preds = model.inference(img_list, char_img, 120)
                bs = character_id.shape[0]
                SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
                preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
                preds = preds.detach().cpu().numpy()

                test_cache = {}
                coords = coords.detach().cpu().numpy()
                if opt.store_type == 'online':
                    for i, pred in enumerate(preds):
                        pred, _ = dxdynp_to_list(preds[i])
                        coord, _ = dxdynp_to_list(coords[i])
                        data = {'coordinates': pred, 'writer_id': writer_id[i].item(),
                                'character_id': character_id[i].item(), 'coords_gt':coord}
                        data_byte = pickle.dumps(data)
                        data_id = str(num_count).encode('utf-8')
                        test_cache[data_id] = data_byte
                        num_count += 1
                    test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
                    writeCache(test_env, test_cache)
                elif opt.store_type == 'img':
                    for i, pred in enumerate(preds):
                        """intends to blur the boundaries of each sample to fit the actual using situations,
                            as suggested in 'Deep imitator: Handwriting calligraphy imitation via deep attention networks'"""
                        sk_pil = coords_render(preds[i], split=True, width=48, height=48, thickness=1, board=0)
                        sk_pil = sk_pil.resize((113, 113))

                        character = char_dict[character_id[i].item()]
                        save_path = os.path.join(opt.save_dir, 'test',
                                        str(writer_id[i].item()) + '_' + character+'.jpg')
                        try:
                            sk_pil.save(save_path)
                        except:
                            print('error. %s, %s, %s' % (save_path, str(writer_id[i].item()), character))
                else:
                    raise NotImplementedError('only support online or img format')

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/Chinese', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True, help='continue train model')
    parser.add_argument('--store_type', dest='store_type', default='online', help='online or img')
    parser.add_argument('--sample_size', dest='sample_size', default='500', help='randomly generate a certain number of characters for each writer')
    opt = parser.parse_args()
    main(opt)