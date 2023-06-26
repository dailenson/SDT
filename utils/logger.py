
import time
import os

""" prepare logdir for tensorboard and logging output"""
def set_log(output_dir, cfg_file, log_name):
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_name = os.path.basename(cfg_file).split('.')[0]
    log_dir = os.path.join(output_dir, base_name, log_name + "-" + t)
    logs = {}
    for temp in ['tboard', 'model', 'sample']:
        temp_dir = os.path.join(log_dir, temp)
        os.makedirs(temp_dir, exist_ok=True)
        logs[temp] = temp_dir
    return logs