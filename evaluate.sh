CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/Chinese --metric DTW
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/Chinese --metric Content_score --pretrained_content_model /home/HDD/qiangya/SDT/SDT/eval/content_model/chinese_iter30k_acc95.pth
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated_img/Chinese --metric Style_score --pretrained_style_model /home/HDD/qiangya/SDT/SDT/eval/style_model/chinese_iter60k_acc999.pth
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/Japanese --metric DTW
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/Japanese --metric Content_score --pretrained_content_model /home/HDD/qiangya/SDT/SDT/eval/content_model/japan_merge_multieps_iter36k_acc93.pth
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated_img/Japanese --metric Style_score --pretrained_style_model /home/HDD/qiangya/SDT/SDT/eval/style_model/japan_style_iter16k_acc997.pth
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/English --metric DTW
CUDA_VISIBLE_DEVICES=5 python evaluate.py --data_path /home/HDD/qiangya/SDT/SDT/Generated/English --metric Content_score --pretrained_content_model /home/HDD/qiangya/SDT/SDT/eval/content_model/eng_eps2_iter3k_acc80.pth
