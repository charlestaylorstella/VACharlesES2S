alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

mark=$1
gpuid=1
pythont train.py -data data/douban_bb_q2r_v5k -save_model model_douben_bb_q2r5k -gpuid ${gpuid} > log${mark} 2>err${mark}
#CUDA_VISIBLE_DEVICES=${gpuid} pythont train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}
#python train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}
