alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

mark=$1
gpuid=1
pythont train.py -data data/wmt_europarlv7_en2de_v5k_0p01 -save_model model_wmt_eur7_en2de5k_0p01 -gpuid ${gpuid} > log${mark} 2>err${mark}
#CUDA_VISIBLE_DEVICES=${gpuid} pythont train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}
#python train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}
