alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

mark=$1
gpuid=2
pythont train.py -data data/db_bb2_q2r_v10w3w -save_model model_dbbb2_v10w3w -gpuid ${gpuid} > log${mark} 2>err${mark}
#CUDA_VISIBLE_DEVICES=${gpuid} pythont train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}
#python train.py -data data/demo -save_model demo-model -gpuid ${gpuid} > log${mark} 2>err${mark}