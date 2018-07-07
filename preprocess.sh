#src_train=data/src-train.txt
#tgt_train=data/tgt-train.txt
#src_val=data/src-val.txt
#tgt_val=data/tgt-val.txt

#src_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.train
#tgt_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.train
#src_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test
#tgt_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.test
#src_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.val
#tgt_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.val

#src_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p1.train
#tgt_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p1.train
#src_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p1.test
#tgt_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p1.test
#src_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p1.val
#tgt_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p1.val

src_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p01.train
tgt_train=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p01.train
src_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p01.test
tgt_test=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p01.test
src_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.0p01.val
tgt_val=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.0p01.val

output_name=data/wmt_europarlv7_en2de_v5k_0p01
src_vocab_size=5000
tgt_vocab_size=5000

alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'
pythont preprocess.py -train_src ${src_train} -train_tgt ${tgt_train} -valid_src ${src_val} -valid_tgt ${tgt_val} -test_src ${src_test} -test_tgt ${tgt_test} -save_data ${output_name} -src_vocab_size ${src_vocab_size} -tgt_vocab_size ${tgt_vocab_size}
#pythont preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000
