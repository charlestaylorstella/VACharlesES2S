alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

model=$1
mark=${model}_$2

gpuid=0
test_txt=data/oridata/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test
pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_txt} -output pred_${mark}.txt -replace_unk -verbose > loggene_${mark}.txt 2> errgene_${mark}.txt
#python translate.py -gpu ${gpuid} -model ${model} -src data/src-test.txt -output pred_${mark}.txt -replace_unk -verbose > loggene_${mark}.txt 2> errgene_${mark}.txt
