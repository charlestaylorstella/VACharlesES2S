alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

model=$1
mark=${model}_$2

gpuid=-1
#test_query=data/oridata/DialogDataFromBenben/douban_data_seg_done/q_train_r2000
#text_response=data/oridata/DialogDataFromBenben/douban_data_seg_done/r_train_r2000

test_query=data/oridata2/DialogDataFromBenben/douban_data_seg_done_v2/q_train_r2k
text_response=data/oridata2/DialogDataFromBenben/douban_data_seg_done_v2/r_train_r2k

mkdir -p predict_result
mkdir -p log_for_predict
pythont translate.py -model ${model} -src ${test_query} -output predict_result/predtrain_${mark}.txt -replace_unk -verbose > log_for_predict/loggenetrain_${mark}.txt 2> log_for_predict/errgenetrain_${mark}.txt
#pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/predtrain_${mark}.txt -replace_unk -verbose > log_for_predict/loggenetrain_${mark}.txt 2> log_for_predict/errgenetrain_${mark}.txt
#pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predtrain_${mark}.txt -replace_unk -verbose > loggenetrain_${mark}.txt 2> errgenetrain_${mark}.txt

#python translate.py -gpu ${gpuid} -model ${model} -src data/src-test.txt -output pred_${mark}.txt -replace_unk -verbose > loggene_${mark}.txt 2> errgene_${mark}.txt

sh bleu.sh predict_result/predtrain_${mark}.txt ${text_response}
