alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

model=$1
mark=${model}_$2

gpuid=-1
#test_query=data/oridata/DialogDataFromBenben/douban_data_seg_done/q_train_r2000

mkdir -p predict_result
mkdir -p log_for_predict

function test_2k() {
test_query=data/oridata2/DialogDataFromBenben/douban_data_seg_done/q_test
text_response=data/oridata2/DialogDataFromBenben/douban_data_seg_done/r_test

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_${mark}.txt 2> log_for_predict/errgene_${mark}.txt
sh bleu.sh predict_result/pred_${mark}.txt ${text_response}
}

function test_t5w() {
test_query=data/oridata2/DialogDataFromBenben/douban_data_seg_done_v2/q_test_5w
text_response=data/oridata2/DialogDataFromBenben/douban_data_seg_done_v2/r_test_5w

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_t5w_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_t5w_${mark}.txt 2> log_for_predict/errgene_t5w_${mark}.txt
sh bleu.sh predict_result/pred_t5w_${mark}.txt ${text_response}
}

test_2k
#test_t5w
