alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'

model=$1
mark=${model}_$2

gpuid=-1
data_mark="" # _r3w _r1w _r4k _h1w (empty)
test_query=data/oridata2/ConvSeq2Seq2017_WMT14_English2German_Data/English2German/newstest2014.en
test_response=data/oridata2/ConvSeq2Seq2017_WMT14_English2German_Data/English2German/newstest2014.de

mkdir -p predict_result
mkdir -p log_for_predict

function test() {
test_query_real=${test_query}${data_mark}
test_response_real=${test_response}${data_mark}

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query_real} -output predict_result/pred${data_mark}_${mark}.txt -replace_unk -verbose > log_for_predict/loggene${data_mark}_${mark}.txt 2> log_for_predict/errgene${data_mark}_${mark}.txt
sh bleu.sh predict_result/pred${data_mark}_${mark}.txt ${test_response_real}
}

function test_h1w() {
test_query=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test_h1w
test_response=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.test_h1w
pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_h1w_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_h1w_${mark}.txt 2> log_for_predict/errgene_h1w_${mark}.txt
sh bleu.sh predict_result/pred_h1w_${mark}.txt ${test_response}
}

function test_r1w() {
test_query=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test_r1w
test_response=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.test_r1w

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_${mark}.txt 2> log_for_predict/errgene_${mark}.txt
sh bleu.sh predict_result/pred_${mark}.txt ${test_response}
}

function test_r4k() {
test_query=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test_r4k
test_response=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.test_r4k

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_r4k_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_r4k_${mark}.txt 2> log_for_predict/errgene_r4k_${mark}.txt
sh bleu.sh predict_result/pred_r4k_${mark}.txt ${test_response}
}

function test_r3w() {
test_query=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.en.sp.test_r3w
test_response=data/oridata2/wmt_ende_sp_preprocess/europarl-v7.de-en.de.sp.test_r3w

pythont translate.py -gpu ${gpuid} -model ${model} -src ${test_query} -output predict_result/pred_r3w_${mark}.txt -replace_unk -verbose > log_for_predict/loggene_r3w_${mark}.txt 2> log_for_predict/errgene_r3w_${mark}.txt
sh bleu.sh predict_result/pred_r3w_${mark}.txt ${test_response}
}

test
