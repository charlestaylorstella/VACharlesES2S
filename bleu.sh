
GENERATED=$1
GROUND_TRUTH=$2
#GROUND_TRUTH=a
#GROUND_TRUTH=data/oridata/DialogDataFromBenben/douban_data_seg_done/r_test
#GROUND_TRUTH=data/oridata/DialogDataFromBenben/douban_data_seg_done_v2/r_test_5w
#GROUND_TRUTH=data/oridata/DialogDataFromBenben/douban_data_seg_done/r_train_r2000
PYTHON=python2.7

${PYTHON} bleu.py ${GENERATED} ${GROUND_TRUTH} > ${GENERATED}.bleu
cat ${GENERATED}.bleu
