model=$1
mark=${model}_$2

gpuid=0
python translate.py -gpu ${gpuid} -model ${model} -src data/src-test.txt -output pred_${mark}.txt -replace_unk -verbose > loggene_${mark}.txt 2> errgene_${mark}.txt
