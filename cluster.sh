input=errvae_ae_savez_ed1h_hd1h_last2epoch
cluster_num=1000

output=${input}.cluster_${cluster_num}

python cluster.py ${input} ${cluster_num=1000} > ${output}


