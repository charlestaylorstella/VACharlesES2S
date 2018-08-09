input=errvae_ae_savez_ed1h_hd1h_last2epoch
lines=120955
#lines=11200

inputp1=${input}.part1
inputp2=${input}.part2
inputeuc=${input}.euc

start1=0
end1=${lines}
start2=$((${lines} + 2))
end2=$((${start2} + ${lines} - 1))

awk -v s=${start1} -v e=${end1} 's<=NR && NR<=e' ${input} > ${inputp1} &
awk -v s=${start2} -v e=${end2} 's<=NR && NR<=e' ${input} > ${inputp2} &
wait

alias TstatisticsForEmb='python2.7 /homes/tianzhiliang/.tzltools/statisticsForEmb/statisticsForEmb.py'
#paste ${inputp1} ${inputp2} | awk -F "\t" '{print $2 "\t" $4}' > ${inputeuc}
paste ${inputp1} ${inputp2} | awk -F "\t" '{print $2 "\t" $4}' | TstatisticsForEmb pairData > ${inputeuc}
