
LOGGER=$1 #es6_bench.dat
PROG=$2 #es6

N_INIT=100000
N_MAX=3200000
NTH=(8 16 32)


for (( i = N_INIT; i < N_MAX; i*=2 )); do
	for j in "${NTH[@]}"; do
		echo "./$PROG $i $j 0 >> $LOGGER" >> $LOGGER
		./$PROG $i $j 0 >> $LOGGER
	done
done
