
LOGGER=$1 #es5_p1_bench.dat
PROG=$2 #es5_p1

N_INIT=100000
N_MAX=3200000
NTH=(8 16 32)


for (( i = N_INIT; i < N_MAX; i*=2 )); do
	for j in "${NTH[@]}"; do
		echo $i $j | ./$PROG >> $LOGGER
	done
done
