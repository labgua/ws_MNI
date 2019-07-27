

LOGGER=$1 #es5_p2_bench.dat
PROG=$2

N_INIT=1024
N_MAX=32768
NTH=(8 16 32)


for (( i = N_INIT; i < N_MAX; i*=2 )); do
	for j in "${NTH[@]}"; do
		echo "echo $i $j | ./$PROG >> $LOGGER" >> $LOGGER
		echo $i $j | ./$PROG >> $LOGGER
	done
done

