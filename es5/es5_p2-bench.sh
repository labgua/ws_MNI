
LOGGER=es5_p2_bench.dat

N_INIT=1024
N_MAX=32768
NTH=(8 16 32)


for (( i = N_INIT; i < N_MAX; i*=2 )); do
	for j in "${NTH[@]}"; do
		echo $i $j | ./es5_p2 >> $LOGGER
	done
done

