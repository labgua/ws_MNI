
N_INIT=100

DOUBLING=5

ATTEMPT=3

LOGGER=benchmark_es4_p1.dat

PROG=es4_p1

echo "ES4 p1 bench" >> $LOGGER

for (( i = 0; i < DOUBLING; i++ )); do

	N=$N_INIT

	echo "=====> TEST-$i <====="
	echo -e "\n\n ====================================> TEST-$i <====================================" >> $LOGGER	

	for (( j = 0; j < ATTEMPT; i++ )); do

		echo $N $N_THREAD | ./es4_p1 >> $LOGGER
		
		echo "========================================================================================" >> $LOGGER

		N=$((N*2))
	done
done