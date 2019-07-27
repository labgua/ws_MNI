

N_INIT=$1

DOUBLING=$2

ATTEMPT=$3

N_THREAD=32
LOGGER=benchmark.dat

for (( i = 0; i < DOUBLING; i++ )); do

	N=$N_INIT

	echo "=====> TEST-$i <====="
	echo -e "\n\n ====================================> TEST-$i <====================================" >> $LOGGER	

	for (( j = 0; j < ATTEMPT; i++ )); do

		echo $N $N_THREAD | ./tempi >> $LOGGER
		
		echo "========================================================================================" >> $LOGGER

		N=$((N*2))
	done
done