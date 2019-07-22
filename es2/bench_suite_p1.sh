# bash

export MNIDEBUG 
MNIDEBUG=0

LOGGER=es2_p1_benchmark.dat
SEQPROG=matvet_seq
MPIPROG=mvet

N_INIT=1000

### CHECKING DEBUGGING in console
if [[ $MNIDEBUG == 1 ]]; then
	echo "DEBUGGING is ON"
else 
	echo "DEBUGGING is OFF"
fi


echo "ES2 p1 - prodotto matrice vettore benchmark" >> $LOGGER
echo "N = $N" >> $LOGGER
date >> $LOGGER


for (( i = 0; i < 3; i++ )); do

	N=$N_INIT

	echo "=====> TEST-$i <====="
	echo "=====> TEST-$i <=====" >> $LOGGER	
	for (( j = 1; j <= 3; j++ )); do
		echo "--- TEST-$i N=$N"
		echo "--- TEST-$i N=$N" >> $LOGGER


		################### ESECUZIONE DEL TEST #################

		echo "Sequencial Benchmark"
		echo "Sequencial Benchmark" >> $LOGGER

		###./$SEQPROG 4000 >> $LOGGER
		echo $N | mpirun -x MNIDEBUG -np 1 $MPIPROG >> $LOGGER

		echo "----------------------" >> $LOGGER



		echo "======================" >> $LOGGER


		echo "Parallel Benchmark"
		echo "Parallel Benchmark" >> $LOGGER

		echo $N | mpirun -x MNIDEBUG -np 2 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER

		echo $N | mpirun -x MNIDEBUG -np 4 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER

		echo $N | mpirun -x MNIDEBUG -np 8 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER



		echo "========================================================================================" >> $LOGGER
		#########################################################


		N=$((N*2))
	done
done
