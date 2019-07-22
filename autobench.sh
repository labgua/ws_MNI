# bash

export MNIDEBUG 
MNIDEBUG=0

LOGGER=$1
##SEQPROG=matvet_seq
MPIPROG=$2

N_INIT=$3

DOUBLING=$4

ATTEMPT=$5


echo "AutoBench, for MNI"

if [ $# = 0 ] || [ $# != 5 ]; 
then
	read -p "LOGGER           : " LOGGER
	read -p "MPIPROG          : " MPIPROG
	read -p "INITIAL N VALUE  : " N_INIT
	read -p "# DOUBLING N     : " DOUBLING
	read -p "# ATTEMPT TEST   : " ATTEMPT
fi

LOGGER=$(date +%s)_$LOGGER

### CHECKING DEBUGGING in console
if [[ $MNIDEBUG == 1 ]]; then
	echo "DEBUGGING is ON"
else 
	echo "DEBUGGING is OFF"
fi

read -p "Titolo esperimento: " TITLE

echo $TITLE >> $LOGGER
echo "MPIPROG = $MPIPROG" >> $LOGGER
echo "N_INIT = $N_INIT" >> $LOGGER
echo "DOUBLING = $DOUBLING" >> $LOGGER
echo "ATTEMPT = $ATTEMPT" >> $LOGGER
date >> $LOGGER


for (( i = 0; i < DOUBLING; i++ )); do

	N=$N_INIT

	echo "=====> TEST-$i <====="
	echo -e "\n\n ====================================> TEST-$i <====================================" >> $LOGGER	
	for (( j = 1; j <= ATTEMPT; j++ )); do
		echo "--- TEST-$i N=$N"
		echo "--- TEST-$i N=$N" >> $LOGGER


		################### ESECUZIONE DEL TEST #################

		echo "Sequencial Benchmark"
		echo "Sequencial Benchmark" >> $LOGGER

		echo "echo $N | mpirun -x MNIDEBUG -np 1 $MPIPROG >> $LOGGER"
		echo $N | mpirun -x MNIDEBUG -np 1 $MPIPROG >> $LOGGER

		echo "----------------------" >> $LOGGER



		echo "======================" >> $LOGGER


		echo "Parallel Benchmark"
		echo "Parallel Benchmark" >> $LOGGER

		echo "echo $N | mpirun -x MNIDEBUG -np 2 $MPIPROG >> $LOGGER"
		echo $N | mpirun -x MNIDEBUG -np 2 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER

		echo "echo $N | mpirun -x MNIDEBUG -np 4 $MPIPROG >> $LOGGER"
		echo $N | mpirun -x MNIDEBUG -np 4 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER

		echo "echo $N | mpirun -x MNIDEBUG -np 8 $MPIPROG >> $LOGGER"
		echo $N | mpirun -x MNIDEBUG -np 8 $MPIPROG >> $LOGGER
		echo "----------------------" >> $LOGGER



		echo "========================================================================================" >> $LOGGER
		#########################################################


		N=$((N*2))
	done
done
