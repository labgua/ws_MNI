# bash

export MNIDEBUG 
MNIDEBUG=0

LOGGER=benchmark.dat
SEQPROG=es1_p1
MPIPROG=somma


### CHECKING DEBUGGING in console
if [[ $MNIDEBUG == 1 ]]; then
	echo "DEBUGGING is ON"
else 
	echo "DEBUGGING is OFF"
fi


date >> $LOGGER

echo "Sequencial Benchmark"
echo "Sequencial Benchmark" >> $LOGGER

./$SEQPROG 100000 >> $LOGGER
echo "----------------------" >> $LOGGER

./$SEQPROG 200000 >> $LOGGER
echo "----------------------" >> $LOGGER

./$SEQPROG 400000 >> $LOGGER
echo "----------------------" >> $LOGGER


echo "======================" >> $LOGGER


echo "Parallel Benchmark"
echo "Parallel Benchmark" >> $LOGGER

echo 100000 | mpirun -x MNIDEBUG -np 2 $MPIPROG >> $LOGGER
echo "----------------------" >> $LOGGER

echo 200000 | mpirun -x MNIDEBUG -np 4 $MPIPROG >> $LOGGER
echo "----------------------" >> $LOGGER

echo 400000 | mpirun -x MNIDEBUG -np 8 $MPIPROG >> $LOGGER
echo "----------------------" >> $LOGGER



echo "======================" >> $LOGGER