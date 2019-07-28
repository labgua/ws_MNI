
PARAMS=-arch=sm_50

all:es1/es1_p1 es1/somma es1/es1_p3 es2/matvet_seq es2/mvet es2/es2_p2 es3/es3_p1 \
	train/cuda/info train/cuda/esercizio01 train/cuda/esercizio02 train/cuda/esercizio03 train/cuda/esercizio04 \
	es4/es4_p1 es4/es4_p2 es5/tempi es5/es5_p2 es6/es6

utility/utility.o: utility/utility.c utility/utility.h
#	gcc -c utility/utility.c -o utility/utility.o
	mpicc -c utility/utility.c -o utility/utility.o

utility/vmatrix.o: utility/vmatrix.c utility/vmatrix.h
	gcc -c utility/vmatrix.c -o utility/vmatrix.o


es1/es1_p1: es1/es1_p1.c utility/utility.o
#	gcc -o es1/es1_p1 es1/es1_p1.c utility/utility.o
	mpicc -o es1/es1_p1 es1/es1_p1.c utility/utility.o

es1/somma: es1/somma.c utility/utility.o
	mpicc -o es1/somma es1/somma.c utility/utility.o

es1/es1_p3: es1/es1_p3.c utility/utility.o
	mpicc -o es1/es1_p3 es1/es1_p3.c utility/utility.o

es2/matvet_seq: es2/matvet_seq.c utility/utility.o
#	gcc -o es2/matvet_seq es2/matvet_seq.c utility/utility.o
	mpicc -o es2/matvet_seq es2/matvet_seq.c utility/utility.o

es2/mvet: es2/mvet.c utility/utility.o
	mpicc -o es2/mvet es2/mvet.c utility/utility.o -lm

es2/es2_p2: es2/es2_p2.c utility/utility.o
	mpicc -o es2/es2_p2 es2/es2_p2.c utility/utility.o -lm


es3/es3_p1: es3/es3_p1.c utility/utility.o utility/vmatrix.o
	mpicc -o es3/es3_p1 es3/es3_p1.c utility/utility.o utility/vmatrix.o -lm

es4/es4_p1: es4/es4_p1.cu
	nvcc $(PARAMS) es4/es4_p1.cu -o es4/es4_p1

es4/es4_p2: es4/es4_p2.cu
	nvcc $(PARAMS) es4/es4_p2.cu -o es4/es4_p2

es5/tempi: es5/tempi.cu
	nvcc $(PARAMS) es5/tempi.cu -o es5/tempi

es5/es5_p2: es5/es5_p2.cu
	nvcc $(PARAMS) es5/es5_p2.cu -o es5/es5_p2

es6/es6: es6/es6.cu
	nvcc $(PARAMS) es6/es6.cu -o es6/es6



### TRAINING
train/cuda/info: train/cuda/info.cu
	nvcc $(PARAMS) train/cuda/info.cu -o train/cuda/info

train/cuda/esercizio01: train/cuda/esercizio01.cu
	nvcc $(PARAMS) train/cuda/esercizio01.cu -o train/cuda/esercizio01

train/cuda/esercizio02: train/cuda/esercizio02.cu
	nvcc $(PARAMS) train/cuda/esercizio02.cu -o train/cuda/esercizio02

train/cuda/esercizio03: train/cuda/esercizio03.cu
	nvcc $(PARAMS) train/cuda/esercizio03.cu -o train/cuda/esercizio03

train/cuda/esercizio04: train/cuda/esercizio04.cu
	nvcc $(PARAMS) train/cuda/esercizio04.cu -o train/cuda/esercizio04
