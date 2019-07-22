
all: es1/es1_p1 es1/somma es1/es1_p3 es2/matvet_seq es2/mvet es2/oomvet es2/es2_p2 es3/es3_p1 train/cuda/esercizio01 train/cuda/esercizio02


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

es2/oomvet: es2/oomvet.c
	mpicc -o es2/oomvet es2/oomvet.c -lm

es2/es2_p2: es2/es2_p2.c utility/utility.o
	mpicc -o es2/es2_p2 es2/es2_p2.c utility/utility.o -lm


es3/es3_p1: es3/es3_p1.c utility/utility.o utility/vmatrix.o
	mpicc -o es3/es3_p1 es3/es3_p1.c utility/utility.o utility/vmatrix.o -lm



### TRAINING
train/cuda/esercizio01: train/cuda/esercizio01.cu
	nvcc train/cuda/esercizio01.cu -o train/cuda/esercizio01

train/cuda/esercizio02: train/cuda/esercizio02.cu
	nvcc train/cuda/esercizio02.cu -o train/cuda/esercizio02