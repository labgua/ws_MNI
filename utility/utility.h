/*
 * utility.h
 *
 *  Created on: 15 lug 2019
 *      Author: sergio
 */

#ifndef UTILITY_UTILITY_H_
#define UTILITY_UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_NUM 100
#define MAXLEN_MSG_BUFFER 5000

char MSG_BUFFER[MAXLEN_MSG_BUFFER];

/**
 *	Meccaniscmo di debugging
 *	impostare la variabile con export prima di lanciare il programma
 *  Se DEBUG_VAR == 1 --> esegue il corpo (es. stampa)
 */
#define DEBUG_VAR "MNIDEBUG"
#define DD if( getenv(DEBUG_VAR) && atoi(getenv(DEBUG_VAR)) == 1 )

/**
 *	Sentinella per debugging
 **/
#define HERE(file)	do{ MPI_Barrier(MPI_COMM_WORLD); int ____me____; MPI_Comm_rank (MPI_COMM_WORLD, &____me____); fprintf(file, "I'm HERE p:%d\n", ____me____ ); } while(0)

/**
 *	Sentinella per degugging, v2
 *	per compatibilita del codice vecchio è stata rionominata
 *	in W(ORLD)HERE: cioè HERE in COMMUNICATOR_WORLD
 *	... ma è anche un gioco di parole
 *	--> scrivere WHERE e sentirsi rispondere (in ordine) I'm HERE
 */
#define WHERE		ordered_printf(MPI_COMM_WORLD, "I'm HERE")

/**
 *	Stampa ordinata
 *	Esegui la stampa ordinata di un messaggio formattato
 *	come l'uso di una printf.
 *	Per ogni processore nel CommonWorld avviene una attesa
 *	pari a rank*500us seguita dalla stampa
 */
void ordered_printf(MPI_Comm comm, const char *format, ...);


/**
 * Stampa su file di un vettore di double
 * f: file su cui scrivere
 * label: stringa di debug
 * v: vettore da stampare
 * n: lunghezza del vettore da stampare
 */
void fprint_row(FILE* f, char* label, double* v, int n);

void fprint_int_row(FILE* f, char* label, int* v, int n);


char* tostr_row(double* v, int n);


/**
 * Genera un vettore di double casuali
 * size: dimensione del vettore
 * seed: seme di partenza per la generazione casuale
 */
double* get_numbers_stub(int size, int seed);

int* get_int_numbers_stub(int size, int seed);

/**
 *	Crea una matrice di double casuali in mem dinamica
 *	rows: numero di righe
 *	cols: numero di colonne
 *	seed: seme di partenza per la generazione casuale 
 */
double** get_double_matrix_stub(int rows, int cols, int seed);

/**
 *	Crea una matrice di int casuali in mem dinamica
 *	rows: numero di righe
 *	cols: numero di colonne
 *	seed: seme di partenza per la generazione casuale
 */
int** get_int_matrix_stub(int rows, int cols, int seed);



/**
 *	Stampa matrice su file
 *	f: file su cui stampare
 *	M: matrice
 *	
 *	Se la matrice è array di array usare *matrix*
 *	Se la matrice è array usare *vmatrix*
 *	Se il tipo è INT usare *int*, se DOUBLE usare *double*
 **/
void fprint_double_matrix(FILE* f, double** M, int num_rows, int num_cols);
void fprint_double_vmatrix(FILE* f, double* M, int num_rows, int num_cols);
void fprint_int_matrix(FILE* f, int** M, int num_rows, int num_cols);


char* tostr_double_vmatrix(double* M, int num_rows, int num_cols);


/**
 *	Restituisce una nuova matrice trasposta
 *	M: matrice di partenza
 *	rows: numero di righe
 *	cols: numero di colonne
 *
 *	Se la matrice è array di array usare *matrix*
 *	Se la matrice è un array usare *vmatrix*
 *
 **/
double* traspose_vmatrix( double* M, int rows, int cols );
double** traspose_matrix( double** M, int rows, int cols );


#endif /* UTILITY_UTILITY_H_ */
