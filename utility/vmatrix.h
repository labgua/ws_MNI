/*
 * vmatrix.h
 *
 *  Created on: 20 lug 2019
 *      Author: sergio
 *
 *	vmatrix definisce la struttura dati matrice
 *	definita su un vettore, composta da blocchi
 *	di memoria contigua.
 *	Essa è indispensabile per la comunicazione MPI
 *	
 *	Questi codici sono del corso MNI e sono stati
 *	rielaborati
 */

#ifndef UTILITY_VMATRIX_H_
#define UTILITY_VMATRIX_H_

/**
* Funzione per allocare lo spazio per una matrice
* ROWS  numero di righe
* COLS  numero di colonne
*/
double* get_matrix_space(int ROWS, int COLS);


/**
* Funzione per ottenere un elemento della matrice.
* A     puntatore alla matrice
* ROWS  numero di righe della matrice
* COLS  numero di colonne della matrice
* r     riga dell'elemento da prelevare
* c     colonna dell'elemento da prelevare
*/
#define MATRIX_GET(A, ROWS, COLS, r, c)		A[ (r * COLS) + c ]


/**
* Funzione per impostare un elemento della matrice.
* A     puntatore alla matrice
* ROWS  numero di righe della matrice
* COLS  numero di colonne della matrice
* r     riga dell'elemento da impostare
* c     colonna dell'elemento da impostare
* v     valore da impostare
*/
#define MATRIX_SET(A, ROWS, COLS, r, c, v)	A[(r * COLS) + c] = v

#endif /* UTILITY_VMATRIX_H_ */