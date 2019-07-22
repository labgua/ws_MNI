
#include <stdlib.h>
#include "vmatrix.h"

/**
* Funzione per allocare lo spazio per una matrice
* ROWS  numero di righe
* COLS  numero di colonne
*/
double* get_matrix_space(int ROWS, int COLS)
{
    double* A = malloc(ROWS * COLS * sizeof(double));
    return A;
}