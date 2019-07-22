
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>
#include "utility.h"

void fprint_row(FILE* f, char* label, double* v, int n){

	fprintf(f,"%s -- [ ", label);
	for(int i = 0; i < n; i++){
		fprintf(f,"%lf  ", v[i]);
	}
	fprintf(f,"]\n");
}

void fprint_int_row(FILE* f, char* label, int* v, int n){

	fprintf(f,"%s -- [ ", label);
	for(int i = 0; i < n; i++){
		fprintf(f,"%d  ", v[i]);
	}
	fprintf(f," ]\n");
}

char* tostr_row(double* v, int n){
	MSG_BUFFER[0] = '\0';
	int offset = 0;
	for(int i = 0; i < n; i++){
		offset += sprintf(MSG_BUFFER + offset,"%lf  ", v[i]);
	}
	return MSG_BUFFER;
}


double* get_numbers_stub(int size, int seed){

	double* v = malloc(sizeof(double) * size);

	srand(seed);

	for( int i = 0; i < size; i++ ){
		v[i] = (double)(rand() % MAX_NUM);
	}

	return v;

}

int* get_int_numbers_stub(int size, int seed){

	int* v = malloc(sizeof(int) * size);

	srand(seed);

	for( int i = 0; i < size; i++ ){
		v[i] = (int)(rand() % MAX_NUM);
	}

	return v;

}

double** get_double_matrix_stub(int rows, int cols, int seed){

	srand(seed);
	
	double** M = malloc( sizeof(double*) * rows );

	for( int i = 0; i < rows; i++ ){
		M[i] = malloc( sizeof(double) * cols );

		for( int j = 0; j < cols; j++ ){
			M[i][j] = (double)(rand() % MAX_NUM);
		}

	}

	return M;

}


int** get_int_matrix_stub(int rows, int cols, int seed){

	srand(seed);

	int** M = malloc( sizeof(int*) * rows );

	for( int i = 0; i < rows; i++ ){
		M[i] = malloc( sizeof(int) * cols );

		for( int j = 0; j < cols; j++ ){
			M[i][j] = (rand() % MAX_NUM);
		}

	}

	return M;

}

void fprint_double_vmatrix(FILE* f, double* M, int num_rows, int num_cols){

	for( int i = 0; i < num_rows; i++ ){
		for( int j = 0; j < num_cols; j++ ){
			fprintf(f, "%3.3f\t ", M[(i * num_cols) + j]);
		}
		fprintf(f, "\n");
	}

}

char* tostr_double_vmatrix(double* M, int num_rows, int num_cols){
	MSG_BUFFER[0] = '\0';
	int offset = 0;
	for( int i = 0; i < num_rows; i++ ){
		for( int j = 0; j < num_cols; j++ ){
			offset += sprintf(MSG_BUFFER + offset, "%3.3f\t ", M[(i * num_cols) + j]);
		}
		offset += sprintf(MSG_BUFFER + offset, "\n");
	}
	return MSG_BUFFER;
}


void fprint_double_matrix(FILE* f, double** M, int num_rows, int num_cols){

	for( int i = 0; i < num_rows; i++ ){
		for( int j = 0; j < num_cols; j++ ){
			fprintf(f, "%3.3f\t ", M[i][j]);
		}
		fprintf(f, "\n");
	}

}

void fprint_int_matrix(FILE* f, int** M, int num_rows, int num_cols){

	for( int i = 0; i < num_rows; i++ ){
		for( int j = 0; j < num_cols; j++ ){
			fprintf(f, "%d\t ", M[i][j]);
		}
		fprintf(f, "\n");
	}

}


void ordered_printf(MPI_Comm comm, const char *format, ...){

    va_list vargs;
    va_start(vargs, format);

    int menum;
    MPI_Barrier(comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &menum);
    usleep(menum * 500);

    printf("p:%d [ ", menum);
    vprintf(format, vargs);
    printf(" ]\n");

    va_end(vargs);
}


/**
 *	crea una matrice trasposta
 *  legge e crea una matrice come vettore di double 
 **/
double* traspose_vmatrix( double* M, int rows, int cols ){
	double* N = malloc( sizeof(double) * rows * cols );
	for(int i = 0; i < rows; i++){
		for( int j = 0; j < cols; j++ ){
			////MATRIX_GET(A, ROWS, COLS, r, c)		A[ (r * COLS) + c ]
			///N[(j * cols) + i] = M[(i * cols) + j];
			N[(j * rows) + i] = M[(i * cols) + j];
		}
	}
	return N;
}

double** traspose_matrix( double** M, int rows, int cols ){
	double** N = malloc( sizeof(double*) * rows );
	for( int i = 0; i < rows; i++ ){
		N[i] = malloc( sizeof(double) * cols );
	}

	for( int i = 0; i < rows; i++ ){
		for (int j = 0; i < cols; j++){
			N[j][i] = M[i][j];
		}
	}
	return N;
}