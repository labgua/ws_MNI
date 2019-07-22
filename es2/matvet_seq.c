
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../utility/utility.h"

int main(int argc, char const *argv[])
{
	
	int n = 10;
	int seed = 1;
	int seed2 = 7;

	if(argc > 1){
		n = atoi(argv[1]);
	}

	double** A = get_double_matrix_stub(n, n, seed);
	double*  X = get_numbers_stub(n, seed2);
	double*  Y = malloc( sizeof(double) * n );

	DD printf("MatVet sequenziale\n");

	DD fprintf(stdout, "A matrix\n");
	DD fprint_double_matrix(stdout, A, n, n);
	DD fprint_row(stdout, "X vector", X, n);

	///START
	clock_t start_inst = clock();

	for( int i = 0; i < n; i++ ){
		Y[i] = 0;
		for( int j = 0; j < n; j++ ){
			Y[i] = Y[i] + A[i][j] * X[j];
		}
	}

	///STOP
	clock_t stop_inst = clock();

	double timer = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	DD fprint_row(stdout, "Y result", Y, n);

	printf("Time execution %lf\n", timer);

	return 0;

}