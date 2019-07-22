/*
 ============================================================================
 Name        : es1_p1.c
 Author      : Sergio Guastaferro
 Version     :
 Copyright   : Your copyright notice
 Description : Realizzare un programma sequenziale per la somma di N numeri
 ============================================================================
 */

#include <stdlib.h>
#include <time.h>
#include "../utility/utility.h"

int main(int argc, char **argv) {

	int n = 10;
	int seed = 1;

	if(argc > 1){
		n = atoi(argv[1]);
	}

	double* v = get_numbers_stub(n, seed);

	DD fprint_row(stdout, "numeri generati", v, n);

	///START
	clock_t start_inst = clock();

	int sum = v[0];

	for( int  i = 1; i < n; i++ ){
		sum += v[i];
	}

	///STOP
	clock_t stop_inst = clock();

	double timer = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	DD printf("la somma dei numeri è : %d\n", sum);

	printf("Time execution %lf\n", timer);

	return 0;
}
