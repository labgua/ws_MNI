#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#include "../utility/utility.h"
#include "../utility/vmatrix.h"

/**
* Esegue il prodotto matrice vettore
*/
void prod_mat_vett(double w[], double *a, int ROWS, int COLS, double v[])
{
    int i, j;
    
    for(i=0;i<ROWS;i++)
    {
        w[i]=0;
        for(j=0;j<COLS;j++)
        { 
            w[i] += MATRIX_GET(a, ROWS, COLS, i, j) * v[j];
        } 
    }    
}



/**
* Esegue il prodotto matrice vettore su trasposta
*/
void prod_mat_vett_intrasp(double w[], double *a, int ROWS, int COLS, double v[])
{

	//int me;
	//MPI_Comm_rank (MPI_COMM_WORLD, &me);

    int i, j;

    //printf("p:%d -- prod mat in trasp, ROWS=%d  COLS=%d\n", me, ROWS, COLS);
    
    for(i=0;i<COLS;i++)
    {
        w[i]=0;
        for(j=0;j<ROWS;j++)
        { 
        	//printf("---%d---- M[j=%d][i=%d]=%f  *  v[j=%d]=%f\n", me, j, i, MATRIX_GET(a, ROWS, COLS, j, i) , j, v[j] );
            w[i] += MATRIX_GET(a, ROWS, COLS, j, i) * v[j];
        } 
    }    
}


int main(int argc, char* argv[]){
	int menum, numproc;
	int menum_grid, row, col;
	int dim, *ndim, reorder, *period, coordiate[2];
	MPI_Comm comm_grid;

	int n, p, q;

	double *A, *v;		//A:Matrice, v:vettore

	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	if( menum == 0 ){
		DD printf("Es3 p1 - Distribuzione di dati Mat-Vet, caso numproc=p*q con p=q ed n=p*q \n");

		DD printf("Inserisci il numero di righe-colonne della matrice: \n");
		scanf("%d", &n);

		//leggi il numero di righe della griglia
		DD printf("Inserisci il numero p della griglia (num righe): \n");
		scanf("%d", &row);
		DD printf("Inserisci il numero q della griglia (num colonne): \n");
		scanf("%d", &col);

		printf("CHECK numproc=p*q ... ");
		if( row * col != numproc ){
			printf("FALSE! exit..\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}
		printf("OK\n");


		// Calcolo la matrice A e alloco v e w
		A = get_matrix_space(n,n); 
		v = malloc(sizeof(double)*n);
		//w =  malloc(sizeof(double)*n);

		// inizializzo matrice e vettore, con numeri progressivi
		// --> cio ci sarà di aiuto per verificare il funzionamento
		for( int i = 0; i < n; i++ ){
			v[i] = i;
			for (int j = 0; j < n; j++)
			{
				MATRIX_SET(A, n, n, i, j,   i*n + j   );
			}
		}

		DD{
			printf("\nStampa di A\n%s\n", tostr_double_vmatrix(A, n, n));
			printf("Stampa di v\n%s\n", tostr_row(v, n));
		}

	}

	MPI_Barrier(MPI_COMM_WORLD);


	/////////// CREAZIONE GRIGLIA PROCESSORI
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	dim = 2; //dimensioni della griglia

	//vettore contenente le lunghezze di ciascuna dimensione
	ndim = malloc( sizeof(int) * 2 );
	ndim[0] = row;
	ndim[1] = col;

	//Vettore contenete le periodicità
	period = malloc( sizeof(int) * dim );
	period[0] = 0;
	period[1] = 0;

	//riodino id dei processori, 0 nessun riordino
	reorder = 0;


	//creazione griglia bidimensionale
	MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder, &comm_grid);

	//Prendi il rank del processore i-esimo sul nuovo communicator
	MPI_Comm_rank(comm_grid, &menum_grid);

	//Ottieni le coordinate del processore i-esimo
	MPI_Cart_coords(comm_grid, menum_grid, dim, coordiate);


	//////creazione dei sottogruppi, righe e colonne
	int belongs[2];
	int idrow, coord_row,   idcol, coord_col;
	MPI_Comm commrow, commcol;

	//creazione gruppi righe
	belongs[0] = 0;
	belongs[1] = 1;
	MPI_Cart_sub(comm_grid, belongs, &commrow);
	MPI_Comm_rank(commrow, &idrow);
	MPI_Cart_coords(commrow, idrow, dim, &coord_row);

	//creazione gruppi colonne
	belongs[0] = 1;
	belongs[1] = 0;
	MPI_Cart_sub(comm_grid, belongs, &commcol);
	MPI_Comm_rank(commcol, &idcol);
	MPI_Cart_coords(commcol, idcol, dim, &coord_col);


	DD ordered_printf(MPI_COMM_WORLD, "CHECK GRIGLIA -- ABS(%d,%d)   C-R(%d)  C-C(%d)", coordiate[0], coordiate[1], coord_row, coord_col);
	DD MPI_Barrier(MPI_COMM_WORLD);

	/////////// fine CREAZIONE GRIGLIA PROCESSORI

	//ALGO
	// fase matrice
	// 1 il processore p0 fa scatter inviando righe a tutti i processori della col0
	// 2 per ogni processore pi della col0
	// 3	calcola TA=trasposta della matrice ricevuta
	// 4	pi fa scatter inviando righe (quindi colonne) a tutti i processi della rowi

	//fase vettore
	// A il vettore viene ripartito su tutti processi della row0
	// B poi ogni processore pi di row0 lo copia ai processi della coli

	//1
	int local_n = n/row;
	double* localA = get_matrix_space(local_n, n); 
	double* localAA = get_matrix_space(local_n, local_n); 
	double* TA = NULL;

	if( coord_row == 0  ){
		int num = local_n*n;

		MPI_Scatter(
			A, num, MPI_DOUBLE,
			localA, num, MPI_DOUBLE,
			0, commcol //commrom
		);

		//2 e 3
		TA = traspose_vmatrix( localA, local_n, n );
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/*
	printf("stampa di localA\n");
	fprint_double_vmatrix(stdout, localA, local_n, n);
	printf("\n");
	printf("stampa di TA\n");
	fprint_double_vmatrix(stdout, TA, n, local_n);
	printf("\n");
	fflush(stdout);
	*/

	//4
	int num2 = local_n*local_n; //quante colonne sono???
	//ordered_printf(MPI_COMM_WORLD, "2scatter");
	MPI_Scatter(
		TA, num2, MPI_DOUBLE,
		localAA, num2, MPI_DOUBLE,
		0, commrow  //commcol
	);

	// TRASPOSTA, passo non obbligatorio
	// è possibile utilizzare la funzione prodMat che lavora
	// sulla matrice trasposta
	localAA = traspose_vmatrix(localAA, local_n, local_n);

	////fprint_double_vmatrix(stdout, localAA, local_n, local_n);
	ordered_printf(MPI_COMM_WORLD, "\nstampa di localAA (ritrasposto)\n%s", tostr_double_vmatrix(localAA, local_n, local_n));


	//fase vettore
	//A scatter sui processori della prima colonna
	double* local_v = malloc(local_n * sizeof(double));
	if( coord_col == 0 ){
		MPI_Scatter(
			v, local_n, MPI_DOUBLE,
			local_v, local_n, MPI_DOUBLE,
			0, commrow
		);
	}

	//B broadcast sui processori delle singole colonne
	MPI_Bcast(local_v, local_n, MPI_DOUBLE, 0, commcol);


	DD ordered_printf(MPI_COMM_WORLD, "local_v: %s", tostr_row(local_v, local_n));

	//a questo punto è possibile adottare un prodotto
	//matrice vettore che lavota sulla matrice trasposta 
	//oppure ricreare una trasposta per ottenere la
	//sottomatrice di definizione 

	//TODO ....

	MPI_Finalize();

	return 0;
}
