
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

#include "../utility/utility.h"

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

/**
* Funzione per ottenere un elemento della matrice.
* A     puntatore alla matrice
* ROWS  numero di righe della matrice
* COLS  numero di colonne della matrice
* r     riga dell'elemento da prelevare
* c     colonna dell'elemento da prelevare
*/
double matrix_get(double* A, int ROWS, int COLS, int r, int c)
{
    return A[ (r * COLS) + c ];
}

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
double matrix_set(double* A, int ROWS, int COLS, int r, int c, double v)
{
    A[(r * COLS) + c] = v;
}

#define MATRIX_SET(A, ROWS, COLS, r, c, v)	A[(r * COLS) + c] = v

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




int main(int argc, char **argv)
{
	int nproc;              // Numero di processi totale
	int me;                 // Il mio id
	int n;                  // Dimensione della matrice
	int local_n;            // Dimensione dei dati locali
	int i,j;                    // Iteratori vari 

	// Variabili di lavoro
	double* A, *TA;
	double* v, *local_v;
	double* localA;
	double* local_w;
	double* w;

	// variabili timer
	double T_inizio,T_fine,T_max;

	/*Attiva MPI*/
	MPI_Init(&argc, &argv);
	/*Trova il numero totale dei processi*/
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);
	/*Da ad ogni processo il proprio numero identificativo*/
	MPI_Comm_rank (MPI_COMM_WORLD, &me);


	// Se sono a radice
	if(me == 0)
	{
		DD printf("Prodotto matrice-vettore II strategia (colonne).\n");
		DD printf("inserire n = \n"); 
	 	scanf("%d",&n); 
		// Porzione di dati da processare
		local_n = n/nproc;  
	    
		// Calcolo la matrice A e alloco v e w
		A = get_matrix_space(n,n); 
		v = malloc(sizeof(double)*n);
		w =  malloc(sizeof(double)*n); 
	    
		DD printf("A = \n"); 
		for (i=0;i<n;i++)
		{
			v[i]=i;  
			for(j=0;j<n;j++)
			{
				if (j==0)
	    			MATRIX_SET(A, n, n, i, j, (1.0/(i+1))-1);
	    		else
	    			MATRIX_SET(A, n, n, i, j, (1.0/(i+1))-(pow(1.0/2.0,j))); 
	    		DD printf("%f ", MATRIX_GET(A, n, n, i, j) );
	    	}
	    	DD printf("\n");
	    }
	    DD printf("\n"); 

	    DD printf("Crea la matrice trasposta\n");
	    TA = traspose_vmatrix( A, n, n );

	    DD{
			printf("stampa di A\n");
			fprint_double_vmatrix(stdout, A, n, n);
			printf("\nstampa di TA\n");
			fprint_double_vmatrix(stdout, TA, n, n);
		}

	    DD{
	    	printf("v = \n"); 
	    	for (i=0;i<n;i++)
	    	{   
	    		printf("%f\n", v[i]);
	    	}
	    	printf("\n");
	    }
	}


	////////// START
	MPI_Barrier(MPI_COMM_WORLD);
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

	//////CODICE!!!
	// Spedisco n e local_v
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);            
	MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);

	// Se sono un figlio alloco local_v, che inquesto caso è ridotto
	//if(me != 0)
		local_v = malloc(sizeof(double)*local_n);

	// invio in scatter del pezzo del vettore
	MPI_Scatter(
		&v[0], local_n, MPI_DOUBLE,
		&local_v[0], local_n, MPI_DOUBLE,
		0, MPI_COMM_WORLD); 

	DD fprint_row(stdout, "local_v received: \n", local_v, local_n );

	// tutti allocano A locale e il vettore dei risultati
	localA  = get_matrix_space(local_n, n);
	local_w = malloc(n * sizeof(double));

	// Adesso 0 invia a tutti un pezzo della matrice
	// usando TA, la matrice trasposta generata da A
	int num = local_n*n;
	MPI_Scatter(
	    &TA[0], num, MPI_DOUBLE,
	    &localA[0], num, MPI_DOUBLE,
	    0, MPI_COMM_WORLD);



	// Scriviamo la matrice locale ricevuta
	DD{
	    printf("localA di %d = \n", me); 
	    fprint_double_vmatrix(stdout, localA, local_n, n);
	}

	// Effettuiamo i calcoli, operando in trasposta
	prod_mat_vett_intrasp(local_w,localA,local_n,n,local_v);

	//stampa del risultato locale
	DD fprint_row(stdout, "local_w result:\n", local_w, n);

	// 0 raccoglie i risultati parziali, FACENDONE LA SOMMA
	MPI_Reduce(&local_w[0], &w[0], n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	/////////STOP
	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine

	/* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	// 0 stampa la soluzione
    if(me==0) 
    	DD fprint_row(stdout, "w result in root :\n", w, n);

	/*stampa per benchmark*/
	if(me==0)
	{
	    printf("Processori impegnati: %d\n", nproc);
	    printf("Tempo calcolo locale: %lf\n", T_fine);
	    printf("MPI_Reduce max time: %f\n",T_max);
	}// end if



	MPI_Finalize (); /* Disattiva MPI */
	return 0; 

}