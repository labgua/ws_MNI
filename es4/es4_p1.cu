/**
 *	Es4 p1
 *	Calcolo prodotto scalare di due vettori
 *	- Il device nel kernel calcola il vettore w prodottoCompPerComp
 *	- l'host calcola la somma dei singoli elementi
 *	Usare blocchi monodimensionali di 32 threads
 *	Contemplare sia il caso N < che N > 32D (=numero totale di threads)
 *
 **/

#include <cuda.h>
#include <stdio.h>
#include <time.h>

void initializeArray(int*,int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);

void prodottoArrayCompPerCompCPU(int* a, int* b, int* c, int n);
__global__ void prodottoArrayCompPerCompGPU(int* a, int* b, int* c, int n);

int sommaTot( int* c, int n );

int main(int argn, char* argv[]){

	//numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;

	int N; 								 //numero totale di elementi dell'array
	int *A_host, *B_host, *C_host; 		 //array memorizzati sull'host
	int *A_device, *B_device, *C_device; //array memorizzati sul device
	int *copy;							 //array in cui copieremo i risultati di C_device
	int size;							 //size in byte di ciascun array

	printf("***\t PRODOTTO SCALARE DI DUE ARRAY \t***\n");
	printf("Inserisci il numero elementi dei vettori\n");
	scanf("%d",&N); 
	//printf("Inserisci il numero di thread per blocco\n");
	//scanf("%d",&blockDim); 
	blockDim=32;

	//determinazione esatta del numero di blocchi
	gridDim = N/blockDim.x + ((N%blockDim.x)==0?0:1);

	//size in byte di ogni array
	size=N*sizeof(int);

	//stampa delle info sull'esecuzione del kernel
	printf("Numero di elementi = %d\n", N);
	printf("Numero di thread per blocco = %d\n",
	blockDim.x);
	printf("Numero di blocchi = %d\n", gridDim.x);

	//allocazione dati sull'host
	A_host=(int*)malloc(size);
	B_host=(int*)malloc(size);
	C_host=(int*)malloc(size);
	copy=(int*)malloc(size);

	//allocazione dati sul device
	cudaMalloc((void**)&A_device,size);
	cudaMalloc((void**)&B_device,size);
	cudaMalloc((void**)&C_device,size);

	//inizializzazione dati sull'host
	initializeArray(A_host, N);
	initializeArray(B_host, N);

	//copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

	//azzeriamo il contenuto della matrice C
	memset(C_host, 0, size);
	cudaMemset(C_device, 0, size);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	//invocazione del kernel
	prodottoArrayCompPerCompGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
	cudaEventRecord(stop);
	float elapsed;
	// tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("tempo GPU=%f\n", elapsed);

	//copia dei risultati dal device all'host
	cudaMemcpy(copy,C_device,size, cudaMemcpyDeviceToHost);


	///START
	clock_t start_inst = clock();
	//chiamata alla funzione seriale per il prodotto di due array
	prodottoArrayCompPerCompCPU(A_host, B_host, C_host, N);
	///STOP
	clock_t stop_inst = clock();
	double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	printf("tempo CPU=%lf\n", elapsedCPU);


	//stampa degli array e dei risultati
	if(N<20)
	{
		printf("array A\n"); stampaArray(A_host,N);
		printf("array B\n"); stampaArray(B_host,N);
		printf("Risultati host\n"); stampaArray(C_host, N);
		printf("Risultati device\n"); stampaArray(copy,N);
	}

	//test di correttezza
	equalArray(copy, C_host,N);

	printf("Esegui la somma totale sull'host\n");

	int result = sommaTot( copy, N );

	printf("La somma totale è %d\n", result);


	//de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);

	//de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

	exit(0);
}


void initializeArray(int *array, int n){
	int i;
	for(i=0;i<n;i++)
		array[i] = i;
}

void stampaArray(int* array, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%d ", array[i]);
	printf("\n");
}

void equalArray(int* a, int*b, int n){
	int i=0;
	while(a[i]==b[i])
		i++;

	if(i<n)
		printf("I risultati dell'host e del device sono diversi\n");
	else
		printf("I risultati dell'host e del device coincidono\n");
}

//Seriale
void prodottoArrayCompPerCompCPU(int *a, int *b, int *c, int n){
	int i;
	for(i=0;i<n;i++)
		c[i]=a[i]*b[i];
}

//Parallelo
__global__ void prodottoArrayCompPerCompGPU(int* a, int* b, int* c, int n){
	int index=threadIdx.x + blockIdx.x*blockDim.x;
	if(index < n)
		c[index] = a[index]*b[index];
}


//risultato tot, Seriale
int sommaTot( int* c, int n ){
	int T = 0;
	for( int i = 0; i < n; i++ )
		T += c[i];
	return T;
}