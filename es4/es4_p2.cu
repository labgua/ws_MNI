/**
 *	es4_p2
 *	- somma di due matrici
 * eseguito su GeForce 940MX:
 * Maximum number of threads per block:           1024
 *
 **/

#include <cuda.h>
#include <stdio.h>

#define MAX_THREAD_PER_BLOCK 1024

void initializeArray(int*,int);
void stampaMat(int*, int);
void equalArray(int*, int*, int);


void sumMatrixCPU(int* A, int* B, int* C, int N);

__global__ void sumMatrixGPU(int* A, int* B, int* C, int N);


int main(int argn, char* argv[]){

	dim3 gridDim, blockDim(8, 4); 

	int N; 								 //numero totale di elementi dell'array
	int *A_host, *B_host, *C_host; 		 //array memorizzati sull'host
	int *A_device, *B_device, *C_device; //array memorizzati sul device
	int *copy;							 //array in cui copieremo i risultati di C_device
	int size;							 //size in byte di ciascun array

	printf("***\t SOMMA DI DUE MATRICI \t***\n");
	printf("Inserisci N num righe e colonne : ");
	scanf("%d", &N);

	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0? 0: 1);
    gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0? 0: 1);

    printf("gridDim:X=%d  gridDim:Y=%d\n", gridDim.x, gridDim.y);

	//size in byte di ogni array
	size = N * N * sizeof(int);

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
	initializeArray(A_host, N*N);
	initializeArray(B_host, N*N);


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
	sumMatrixGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N*N);
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
	//chiamata alla funzione seriale 
	sumMatrixCPU(A_host, B_host, C_host, N);
	///STOP
	clock_t stop_inst = clock();
	double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	printf("tempo CPU=%lf\n", elapsedCPU);



	//stampa degli array e dei risultati
	if(N<20)
	{
		printf("matrice A\n"); stampaMat(A_host,N);
		printf("matrice B\n"); stampaMat(B_host,N);
		printf("Risultati host\n"); stampaMat(C_host, N);
		printf("Risultati device\n"); stampaMat(copy,N);
	}

	//test di correttezza
	equalArray(copy, C_host,N*N);

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

void stampaMat(int* array, int n){
	int i,j;
	for( i = 0; i < n; i++ ){
		for( j = 0; j < n; j++ ){
			printf("%d\t", array[ (i * n) + j ]  );
		}
		printf("\n");
	}
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






//seriale
void sumMatrixCPU(int* A, int* B, int* C, int N){
	int l=N*N;
	for( int i = 0; i < l; i++ ){
		C[i] = A[i] + B[i];
	}
}

//parallelo
__global__ void sumMatrixGPU(int* A, int* B, int* C, int N){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = j * gridDim.x * blockDim.x + i;

	if( index < N )
		C[index] = A[index] + B[index];

}

