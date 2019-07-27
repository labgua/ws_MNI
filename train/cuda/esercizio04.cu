#include <cuda.h>
#include <stdio.h>

void initializeArray(int*,int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);

void sommaArrayCompPerCompCPU(int *a, int *b, int *c, int n);
__global__ void sommaArrayCompPerCompGPU(int* a, int* b, int* c, int n);

void prodScalarePerVettCPU(int* a, int k, int* c, int n);
__global__ void prodScalarePerVettGPU(int* a, int k, int* c, int n);

int main(int argn, char* argv[]){

	//numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;

	int N; 						//numero totale di elementi dell'array
	int *A_host, *C_host;		//array memorizzati sull'host
	int *A_device, *C_device;	//array memorizzati sul device
	int *copy;					//array in cui copieremo i risultati di C_device
	int size;					//size in byte di ciascun array
	int k;						//scalare, k=N

	printf("***\t PRODOTTO SCALARE PER UN VETTORE \t***\n");
	printf("Inserisci il numero elementi dei vettori\n");
	scanf("%d",&N); 

	printf("Inserisci il numero di thread per blocco\n");
	scanf("%d",&blockDim); 

	//determinazione esatta del numero di blocchi
	gridDim = N/blockDim.x + ((N%blockDim.x)==0?0:1);

	//size in byte di ogni array
	size=N*sizeof(int);

	//stampa delle info sull'esecuzione del kernel
	printf("Numero di elementi = %d\n", N);
	printf("Numero di thread per blocco = %d\n", blockDim.x);
	printf("Numero di blocchi = %d\n", gridDim.x);

	//allocazione dati sull'host
	A_host=(int*)malloc(size);
	C_host=(int*)malloc(size);
	copy=(int*)malloc(size);

	//allocazione dati sul device
	cudaMalloc((void**)&A_device,size);
	cudaMalloc((void**)&C_device,size);

	//inizializzazione dati sull'host
	initializeArray(A_host, N);
	k = N;

	//copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);

	//azzeriamo il contenuto della matrice C
	memset(C_host, 0, size);
	cudaMemset(C_device, 0, size);

	//invocazione del kernel
	prodScalarePerVettGPU<<<gridDim, blockDim>>>(A_device, k, C_device, N);

	//copia dei risultati dal device all'host
	cudaMemcpy(copy,C_device,size, cudaMemcpyDeviceToHost);


	//chiamata alla funzione seriale
	prodScalarePerVettCPU(A_host, k, C_host, N);

	//stampa degli array e dei risultati
	if(N<20)
	{
		printf("array A\n"); stampaArray(A_host,N);
		printf("k = %d\n", k);
		printf("Risultati host\n"); stampaArray(C_host, N);
		printf("Risultati device\n"); stampaArray(copy,N);
	}

	//test di correttezza
	equalArray(copy, C_host,N);

	//de-allocazione host
	free(A_host);
	free(C_host);
	free(copy);

	//de-allocazione device
	cudaFree(A_device);
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
void prodScalarePerVettCPU(int* a, int k, int* c, int n){
	int i;
	for(i=0;i<n;i++)
		c[i]=a[i]*k;	
}

//Parallelo
__global__ void prodScalarePerVettGPU(int* a, int k, int* c, int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index < n )
		c[index] = a[index] * k;
}