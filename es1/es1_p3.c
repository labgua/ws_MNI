#include <stdio.h>
#include <time.h>
#include <mpi.h>

#include "../utility/utility.h"

int main (int argc, char **argv){

	/*dichiarazioni variabili*/
	int menum,nproc,tag;
	int n,nloc,i,somma,resto,nlocgen;
	int ind,p,r,sendTo,recvBy,tmp;
	int *potenze,*vett,*vett_loc,passi=0;
	int sommaloc=0;
	double T_inizio,T_fine,T_max;

	MPI_Status info;
	
	/*Inizializzazione dell'ambiente di calcolo MPI*/
	MPI_Init(&argc,&argv);
	/*assegnazione IdProcessore a menum*/
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	/*assegna numero processori a nproc*/
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);


	/* lettura e inserimento dati*/
	if ( menum == 0 ) {

		DD printf("Somma III strategia\n");
		DD printf("Inserire il numero di elementi da sommare: \n");
		fflush(stdout);
		scanf("%d",&n);
		
       	vett=(int*)calloc(n,sizeof(int));
	}

	/*invio del valore di n a tutti i processori appartenenti a MPI_COMM_WORLD*/
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);



	////////PARTIZIONAMENTO DATI
	/*numero di addendi da assegnare a ciascun processore*/
	nlocgen=n/nproc; // divisione intera
	
	resto=n%nproc; // resto della divisione

	/* Se resto è non nullo, i primi resto processi ricevono un addento in più */
	if(menum<resto){
		nloc=nlocgen+1;
	}
	else{
		nloc=nlocgen;
	}
	////////////////////////////////////////


	////////ALLOCAZIONE DI MEMORIA del vettore per le somme parziali
	vett_loc=(int*)calloc(nloc, sizeof(int));
	////////////////////////////////////////




	////////GENERAZIONE, INVIO e RICEZIONE DATI
	// P0 genera e assegna gli elementi da sommare
	if ( menum == 0 ){

        /*Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema*/                
        srand((unsigned int) time(0)); 
		
        for(i=0; i<n; i++){
			/*creazione del vettore contenente numeri casuali */
			*(vett+i)=(int)rand()%5-2;
		}
		
   		// Stampa del vettore che contiene i dati da sommare, se sono meno di 100 
		if (n<100){

			for (i=0; i<n; i++){

				DD printf("\n\nElemento %d del vettore =%d ",i,*(vett+i));
			}
        }

		//assegnazione dei primi addendi a P0
        for(i=0;i<nloc;i++){
			*(vett_loc+i)=*(vett+i);
		}
  
  		//ind è il numero di addendi già assegnati     
		ind=nloc;
        
		/* P0 assegna i restanti addendi agli altri processori */
		for(i=1; i<nproc; i++){

			tag=i; /* tag del messaggio uguale all'id del processo che riceve*/

			/*SE ci sono addendi in sovrannumero da ripartire tra i processori*/
            if (i<resto) 
			{
				/*il processore P0 gli invia il corrispondete vettore locale considerando un addendo in piu'*/
				MPI_Send(vett+ind,nloc,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind=ind+nloc;
			} 
			else 
			{
				/*il processore P0 gli invia il corrispondete vettore locale*/
				MPI_Send(vett+ind,nlocgen,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind=ind+nlocgen;
			}// end if
		}//end for

	}
    /*SE non siamo il processore P0 riceviamo i dati trasmessi dal processore P0*/
    else
    {
		// tag è uguale numero di processore che riceve
		tag=menum;
  
		/*fase di ricezione*/
		MPI_Recv(vett_loc,nloc,MPI_INT,0,tag,MPI_COMM_WORLD,&info);
	}// end if
	////////////////////////////////////////


	/* sincronizzazione dei processori del contesto MPI_COMM_WORLD*/
	MPI_Barrier(MPI_COMM_WORLD);


	/// da questo momento avviamo il timer, in quanto i calcoli delle  
	/// potenze fanno comunque parte della dell'algoritmo parallelo

	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio
 

	//CALCOLO PROPRIA SOMMA PARZIALE
 	for(i=0;i<nloc;i++){
		/*ogni processore effettua la somma parziale*/
		sommaloc=sommaloc+*(vett_loc+i);
	}

	//DD printf("--> sono p%d e la mia somma è %d\n", menum, sommaloc);


	//  calcolo di p=log_2 (nproc)
	p=nproc;
	while( p != 1 ){
		/*shifta di un bit a destra*/
		p = p >> 1;
		passi++;
	}

	/* creazione del vettore potenze, che contiene le potenze di 2*/
	potenze = (int*) calloc( passi + 1, sizeof(int) );
	for( i = 0; i <= passi; i++ ){
		potenze[i] = p << i;
	}


	///// INIZIO DEI PASSI DELLA 3 STRATEGIA
	/* calcolo delle somme parziali e combinazione dei risultati parziali */
	for(i=0;i<passi;i++){

		// ... calcolo identificativo del processore
		r = menum % potenze[i+1];

		if( r < potenze[i] ){
			// comunica con me+dist
			// 1) invia la propria somma a me+dist
			// 2) ricevi la somma da me+dist
			sendTo=menum+potenze[i];
			recvBy=sendTo;

			tag=sendTo;

			MPI_Send(&sommaloc,1,MPI_INT,sendTo,tag,MPI_COMM_WORLD);

			tag=menum;

			MPI_Recv(&tmp,1,MPI_INT,recvBy,tag,MPI_COMM_WORLD,&info);
		
		}
		else{
			// comunica con me-dist
			// 2) ricevi la somma da me-dist
			// 1) invia la propria somma a me-dist
			sendTo=menum-potenze[i];
			recvBy=sendTo;
			tag=menum;

			MPI_Recv(&tmp,1,MPI_INT,recvBy,tag,MPI_COMM_WORLD,&info);

			tag=sendTo;

			MPI_Send(&sommaloc,1,MPI_INT,sendTo,tag,MPI_COMM_WORLD);
		}

		sommaloc=sommaloc+tmp;

	}



	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine
 
	/* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	/*stampa a video dei risultati finali*/
	if(menum==0)
	{
		printf("Processori impegnati: %d\n", nproc);
		printf("La somma e': %d\n", sommaloc);
		printf("Tempo calcolo locale: %lf\n", T_fine);
		printf("MPI_Reduce max time: %f\n",T_max);
	}// end if
 
	/*routine chiusura ambiente MPI*/
	MPI_Finalize();

}