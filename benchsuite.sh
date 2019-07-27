#bash

chmod +x autobench.sh 
chmod +x es1/bench_suite_p2.sh 
chmod +x es5/tempi-bench.sh 
chmod +x es5/es5_p2-bench.sh 
chmod +x es6/es6_bench.sh 


#Es1 - Somma II strategia
./es1/bench_suite_p2.sh es1/bench_sum_ii_str.dat es1/somma es1/es1_p1

#Es1 - Somma III strategia
./autobench.sh es1/bench_sum_iii_str.dat es1/es1_p3 100000 3 5


#Es2 - Prod matrice vettore, righe
./autobench.sh es2/bech_prodmat_i_str.dat es2/mvet 100 3 5

#Es2 - Prod matrice vettore, colonne
./autobench.sh es2/bech_prodmat_ii_str.dat es2/es2_p2 100 3 5


#es4 - prod scalare - w in GPU e somma in CPU --- NO SOLO il tempo


#es5 - bench di tempi.cu
./es5/tempi-bench.sh es5/bench_es5_p1.dat es5/tempi

#es5 - somma di matrici
./es5/es5_p2-bench.sh es5/bench_es5_p2.dat es5/es5_p2

#es6 - prodotto scalare
./es6/es6_bench.sh es6/bench_es6.dat es6/es6
