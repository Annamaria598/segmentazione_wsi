# Classificazione con KNN

In questa sezione è stato utilizzato l'algoritmo knn
per allenare dei modelli per la segmentazione automatica.

Il file **classificazione_knn.py** contiene tutte le funzioni
con rispettiva documentazione usate nei vari step del lavoro.

L'applicazione delle funzionalità sviluppate al fine di generare
i modelli migliori ottenuti durante la grid search è disponibile
sia in formato jupyter notebook nel file **KNN.ipynb** che come
script python eseguibile da riga di comando nel file **knn.py**.

Sia nel notebook KNN.ipynb che nello script knn.py viene
allenato e valutato un singolo modello utilizzato anche per
la generazione delle immagini relative alla segmentazione con
e senza post-processing confrontate all'immagine di test originale.

Nelle varie cartelle fileCSV_GSX_risultati (con X valore numerico)
sono presenti le diverse fasi della grid search con i relativi
risultati. Il numero rappresenta il tentativo di
ricerca di features e parametri.

I risultati delle grid search sono disponibili in formato
csv e sono nominati nella forma results_gsX.csv con X
valore numerico corrispondente alla fase della grid search.