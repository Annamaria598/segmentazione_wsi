All'interno di questo repository sono presenti 4 cartelle così suddivise:

- all_results che contiene per comodità una copia di tutte le immagini che rappresentano le segmentazioni ottenute con i modelli allenati corredate di score e di matrici di confusione per KNN, random forest e SVM.

- **segmentazione_knn** che contiene il codice per la generazione dei modelli knn e le immagini che rappresentano le segmentazioni ottenute con tali modelli (più rispettive matrici di confusione e score) suddivise in training set, validation set e test set.
Nel file GS7_results.csv è presente il risultato della grid search.
Tutte le funzioni di supporto sono disponibili nel file classificazione_knn.py.

Nel notebook KNN.ipynb è presente il training del singolo modello e la generazione delle immagini da esso prodotte con e senza post-processing confrontate con l'immagine di test originale.

- **segmentazione_random_forest** che contiene il codice per la generazione dei modelli random forest e le immagini che rappresentano le segmentazioni ottenute con tali modelli (più rispettive matrici di confusione e score) suddivise in training set, validation set e test set.
Nel file GS7_results.csv è presente il risultato della grid search.
Tutte le funzioni di supporto sono disponibili nel file funzioni.py.

Nel notebook RF.ipynb è presente il training del singolo modello e la generazione delle immagini da esso prodotte con e senza post-processing confrontate con l'immagine di test originale.

- **segmentazione_svm** che contiene il codice per la generazione dei modelli SVM e le immagini che rappresentano le segmentazioni ottenute con tali modelli (più rispettive matrici di confusione e score) suddivise in training set, validation set e test set.
Nel file GS7_results.csv è presente il risultato della grid search.
Tutte le funzioni di supporto sono disponibili nel file funzioni.py.

Nel notebook SVM.ipynb è presente il training del singolo modello e la generazione delle immagini da esso prodotte con e senza post-processing confrontate con l'immagine di test originale.
