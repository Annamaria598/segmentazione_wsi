from classificazione_knn import image_to_data,\
    colimage_to_classes,\
    local_entropy,\
    classes_to_colimage,\
    postprocessing_classes,\
    select_layers,\
    image_segmentation,\
    train_on_multi,\
    test_on_multiple_images
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imageio
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, plot_confusion_matrix

# Creazione del modello allenandolo su più immagini
print("Inizio training del modello knn...")
knn, score, conf_matrix = train_on_multi(
    ["img/pelle303R.PNG", "img/pelle304R.PNG", "img/pelle601R.PNG"],
    ["img/pelle303R_colors.PNG", "img/pelle304R_colors.PNG", "img/pelle601R_colors.PNG"],
    ["img/pelle301R.PNG", "img/pelle302R.PNG", "img/pelle305R.PNG"],
    ["img/pelle301R_colors.PNG", "img/pelle302R_colors.PNG", "img/pelle305R_colors.PNG"],
    window_size=30,
    neighbors=50,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)

print("Training completo.")
print("Score: {}".format(score))

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 1001R")
image_segmentation(
    knn,
    "img/test_set/pelle1001R.PNG",
    "img/test_set/pelle1001R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 1002R")
image_segmentation(
    knn,
    "img/test_set/pelle1002R.PNG",
    "img/test_set/pelle1002R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 101R")
image_segmentation(
    knn,
    "img/test_set/pelle101R.PNG",
    "img/test_set/pelle101R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 102R")
image_segmentation(
    knn,
    "img/test_set/pelle102R.PNG",
    "img/test_set/pelle102R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 103R")
image_segmentation(
    knn,
    "img/test_set/pelle103R.PNG",
    "img/test_set/pelle103R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 104R")
image_segmentation(
    knn,
    "img/test_set/pelle104R.PNG",
    "img/test_set/pelle104R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Segmentazione su img di test 105R")
image_segmentation(
    knn,
    "img/test_set/pelle105R.PNG",
    "img/test_set/pelle105R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

test_img_orig = [
    "img/test_set/pelle1001R.PNG",
    "img/test_set/pelle1002R.PNG",
    "img/test_set/pelle101R.PNG",
    "img/test_set/pelle102R.PNG",
    "img/test_set/pelle103R.PNG",
    "img/test_set/pelle104R.PNG",
    "img/test_set/pelle105R.PNG"
]
test_img_color = [
    "img/test_set/pelle1001R_colors.PNG",
    "img/test_set/pelle1002R_colors.PNG",
    "img/test_set/pelle101R_colors.PNG",
    "img/test_set/pelle102R_colors.PNG",
    "img/test_set/pelle103R_colors.PNG",
    "img/test_set/pelle104R_colors.PNG",
    "img/test_set/pelle105R_colors.PNG"
]

# Riutilizziamo il modello creato prima
# ma calcoliamo il matthews coefficient medio per diverse immagini di test
print("Inizio calcolo dei test su immagini multiple.")
mc, cm = test_on_multiple_images(knn,
    test_img_orig,
    test_img_color,
    window_size=30,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy'],
    window_size_postprocessing=6)

print("Matthews correlation coefficient su tutte le img di test:")
print(mc) # matthews coeff

print("Confusion matrix su tutte le img di test:")
print(cm) # confusion matrix


def normalizza_cm(cm):
    """
    Funzione di supporto per normalizzare i valori
    di una confusion matrix come ottenuta dalla funzione
    test_on_multiple_images.
    """

    new_mat = []
    
    for riga in cm:
        new_mat.append([])
        tot_riga = riga.sum()
        for v in riga:
            if v==0:
                new_mat[-1].append(0)
            else:
                new_mat[-1].append(v/tot_riga)
    
    return np.array(new_mat)


print("Confusion matrix su tutte le img normalizzata:")
cm_norm = normalizza_cm(cm)
print(cm_norm)


# Mostriamo la matrice di confusione in un
# formato grafico più facile da interpretare
labels = [None, "vetrino","strato corneo","epidermide","derma","vasi"]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(cm_norm, interpolation='nearest', cmap="Blues")
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(cm_norm.round(2)):
    ax.text(j, i, z, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor="white"))

ax.set_xticklabels(labels, rotation="vertical")
ax.set_yticklabels(labels)
plt.show(block=True)
