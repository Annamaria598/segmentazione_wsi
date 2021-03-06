from funzioni import image_to_data,\
    colimage_to_classes,\
    local_entropy,\
    classes_to_colimage,\
    postprocessing_classes,\
    select_layers,\
    image_segmentation,\
    train_on_multi,\
    test_on_multiple_images
import numpy as np
import imageio
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# Creazione del modello allenandolo su più immagini
print("Inizio training random forest...")
rf, score, conf_matrix = train_on_multi(
    ["img/pelle303R.PNG", "img/pelle304R.PNG", "img/pelle601R.PNG"],
    ["img/pelle303R_colors.PNG", "img/pelle304R_colors.PNG", "img/pelle601R_colors.PNG"],
    ["img/pelle301R.PNG", "img/pelle302R.PNG", "img/pelle305R.PNG"],
    ["img/pelle301R_colors.PNG", "img/pelle302R_colors.PNG", "img/pelle305R_colors.PNG"],
    n_estimators=100,
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)

print("Terminato training random forest.")
print("Score: {}".format(score))

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 1001R")
image_segmentation(
    rf,
    "img/test_set/pelle1001R.PNG",
    "img/test_set/pelle1001R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 1002R")
image_segmentation(
    rf,
    "img/test_set/pelle1002R.PNG",
    "img/test_set/pelle1002R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 101R")
image_segmentation(
    rf,
    "img/test_set/pelle101R.PNG",
    "img/test_set/pelle101R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 102R")
image_segmentation(
    rf,
    "img/test_set/pelle102R.PNG",
    "img/test_set/pelle102R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 103R")
image_segmentation(
    rf,
    "img/test_set/pelle103R.PNG",
    "img/test_set/pelle103R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 104R")
image_segmentation(
    rf,
    "img/test_set/pelle104R.PNG",
    "img/test_set/pelle104R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Generazione dei grafici su una sola immagine di test
print("Generazione grafici per img 105R")
image_segmentation(
    rf,
    "img/test_set/pelle105R.PNG",
    "img/test_set/pelle105R_colors.PNG",
    window_size=30,
    window_size_postprocessing=6,
    feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy']
)
plt.show(block=True)

# Riutilizziamo il modello creato prima
# ma calcoliamo il matthews coefficient medio per diverse immagini di test
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

print("Inizio calcolo score e confusion matrix per tutte le immagini...")
mc, cm = test_on_multiple_images(rf,
                        test_img_orig,
                        test_img_color,
                        window_size=30,
                        feature_names=['R_variance', 'G_variance', 'B_variance', 'H_mf', 'E_mf', 'D_mf', 'entropy'],
                        window_size_postprocessing=3)

print("Score su tutte le immagini: " + str(mc))

def normalizza_cm(cm):
    """Normalizzazione della matrice di confusione come ottenuta
    dalla funzione test_on_multiple_images."""
    
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


# Mostrare la matrice di confusione
cm_norm = normalizza_cm(cm)
labels = [None, "vetrino","strato corneo","epidermide", "derma", "vasi"]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(cm_norm, interpolation='nearest', cmap="Blues")
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(cm_norm.round(2)):
    ax.text(j, i, z, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor="white"))

ax.set_xticklabels(labels, rotation="vertical")
ax.set_yticklabels(labels)

print("Matrice di confusione normalizzata su tutte le immagini di test.")
plt.show(block=True)