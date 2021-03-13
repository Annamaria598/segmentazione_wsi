"""
In questo script sono testate tutte le funzioni
sviluppate per questo lavoro di tesi che non siano
semplicemente applicazioni di funzioni importate
da moduli esterni.
"""
import numpy as np
import pytest
from classificazione_knn import pixel_to_class, \
    colimage_to_classes, \
    class_to_pixel, \
    classes_to_colimage, \
    moda, \
    postprocessing_classes, \
    select_layers


# pixel_to_class
def test_pixel_to_class_vetrino():
    assert pixel_to_class(np.array([255, 255, 255, 255])) == 0

def test_pixel_to_class_strato_corneo():
    assert pixel_to_class(np.array([163, 167, 69, 255])) == 1

def test_pixel_to_class_derma():
    assert pixel_to_class(np.array([0, 255, 255, 255])) == 2

def test_pixel_to_class_epidermide():
    assert pixel_to_class(np.array([25, 55, 190, 255])) == 3

def test_pixel_to_class_vasi():
    assert pixel_to_class(np.array([255, 0, 255, 255])) == 4

def test_pixel_to_class_error():
    with pytest.raises(Exception):
        pixel_to_class(np.array([0, 2, 3, 255]))


# colimage_to_classes
"""
Le immagini con cui si lavora devono essere png con canale
alfa, altrimenti ogni pixel sarà rappresentato da soli 3 valori
anzichè da 4.
"""
def test_colimage_to_classes_1():
    class_matrix = [
        [0, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]
    ]
    colimage_to_classes("test_data/colors.png") == np.array(class_matrix)


def test_class_to_pixel():

    assert (np.array(class_to_pixel(0)) == np.array([255, 255, 255, 255])).all()
    assert (np.array(class_to_pixel(1)) == np.array([163, 167, 69, 255])).all()
    assert (np.array(class_to_pixel(2)) == np.array([0, 255, 255, 255])).all()
    assert (np.array(class_to_pixel(3)) == np.array([25, 55, 190, 255])).all()
    assert (np.array(class_to_pixel(4)) == np.array([255, 0, 255, 255])).all()


def test_classes_to_colimage():

    class_matrix = np.array(
        [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )

    pixel_matrix = np.array([
        [
            [255, 255, 255, 255],
            [163, 167, 69, 255],
            [163, 167, 69, 255],
            [163, 167, 69, 255]
        ],
        [
            [0, 255, 255, 255],
            [0, 255, 255, 255],
            [0, 255, 255, 255],
            [0, 255, 255, 255]
        ],
        [
            [25, 55, 190, 255],
            [25, 55, 190, 255],
            [25, 55, 190, 255],
            [25, 55, 190, 255]
        ],
        [
            [255, 0, 255, 255],
            [255, 0, 255, 255],
            [255, 0, 255, 255],
            [255, 0, 255, 255]
        ]
    ])

    assert (classes_to_colimage(class_matrix, (4,4,4)) == pixel_matrix).all()


def test_moda():

    assert moda([1, 2, 2, 2, 3, 3]) == 2
    # per insiemi senza una moda vogliamo
    # ottenere un valore qualunque tra quelli
    # presenti perchè non possiamo lasciare un
    # pixel senza valore durante l'applicazione del filtro
    assert moda([1, 2, 3, 4, 5]) in {1, 2, 3, 4, 5}


def test_postprocessing_classes():

    class_matrix = np.array(
        [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )

    """ Input con padding "reflect" mode
    0 0 1 1 1 1
    0 0 1 1 1 1
    2 2 2 2 2 2
    3 3 3 3 3 3
    4 4 4 4 4 4
    4 4 4 4 4 4
    """
    """ risultato atteso
        0 1 1 1
        2 2 1 1
        2 2 2 2
        4 4 4 4
    """

    class_matrix_post_3 = np.array([
        [0, 1, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 2, 2],
        [4, 4, 4, 4]
    ])
    print(postprocessing_classes(class_matrix, (4,4), window_size=3))
    assert (postprocessing_classes(class_matrix, (4,4), window_size=3) == class_matrix_post_3.ravel()).all()


def test_select_layers():

    fake_layers = {
        "R": np.array(["R"]),
        "G": np.array(["G"]),
        "B": np.array(["B"]),

        "R_mf": np.array(["R_mf"]),
        "G_mf": np.array(["G_mf"]),
        "B_mf": np.array(["B_mf"]),

        "R_edge": np.array(["R_edge"]),
        "G_edge": np.array(["G_edge"]),
        "B_edge": np.array(["B_edge"]),

        "R_variance": np.array(["R_variance"]),
        "G_variance": np.array(["G_variance"]),
        "B_variance": np.array(["B_variance"]),

        "L": np.array(["L"]),
        "a": np.array(["a"]),
        "b": np.array(["b"]),

        "L_mf": np.array(["L_mf"]),
        "a_mf": np.array(["a_mf"]),
        "b_mf": np.array(["b_mf"]),

        "L_edge": np.array(["L_edge"]),
        "a_edge": np.array(["a_edge"]),
        "b_edge": np.array(["b_edge"]),

        "L_variance": np.array(["L_variance"]),
        "a_variance": np.array(["a_variance"]),
        "b_variance": np.array(["b_variance"]),

        "H": np.array(["H"]),
        "E": np.array(["E"]),
        "D": np.array(["D"]),

        "H_mf": np.array(["H_mf"]),
        "E_mf": np.array(["E_mf"]),
        "D_mf": np.array(["D_mf"]),

        "H_edge": np.array(["H_edge"]),
        "E_edge": np.array(["E_edge"]),
        "D_edge": np.array(["D_edge"]),

        "H_variance": np.array(["H_variance"]),
        "E_variance": np.array(["E_variance"]),
        "D_variance": np.array(["D_variance"]),

        "h": np.array(["h"]),
        "s": np.array(["s"]),
        "v": np.array(["v"]),

        "h_mf": np.array(["h_mf"]),
        "s_mf": np.array(["s_mf"]),
        "v_mf": np.array(["v_mf"]),

        "h_edge": np.array(["h_edge"]),
        "s_edge": np.array(["s_edge"]),
        "v_edge": np.array(["v_edge"]),

        "h_variance": np.array(["h_variance"]),
        "s_variance": np.array(["s_variance"]),
        "v_variance": np.array(["v_variance"]),

        "entropy": np.array(["entropy"])
    }

    assert (select_layers(fake_layers, feature_names=["R", "G", "B"]) == np.array([np.array(["R"]), np.array(["G"]), np.array(["B"])]).T).all()
    assert (select_layers(fake_layers, feature_names=[]) == np.array(list(fake_layers.values())).T).all()