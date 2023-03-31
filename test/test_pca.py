import matplotlib.pyplot as plt
import numpy as np

from ias import Emnist, Iris, PCA


def test_n_composantes():
    data = np.array(
        (np.arange(5) ** 2, np.arange(5) + 1, np.arange(5) + 2, np.arange(5) + 3,
         np.arange(5) ** 2 + 4)).T

    pca1 = PCA(n_composantes=1)
    pca2 = PCA(n_composantes=2)

    pca1.fit(data)
    pca2.fit(data)

    comp1 = pca1.compress(data)
    comp2 = pca2.compress(data)

    decomp1 = pca1.decompress(comp1)
    decomp2 = pca2.decompress(comp2)

    print("Data :")
    print(data)
    print("Compressed 1 :")
    print(decomp1.round(decimals=3))

    print("Compressed 2 :")
    print(decomp2.round(decimals=3))

    print("Deltas : ")
    print("1 composantes : ", np.sum((data - decomp1) ** 2).round(decimals=3))
    print("2 composantes : ", np.sum((data - decomp2) ** 2).round(decimals=3))


def test_with_iris():
    data = Iris.features

    pca1 = PCA(n_composantes=0.9)
    pca2 = PCA(n_composantes=0.95)
    pca3 = PCA(n_composantes=1.0)

    pca1.fit(data)
    pca2.fit(data)
    pca3.fit(data)

    comp1 = pca1.compress(data)
    comp2 = pca2.compress(data)
    comp3 = pca3.compress(data)

    decomp1 = pca1.decompress(comp1)
    decomp2 = pca2.decompress(comp2)
    decomp3 = pca3.decompress(comp3)

    print("Deltas : ")
    print("90%  : ", np.sum((data - decomp1) ** 2).round(decimals=3))
    print("95%  : ", np.sum((data - decomp2) ** 2).round(decimals=3))
    print("100% : ", np.sum((data - decomp3) ** 2).round(decimals=3))


def test_with_emnist():
    data = Emnist.features[:1000]
    var_1 = 0.8
    var_2 = 0.95

    pca1 = PCA(n_composantes=var_1)
    pca2 = PCA(n_composantes=var_2)

    pca1.fit(data)
    pca2.fit(data)

    comp1 = pca1.compress(data)
    comp2 = pca2.compress(data)

    decomp1 = pca1.decompress(comp1)
    decomp2 = pca2.decompress(comp2)

    print("Composantes Numbers")
    print(f"Input : {len(data[0])}")
    print(f"{int(var_1 * 100):3}%  : {pca1.output_dimension}")
    print(f"{int(var_2 * 100):3}%  : {pca2.output_dimension}")

    print("Deltas : ")
    print(f"{int(var_1 * 100):3}%  : ", np.sum((data - decomp1) ** 2).round(decimals=3))
    print(f"{int(var_2 * 100):3}%  : ", np.sum((data - decomp2) ** 2).round(decimals=3))

    img_nb = 5
    plt.figure()

    f, ax = plt.subplots(1, 3)

    ax[0].imshow(data[img_nb].reshape(28, 28).T)
    ax[0].set_title("Original")
    ax[1].imshow(decomp1[img_nb].reshape(28, 28).T)
    ax[1].set_title(f"{int(var_1 * 100)}% of variance")
    ax[2].imshow(decomp2[img_nb].reshape(28, 28).T)
    ax[2].set_title(f"{int(var_2 * 100)}% of variance")

    print("Label : ", Emnist.class_mapping[Emnist.labels[img_nb]])
    plt.show()
