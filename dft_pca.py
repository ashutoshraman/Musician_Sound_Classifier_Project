from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def pca_choice(variance):
    pca = PCA(n_components=variance, svd_solver='full')
    return pca

def graph_eig_pca():
    scaler = StandardScaler()
    pca = pca_choice(.75)
    print("Variance captured: {:f}".format(sum(pca.explained_variance_ratio_)))
    print("Number of components kept: {:d}".format(pca.n_components_))
    plt.plot(pca.singular_values_)
    plt.show()


# dft--> np.real(dft)-->scale-->pca_choice(.75)-->pca_choice.fit_transform(dft)