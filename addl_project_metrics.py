import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import numpy as np


def plot_mfcc(dataframe): #using all songs and no train test split yet
    plt.figure()
    sns.scatterplot(x=dataframe.iloc[:, 1], y=dataframe.iloc[:, 2], hue=dataframe['Target'], style=dataframe['Target'], legend='auto')
    plt.title('First 2 MFCCs for Audio Clips of Artists')
    plt.xlabel('MFCC 1')
    plt.ylabel('MFCC 2')
    plt.show()


def get_eigen_graph(dataframe, variance): #using all songs and no train test split yet
    plt.figure()
    scaler = StandardScaler()
    pca = PCA(n_components=variance, svd_solver='full')

    X_std = scaler.fit_transform(dataframe.iloc[:, 1:])
    X_pca = pca.fit_transform(X_std)

    print("Variance captured: {:f}".format(sum(pca.explained_variance_ratio_)))
    print("Number of components kept: {:d}".format(pca.n_components_))
    print(X_pca.shape)
    plt.plot(pca.singular_values_)
    plt.xlabel('Number of PCs')
    plt.ylabel('Eigenvalue for Respective PC')
    plt.title('Eigenvalues for Corresponding Number of Principal Components Needed to Explain Variance')
    plt.show()
    return

def learning_curve_graph(fitted_pipeline, X_train, Y_train):
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(fitted_pipeline, X_train, Y_train,
                                            train_sizes=np.linspace(.1,1.0,10), cv=5, n_jobs=1)

    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean+train_std,train_mean-train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean+test_std,test_mean-test_std, alpha=0.15, color='green')

    plt.show()