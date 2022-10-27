from sklearn.datasets import load_digits   #veri seti datasets'den yüklendi
digits = load_digits() 		
digits.data.shape 	
              
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components = 4) 			# boyut sayısı dörde indirgendi.
digits_pca=pca.fit_transform(digits.data)
digits_pca.shape
print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

pca = PCA(n_components = 2) 			# boyut sayısı ikiye indirgendi.
digits_pca2 = pca.fit_transform(digits.data)
print(digits_pca2.shape)
print(pca.components_) 
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_)) #kümülatif toplam
                               
import matplotlib.pyplot as plt

plt.scatter(digits_pca2[:, 0], digits_pca2[:, 1])

pca = PCA().fit(digits.data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(1.0, c="y")
plt.axhline(0.90, c="r")
plt.axhline(0.80, c="g")
