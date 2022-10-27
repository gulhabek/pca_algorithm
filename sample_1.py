import numpy as np 
rng=np.random.RandomState() 	
rng_1=rng.rand(2,2)
rng_2=rng.rand(2,200)
data=np.dot(rng_1, rng_2).T	 #rng değişkenini kullanarak 2 dizi üretildi ve birbiri ile çarpıldı
                                            #Ardından transpozu alındı(T) ve veri seti oluştu.
data.shape 

import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1])	 #veri setindeki iki değişkenin ilişki grafiği çizdirildi.

from sklearn.decomposition import PCA
pca=PCA(n_components=2) #boyut 2 olarak girildi
pca.fit(data) 	#ardından fit metodu çağrılarak model kuruldu.

print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

pca=PCA(n_components=1) #boyut 1'e indirgendi.
pca.fit(data) 	
data_pca=pca.transform(data)	 #dönüştürme için transform metodu kullanılır.

print(data.shape) 
print(data_pca.shape)

#grafik çizdirme
data_yeni=pca.inverse_transform(data_pca) 	#azaltılan verinin ters dönüşümü bulundu.
plt.scatter(data[:,0],data[:,1], alpha=0.3)	
plt.scatter(data_yeni[:,0],data_yeni[:,1], alpha=0.8)
