# Gradient-Descent and PCA
This is a custom gradient descent class that have l1, l2, and l-infinity regularization and the ability to perform non-negative matrix factorization for better and more interpretable result.
It also has the ability to limit turning angle for each gradient step as compared to the previous one to force not steping over the minima and follow a cleaner pass, espeecially useful when reaching from a certain side is desired.
It is also capable to perform stochastic gradient descent with the ability to use previous gradients to shrink the zone of confusion and achieve better result while having the speed advantage of being stochastic.
Also include is a custom PCA class that uses the custom gradient class to produce principal components that are better and more interpretable compared to traditional SVD.
