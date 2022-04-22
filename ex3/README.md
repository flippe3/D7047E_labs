# Results

### Comment on the differences between PCA and t-SNE
t-SNE disregards the actual global distances between the data points unlike PCA, but in turn manages to separate local clusters much more clearly which in the case of the MNIST dataset yeilds a good separation of the classes. However, PCA illustrates the actual mathematical distances between the input vectors accurately.

### Comment on the first model (virtually not trained at all) and the second one
The first PCA model is very messy and outside of number 1 class it is difficult to see any pattern. Compare that to the trained version of PCA you can see that clusters have formed and although the clusters are not perfect they are certainly better than the untrained model.

For t-SNE the first model already has made a very good cluster and you can clearly differentiate the classes. If you compare the first t-SNE to the second you can see that even though both do a pretty good job the trained model has clusters that are a bit more dense.
