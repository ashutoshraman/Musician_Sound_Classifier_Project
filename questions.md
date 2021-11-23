MFCC file: 
Upload all the songs to the git?
Cannot retrieve the content, it shows 'With n_samples=0, test_size=0.5 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.'? 

Deploy file:
Do we need to add AUC score? If yes, what method should we use to get y_score (knn/svm/pca?)

1. How to do DFT?
2. Do you just want one ROC curve/ AUC of one vs all (choose one artist)? Or do you want every artist to get a one vs all plot?

3. How to check sampling rate of a song?
4. Can we delete data?
5. You say to allow user to specify number of MFCCs and PC variance. Does it need to be a function arg or can it just be a defined global variable in the script? Also, does this need to be done in deployment script too?**
6. Why are accuracies for best_score_ and score different?**
7. will deployed saved file load for different version of python/packages?**