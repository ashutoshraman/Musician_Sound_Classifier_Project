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

8. In learning curve, should we use all 250 samples or just the samples we trained on when optimizing our own stuff? (using just train stuff from train/test split)
9. Does he want train accuracy reported too, or just test accuracy? 
10. Precision, Recall, and F1 Score are mentioned but aren't mentioned in deliverables, so do we want them?
11. When you test the model in class, will you provide labels, so we can create a confusion matrix with our predictions? Or do you just know the answer and expect us to output the predicted names? Also, what will the labels be (numbers, spaces in names, underscores)?