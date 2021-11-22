# Musician_Sound_Classifier_Project

Things to Do:
1. MFCC calculation and conversion (check Lecture 12 code-AudioFeatureExtractor)- Thomas <br />
        Argument in function for user to specify number of coefficients
3. DFT and PCA (check lecture for DFT code)- Barry <br />
        make sure to include argument in function to allow user to specify amount of variance (i.e. 75%, 90%, 98%). <br />
        Only use amplitude of DFT
3. SVM setup -Ashu done
4. KNN setup- Ashu done
5. Logistic Regression Setup- Ashu done
6. Grid Search functionality and nested CV functionality for each classifier model selection- Ashu done
7. Confusion Matrix- Ashu done
8. AUC- Thomas
9. ROC- Thomas
10. Accuracy, Recall, Precision, F1 Score- Ashu done
11. Deployment code to allow user to classify with a pretrained model on same file structure- Thomas <br />
 -see https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
12. Paper explanation with Figures for: <br />
         a. data plot using first 2 MFCC of each song, each artist gets different color, and commentary on how well data separates <br />
         b. plot of eigenvalues from PCA in descending order, description of number of PC's needed to capture 75, 90, 98% of data, level of compression <br />
         c. description of process for selecting best performing classifier (did you use gridsearch and CV) <br />
         d. plots for ROC, AUC, accuracy, confusion matrix for each top model on each set of data (so 6 sets of these 4 graphs), comment on results <br />
         e. plot of learning curves for each of the 6 cases, comment on generalization, bias, variance, etc. <br />
         e. References (a couple)
14. Presentation <br />
         a. 10 minutes long
