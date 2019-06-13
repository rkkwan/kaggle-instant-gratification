# Instant Gratification
#### A synchronous Kernels-only competition
https://www.kaggle.com/c/instant-gratification/kernels

`data/` not included in repo. Dataset can be acquired from the competition's website.

### Predictor
[Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)
`from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis`

### Metric
ROC-AUC Score.  

QDA: **0.96541**
QDA with Pseudo Labeling: **0.97027**

### Techniques
#### Segmentation
Column `wheezy-copper-turtle-magic` appears to be a categorical variable with values 0 to 511. Dataset is uniformly distributed 512 rows for each value. A separate model will be trained for each value in `wheezy-copper-turtle-magic`.  

Psuedocode:
```
all_preds = empty list
for i in 'wheezy-copper-turtle-magic':
    train_subset = train['wheezy-copper-turtle-magic' == i]
    test_subset = test['wheezy-copper-turtle-magic' == i]
    model.fit(train_subset)
    preds = model.predict_proba(test_subset)
    append preds to all_preds
```

#### Pseudo Labeling
Add confident predicted test data to your training data. Confident predictions are predicted probabilities <= 0.01 or >= 0.99. Parameters may be tuned.

1. Build a model using training data
2. Predict labels for an unseen test dataset
3. Add confident predicted test observations to training data
4. Build a new model using combined data
5. Use your new model to predict the test data.

Pseudocode:
```
model.fit(train)
preds = model.predict_proba(test)

confident_preds = preds[preds >= 0.99 or preds <= 0.01]
pseudo_labeled_train = train + confident_preds

model.fit(pseudo_labeled_train)
new_preds = model.predict_proba(test)
```

### To-Do:
#### Flipping
Randomly flip a portion (2.5%) of incorrectly predicted 0's and 1's.
