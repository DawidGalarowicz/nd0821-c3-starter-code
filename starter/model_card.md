# Model Card

## Model Details
This model has been created by Dawid Galarowicz and it is based on LogisticRegression from sklearn library. All non-default hyperparameters have been specified in params.yaml.

## Intended Use
The aim of this model is predict if someone's annual income will exceed $50,000 given census data.

## Training & Evaluation Data
The data used in this project is the Census Income Data Set provided by UCI Machine Learning Repository. 

The training subset comprises 80% of this file, with the remainder being the testing set.

## Metrics
The final evaluation of the model has produced following metrics:
- 'precision': 0.7295796986518636
- 'recall': 0.5924018029620091
- 'fbeta': 0.6538734896943852

The model's performance has also been assessed on different levels of race, relationship and sex variables.

Race:
- White
    - 'precision': 0.7306015693112468
    - 'recall': 0.5951704545454546
    - 'fbeta': 0.6559686888454013

- Black
    - 'precision': 0.6666666666666666
    - 'recall': 0.5454545454545454
    - 'fbeta':  0.6

- Asian-Pac-Islander
    - 'precision': 0.7924528301886793
    - 'recall': 0.6268656716417911
    - 'fbeta': 0.7000000000000001

- Other
    - 'precision': 0.0
    - 'recall': 0.0
    - 'fbeta': 0.0

- Amer-Indian-Eskimo
    - 'precision': 0.8
    - 'recall': 0.5
    - 'fbeta': 0.6153846153846154

Relationship:

- Not-in-family
    - 'precision': 0.78125
    - 'recall': 0.2958579881656805
    - 'fbeta': 0.4291845493562232

- Unmarried
    - 'precision': 0.8181818181818182
    - 'recall': 0.20454545454545456
    - 'fbeta':   0.3272727272727273

- Husband
    - 'precision': 0.733847637415622
    - 'recall': 0.6482112436115843
    - 'fbeta': 0.6883763003165988

- Own-child
    - 'precision': 0.0
    - 'recall': 0.0
    - 'fbeta': 0.0

- Wife
    - 'precision': 0.6827586206896552
    - 'recall': 0.6346153846153846
    - 'fbeta': 0.6578073089700996

- Other-relative
    - 'precision': 0.3333333333333333
    - 'recall': 0.3333333333333333
    - 'fbeta': 0.3333333333333333

Sex:

- Male
    - 'precision': 0.7309458218549127
    - 'recall': 0.6057838660578386
    - 'fbeta':  0.6625052018310446

- Female
    - 'precision': 0.7209302325581395
    - 'recall': 0.5188284518828452
    - 'fbeta': 0.6034063260340633

## Ethical Considerations
Given the performance on the slices above, the performance of the model across sexes and races does not differ to an excessive extend. 

However, there is an considerable divergence between categories of relationship. For example, recall for unmarried respondents is lower than 21%, dragging F-1 score (denoted as fbeta) down to 33%. On the other hand, recall for husbands is 65%, with F-1 score being 69%.


## Caveats and Recommendations
- Given the performance on relationship slices, one might want to consider methods that influence the model training procedure to ensure more even metrics across categories.

- The model's precision is consistently higher than recall which means that the model is more suited towards making right positive decisions for a given person rather than catching positve instances across the dataset.