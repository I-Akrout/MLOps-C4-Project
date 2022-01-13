# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model yielded by this project is a scikit-learn Random Forest Classifier trained on the Census dataset.

## Intended Use

The model should be used to predict salary of people based on different attributes (numertical and categorical).

## Training Data

The original data (provided here: https://archive.ics.uci.edu/ml/datasets/census+income) is splitted into two chuncks (80% train, 20 % test)

## Evaluation Data

The original data is splitted into two chuncks (20% test)

## Metrics
To mesure the performance of our model, we used 3 different metrics:

    - Precision
    - Recall
    - Fbeta  


On the train set the model yielded the following values:
- Precision: 88.69 %
- Recall: 72.31 %
- Fbeta: 79.60 %

Using the evaluation data, our model gave the following results:
- Precision: 89.35 %
- Recall: 72.07 %
- Fbeta: 79.79 %

A more detailed results are provided in the `starter/slice_output.txt` file. 
## Ethical Considerations

Given the slice study performed on the categorical features, 
We found that the performance on the race feature vary slightly between different ethnicities.

However, a dedicated study on the faireness and unbiasedness of the model is highly recommended before use.


## Caveats and Recommendations
Given the different categories, we recommend a dedicated study to understand the impact of variation on the final results.

We, moreoever, recommend trying other machine learning model to produce a more robust and stable solution