# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model yielded by this project is a Random Forest Classifier trained on the Census dataset.

## Intended Use

The model should be used to predict salary of people based on different attributes (numertical and categorical).

## Training Data

The original data (provided here: https://archive.ics.uci.edu/ml/datasets/census+income) is splitted into two chuncks (80% train)

## Evaluation Data

The original data is splitted into two chuncks (20% test)

## Metrics
_Please include the metrics used and your model's performance on those metrics._

To get the best estimator we used the accuracy as a metric:
We found:
Train Acc: 91.01%
Test Acc: 91.5 %

## Ethical Considerations

Given the slice study performed on the categorical features, 
We found that the performance on the race feature vary slightly between different ethnicities.

However, a dedicated study on the faireness and unbiasedness of the model is highly recommended before use.


## Caveats and Recommendations
Given the different categories, we recommend a dedicated study to understand the impact of variation on the final results.

We, moreoever, recommend trying other machine learning model to produce a more robust and stable solution