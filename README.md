# Lab 5 — Titanic ML Pipeline

## Overview

This lab builds a complete machine learning workflow for predicting passenger survival on the Titanic dataset. The focus is on data exploration, leakage-free preprocessing, feature engineering, model comparison, and hyperparameter tuning.

The project is split into two stages:

* **Day 1:** Data exploration, quality checks, cleaning decisions, and train/test splitting
* **Day 2:** Preprocessing pipelines, cross-validation, feature engineering, model comparison, and tuning

## Project Structure

```text
lab5_ml_pipeline/
├── data.py
├── pipeline.py
├── day1_exploration_starter.ipynb
├── day2_pipeline_modeling_starter.ipynb
├── requirements.txt
└── README.md
```

## Day 1 — Exploration and Cleaning

The dataset was profiled for:

* Missing values
* Data types
* Class balance
* Outliers and impossible values
* Placeholder values
* Missingness versus survival
* Potential target leakage

Important findings included substantial missingness in `Cabin` and `Age`, as well as a smaller amount of missingness in `Embarked`. Missingness itself was also investigated because it can contain predictive information.

The train/test split was performed using an 80/20 split with a fixed random seed and stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)
```

This ensures that the survival class distribution remains similar between the training and test sets.

## Day 2 — Pipeline and Preprocessing

A leakage-free preprocessing pipeline was implemented using `ColumnTransformer` and `Pipeline`.

### Numerical features

Numerical variables are processed using:

1. Median imputation
2. Standard scaling

### Categorical features

Categorical variables are processed using:

1. Most-frequent imputation
2. One-hot encoding

`OneHotEncoder` uses `handle_unknown="ignore"` so that unseen categories do not cause prediction failures.

`Pclass` was manually treated as categorical rather than numerical because its values represent passenger classes rather than a continuous quantity.

## Baseline Model

A cross-validated baseline was evaluated using **F1 score** rather than relying only on accuracy, since the survival classes are imbalanced.

The baseline achieved approximately:

**F1 = 0.73 ± 0.04**

The standard deviation is reported because fold-to-fold variation is meaningful on a dataset of this size.

## Feature Engineering

Several features were investigated:

* `family_size`
* `is_alone`
* `has_cabin`
* `Title`

The feature experiments were evaluated using cross-validation rather than relying on the final test set.

### Feature Results

| Feature Set             | CV F1 |
| ----------------------- | ----: |
| Basic features          |  0.71 |
| Family features         |  0.71 |
| Cabin feature           |  0.71 |
| Title                   |  0.75 |
| All engineered features |  0.75 |

The **Title** feature provided the clearest improvement. The family and cabin features did not improve the cross-validated score in these experiments, and combining all engineered features did not improve upon Title alone.

## Model Comparison

Three different model types were evaluated using the same preprocessing pipeline:

| Model               | Mean F1 | Std. Dev. |
| ------------------- | ------: | --------: |
| Logistic Regression |    0.75 |      0.03 |
| Random Forest       |    0.73 |      0.03 |
| Gradient Boosting   |    0.73 |      0.05 |

Although Logistic Regression had the highest mean F1, the differences between the models are relatively small compared with the fold-to-fold variation. Therefore, the results do not provide especially strong evidence that one model is substantially better than the others.

## Hyperparameter Tuning

The best-performing model from the comparison was tuned using `GridSearchCV` on the training data only.

The Logistic Regression regularization parameter `C` was tuned using the pipeline parameter name:

```python
model__C
```

The best value found was:

```text
C = 10
```

with a cross-validated F1 score of approximately **0.76**.

## Final Test Evaluation

After model selection and tuning were completed, the held-out test set was evaluated once.

**Final Test F1: 0.758**

The test result is close to the cross-validation estimate, suggesting that the model's performance generalizes reasonably well to the held-out data.

The test set was not used to make further modeling or tuning decisions.

## Key Takeaways

1. **Pipeline-based preprocessing prevents data leakage.** Imputation, scaling, and encoding are fitted within the cross-validation training folds rather than on the entire dataset.

2. **Missingness can contain useful information.** In particular, Cabin and Age missingness showed differences in survival rates, making missingness worth investigating rather than automatically discarding.

3. **Feature engineering produced mixed results.** The Title feature improved performance, while family and cabin features did not provide a measurable improvement in these experiments.

4. **Model differences were relatively small.** Logistic Regression had the highest mean CV F1, but the difference was comparable to the variability between folds.

5. **Hyperparameter tuning provided only a modest improvement.** Increasing the Logistic Regression `C` parameter improved the CV result slightly, but the overall performance remained similar.

6. **The final test result was consistent with cross-validation.** The final F1 score of approximately 0.758 was close to the tuned CV score of approximately 0.76.

## Possible Next Steps

With additional time, I would investigate:

* Interactions between `Pclass` and family-related features
* Additional information that could be extracted from `Name` and `Ticket`
* Whether more detailed title categories provide additional signal
* Alternative models and hyperparameter ranges
* More robust validation approaches to better quantify uncertainty on the relatively small Titanic dataset

## Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

* scikit-learn
* pandas
* numpy
* matplotlib

## Reproducibility

The train/test split uses a fixed `random_state=0` and `stratify=y`. Preprocessing and model fitting are performed through scikit-learn pipelines to ensure that transformations are learned only from the appropriate training data.
