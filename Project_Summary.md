# Pokémon Primary Type Prediction

## Project overview

This machine learning project predicts a Pokémon’s primary type from information such as its battle stats, generation, secondary type, and abilities. It explores how much predictive information comes from numeric stats alone and whether adding other attributes improves results.

## What the project does

- Loads and inspects a Pokémon dataset from a CSV file.
- Prepares numeric and text-based features, handles missing values, and creates a stratified training/test split.
- Trains a most-frequent baseline and two Random Forest classifiers: one using stats and generation, and another with secondary type and abilities.
- Evaluates the models using accuracy, balanced accuracy, macro F1, per-type classification metrics, and a normalized confusion matrix.
- Uses five-fold stratified cross-validation on the training data to examine how consistent model performance is across splits.
- Visualizes the distribution of Pokémon types and the feature importance of the stats-only model.

## Tools and techniques

Python, pandas, scikit-learn, Matplotlib, Random Forest classification, TF–IDF text features, one-hot encoding, stratified train/test splitting, and cross-validation.

## Why it is useful

The project demonstrates an end-to-end supervised learning workflow, from data preparation through model comparison and evaluation. Comparing a stats-only model with an expanded model helps show how different feature sets affect predictions, while class-balanced metrics provide a more informative view when some Pokémon types have fewer examples.

## CV-ready description

**Pokémon Primary Type Prediction — Machine Learning Project**  
Built a Python classification workflow to predict Pokémon primary types using Random Forest models. Compared stats-only and expanded feature sets with a majority-class baseline; applied imputation, one-hot encoding, and TF–IDF to mixed data; and assessed performance with stratified cross-validation, macro F1, balanced accuracy, per-class reports, and a confusion matrix.

## Results and limitations

The notebook reports measured results when its cells are run. The project compares feature sets but does not claim a particular accuracy in this summary. Model performance depends on the dataset and evaluation split, and results on this dataset do not guarantee performance on Pokémon outside it. Secondary type is included as an input in the expanded experiment, so that model answers a different question from the stats-only model: it predicts primary type when secondary-type information is already available.
