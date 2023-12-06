#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optunity
import tqdm

global pbar 
pbar = None


def main():   # Set the search space
    global pbar
    search_space = {
        'criterion': [0, 1],
        'max_features': [0, 64],
        'min_samples_leaf': [1, 10],
        'min_samples_split': [2, 10],
        'n_estimators': [10, 10000],
        'max_depth': [2, 50],
        'max_samples': [0.1, 1.0]
    }
    # Set up tqdm progress bar
    num_evals = 100
    pbar = tqdm.tqdm(total=3*num_evals, desc="Optimizing Hyperparameters", position=0, leave=True)


    # Perform PSO with Optunity
    optimal_pars, _, _ = optunity.maximize(model_accuracy, num_evals=num_evals, solver_name='particle swarm', **search_space)
    pbar.close()  # Close the progress bar

    # Train the model with optimal parameters
    optimal_pars['criterion'] = 'gini' if optimal_pars['criterion'] > 0.5 else 'entropy'
    optimal_pars['max_features'] = 'sqrt' if optimal_pars['max_features'] > 0.5 else 'log2'
    optimal_pars['min_samples_leaf'] = int(optimal_pars['min_samples_leaf'])
    optimal_pars['min_samples_split'] = int(optimal_pars['min_samples_split'])
    optimal_pars['n_estimators'] = int(optimal_pars['n_estimators'])
    optimal_pars['max_depth'] = int(optimal_pars['max_depth'])

    # Create the RandomForestClassifier object with optimal parameters
    clf_best = RandomForestClassifier(criterion=optimal_pars['criterion'],
                                    max_features=optimal_pars['max_features'],
                                    min_samples_leaf=optimal_pars['min_samples_leaf'],
                                    min_samples_split=optimal_pars['min_samples_split'],
                                    n_estimators=optimal_pars['n_estimators'],
                                    max_depth=optimal_pars['max_depth'],
                                    max_samples=optimal_pars['max_samples'],
                                    n_jobs=-1)

    # Fit the model with training data
    clf_best.fit(X_train, y_train)

    # Predict using the best model
    y_pred = clf_best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Print the best parameters
    print("Best parameters:", optimal_pars)


if __name__ == "__main__":

    data = pd.read_csv('Round2_Core_Niche_Model_Roary_Input_Data_5k_v1.csv')


    # Create dataframes that retain all columns but exclude the other niche columns
    df_niche_1 = data.drop(columns=['Niche_2', 'Niche_3', 'Niche_4', 'Niche_5', 'Niche_6']).rename(columns={'Niche_1': 'Niche'})
    df_niche_2 = data.drop(columns=['Niche_1', 'Niche_3', 'Niche_4', 'Niche_5', 'Niche_6']).rename(columns={'Niche_2': 'Niche'})
    df_niche_3 = data.drop(columns=['Niche_1', 'Niche_2', 'Niche_4', 'Niche_5', 'Niche_6']).rename(columns={'Niche_3': 'Niche'})
    df_niche_4 = data.drop(columns=['Niche_1', 'Niche_2', 'Niche_3', 'Niche_5', 'Niche_6']).rename(columns={'Niche_4': 'Niche'})
    df_niche_5 = data.drop(columns=['Niche_1', 'Niche_2', 'Niche_3', 'Niche_4', 'Niche_6']).rename(columns={'Niche_5': 'Niche'})
    df_niche_6 = data.drop(columns=['Niche_1', 'Niche_2', 'Niche_3', 'Niche_4', 'Niche_5']).rename(columns={'Niche_6': 'Niche'})

    df_niche_4 = df_niche_4[df_niche_4['Niche'] != 'Epidemic']

    # Assuming df_niche_4 is already loaded
    df_niche_4 = df_niche_4.drop(['Genome_ID'], axis=1)
    top_20_df = df_niche_4.copy()
    original_niches = top_20_df['Niche'].copy()

    # Encoding the labels
    le = LabelEncoder()
    top_20_df['Niche'] = le.fit_transform(top_20_df['Niche'])
    niche_mapping = dict(zip(top_20_df['Niche'], original_niches))

    # Cleaning column names
    def clean_column_names(df):
        cleaned_columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in df.columns]
        seen = set()
        for i, col in enumerate(cleaned_columns):
            original_col = col
            j = 0
            while col in seen:
                j += 1
                col = original_col + "_" + str(j)
            cleaned_columns[i] = col
            seen.add(col)
        return cleaned_columns

    cleaned_columns = clean_column_names(top_20_df)
    top_20_df.columns = cleaned_columns

    X = top_20_df.drop('Niche', axis=1)
    y = top_20_df['Niche']
    X.columns = clean_column_names(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Define a cost function for PSO to optimize
    @optunity.cross_validated(x=X_train.values, y=y_train.values, num_folds=3)
    def model_accuracy(x_train, y_train, x_test, y_test, criterion, max_features, min_samples_leaf, min_samples_split, n_estimators, max_depth, max_samples):
        # Convert continuous values from PSO to discrete hyperparameters
        global pbar
        if pbar is not None:
            pbar.update(1)

        criterion = 'gini' if criterion > 0.5 else 'entropy'
        max_features = 'sqrt' if max_features > 0.5 else 'log2'
        min_samples_leaf = int(min_samples_leaf)
        min_samples_split = int(min_samples_split)
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        # Create and train the model
        model = RandomForestClassifier(criterion=criterion, max_features=max_features, 
                                    min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                    n_estimators=n_estimators, max_depth=max_depth,
                                    max_samples=max_samples, n_jobs=-1)
        model.fit(x_train, y_train)

        # Predict and calculate accuracy
        predictions = model.predict(x_test)
        return accuracy_score(y_test, predictions)
    main()






