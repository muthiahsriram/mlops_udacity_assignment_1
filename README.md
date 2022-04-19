# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- Project on clean coding prinicples for production 

## Files and data description
 - data - Contains the data
 - enviroment - contains the virtual environment
 - images/eda - contains the eda results
 - images/results - contains the model summary
 - logs/run_log.log - Log of code running
 - logs/test_churn_library.log - Log of tests
 - models - model pickles
 - churn_library.py - contains the library of function for loading the data, cleaning, feature engineering, model training, scoring and results reporting
 - churn_script_logging_and_tests.py - contains the tests for functions implemented in churn_library.py
 - requirement.txt - required packages

## Running Files

 - Tests can be run using the following command:

        ipython churn_script_logging_and_tests_solution.py

 - All functions required for model training and scoring are present in churn_library.py like below

        data_frame = import_data(r"./data/bank_data.csv")
        perform_eda(data_frame)
        data_frame = encoder_helper(data_frame, constants.cat_columns, constants.dv)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, constants.dv)
        train_models(x_train, x_test, y_train, y_test)

# Dependencies
- This project was done in my local will be available as github link
- Python 3.9 was used
- please finf requirement.txt for installing libraries




