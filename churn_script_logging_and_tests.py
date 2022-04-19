"""
Module to test the churn library functions.

All the test results will be logged in ./logs/test_log.log
"""


# from asyncio import constants
import os
import logging
import churn_library as cl
import constants


# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler(constants.log_file_test, mode='w')
formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(module)s :%(funcName)s:%(lineno)d :%(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        logger.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logger.error(
            "FAILED: Testing import_eda: The file wasn't found: %s", err)
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "FAILED: Testing import_data: The file doesn't appear to have rows and columns: %s", err)
        raise err


def test_eda():
    '''
    testing perform eda function
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        cl.perform_eda(data_frame)
        file_names = [f for f in os.listdir(
            constants.eda_folder) if os.path.isfile(os.path.join(constants.eda_folder, f))]
        assert len(file_names) > 0
        logger.info("SUCCESS: Testing perform EDA")
    except AssertionError as err:
        logger.error(
            "FAILED: Testing perform_eda - Images not found: %s", err)
        raise err
    except Exception as err:
        logger.error(
            "FAILED: Testing perform_eda - Images not found: %s", err)
        raise err


def test_encoder_helper():
    '''
    testing encoder helper fn
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        data_frame = cl.encoder_helper(
            data_frame, constants.cat_columns, constants.dv)
        new_cat_columns = [
            f"{col}_{constants.dv}" for col in constants.cat_columns]
        for i in new_cat_columns:
            assert i in data_frame.columns
        logger.info("SUCCESS: Testing encoder_helper")

    except AssertionError as err:
        logger.error("FAILED: Testing encoder_helper: %s", err)
        raise err

    except Exception as err:
        logger.error("FAILED: Testing encoder_helper: %s", err)
        raise err


def test_perform_feature_engineering():
    '''
    testing perform_feature_engineering fn

    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        data_frame = cl.encoder_helper(
            data_frame, constants.cat_columns, constants.dv)
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            data_frame, constants.dv)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[1] > 0
        logger.info("SUCCESS: Testing perform_feature_engineering")
    except AssertionError as err:
        logger.error(
            "FAILED: Testing perform_feature_engineering: %s", err)
        raise err
    except Exception as err:
        logger.error(
            "FAILED: Testing perform_feature_engineering: %s", err)
        raise err


def test_train_models():
    '''testing train_models fn '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        data_frame = cl.encoder_helper(
            data_frame, constants.cat_columns, constants.dv)
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            data_frame, constants.dv)
        cl.train_models(X_train, X_test, y_train, y_test)
        model_files = [f for f in os.listdir(
            constants.model_folder) if os.path.isfile(os.path.join(constants.model_folder, f))]
        assert len(model_files) > 0
        logger.info("SUCCESS: Testing train_models")

    except AssertionError as err:
        logger.error("FAILED: Testing train_models: %s", err)
        raise err

    except Exception as err:
        logger.error("FAILED: Testing train_models: %s", err)
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
