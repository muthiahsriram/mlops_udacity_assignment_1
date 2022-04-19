"""
Churn library functions
"""


# import libraries
import logging
import os
import shutil
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import constants

sns.set()

logging.basicConfig(
    filename=constants.log_file_run,
    level=logging.INFO,
    format='%(name)s:%(asctime)s:%(levelname)s:%(message)s',
    filemode='w')

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        logging.info("Successfully imported %s", pth)
        return data_frame
    except FileNotFoundError:
        logging.error("File not found at %s", pth)
        return None
    except Exception as err:
        logging.error("Error importing %s: %s", pth, err)
        return None


def create_histogram(series_s, file_name):
    """"""
    '''
    create a histogram and save the image to the images folder
    input:
            data_frame: pandas dataframe series
            file_name: string of the file name to save the figure

    output:
            None
    '''
    try:
        plt.figure(figsize=(20, 10))
        series_s.hist()
        plt.savefig(file_name)
    except Exception as err:
        logging.error("Error creating histogram for %s: %s", file_name, err)


def move_files(source_dir, target_dir):
    '''
    move files from source_dir to target_dir
    input:
            source directory and taget directory

    output:
            None
    '''

    file_names = [f for f in os.listdir(
        source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    for file_name in file_names:
        shutil.copy(os.path.join(source_dir, file_name), target_dir)
        os.remove(os.path.join(source_dir, file_name))


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    move_files(constants.eda_folder, constants.eda_folder_archive)

    # Churn histogram
    try:
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        create_histogram(
            data_frame['Churn'], f"{constants.eda_folder}churn_histogram.png")
        logging.info("Successfully created churn histogram")
    except Exception as err:
        logging.error("Error creating churn histogram %s", err)

    # Customer age histogram
    try:
        create_histogram(data_frame['Customer_Age'],
                         f"{constants.eda_folder}customer_age.png")
        plt.figure(figsize=(20, 10))
        logging.info("Successfully created customer age histogram")
    except Exception as err:
        logging.error("Error creating customer age histogram %s", err)

    # Marital age distribution
    try:
        data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(f"{constants.eda_folder}marital_status.png")
        logging.info("Successfully created marital status histogram")
    except Exception as err:
        logging.error("Error creating marital status histogram %s", err)

    # Total_Trans_Ct_fig distribution
    try:
        plt.figure(figsize=(20, 10))
        Total_Trans_Ct_fig = sns.histplot(
            data_frame['Total_Trans_Ct'], stat='density', kde=True).get_figure()
        Total_Trans_Ct_fig.savefig(f"{constants.eda_folder}Total_Trans_Ct.png")
        logging.info("Successfully created marital status histogram")
    except Exception as err:
        logging.error("Error creating marital status histogram %s", err)

    # correlation heat map
    try:
        plt.figure(figsize=(20, 10))
        sns.heatmap(data_frame.corr(), annot=False,
                    cmap='Dark2_r', linewidths=2)
        plt.savefig(f"{constants.eda_folder}/heat_map.png")
        logging.info("Successfully created heat map")
    except Exception as err:
        logging.error("Error creating heat map: %s", err)


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]
    output:
            data_frame: pandas dataframe with new columns for
    '''
    logging.info("Starting encoder helper")
    existing_cols = set(data_frame.columns)
    for cat_col in category_lst:
        return_lst = []
        groups = data_frame.groupby(cat_col).mean()[response]
        for val in data_frame[cat_col]:
            return_lst.append(groups.loc[val])
        data_frame[cat_col + "_" + response] = return_lst
        logging.info(
            "Successfully created new column-%s_%s", cat_col, response)

    new_set_columns = list(set(data_frame.columns) - existing_cols)
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[constants.keep_cols] = data_frame[constants.keep_cols]
    y = data_frame[response]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classfication_report_image_generator(
        y_train, y_test, y_train_pred, y_pred, file_name):
    '''
    generates and stores the classification report image
    input:
            y_test: y testing data
            y_pred: y predicted data
            y_train: y training data
            output_pth: path to store the figure

    output:
             None
    '''
    plt.rc('figure', figsize=(7, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_pred)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_pred)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(file_name)
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    classfication_report_image_generator(
        y_train, y_test, y_train_preds_lr,
        y_test_preds_lr, f"{constants.results_folder}classification_report_lr.png")

    classfication_report_image_generator(
        y_train, y_test, y_train_preds_rf,
        y_test_preds_rf, f"{constants.results_folder}classification_report_rf.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    plt.figure(figsize=(25, 12))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        move_files(constants.model_folder, constants.model_folder_archive)
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        lrc_plot = plot_roc_curve(lrc, x_test, y_test)
        plt.close()
        # plots
        plt.figure(figsize=(15, 8))
        a_x = plt.gca()

        plot_roc_curve(cv_rfc.best_estimator_,
                       x_test, y_test, ax=a_x, alpha=0.8)

        lrc_plot.plot(ax=a_x, alpha=0.8)
        plt.savefig(f"{constants.results_folder}roc_curve.png")
        plt.close()

        # save best model
        joblib.dump(cv_rfc.best_estimator_,
                    f'{constants.model_folder}rfc_model.pkl')
        joblib.dump(lrc, f'{constants.model_folder}logistic_model.pkl')

        classification_report_image(y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)
        logging.info("SUCCESS: Train models done")

    except Exception as err:

        logging.error("Error in train_models - %s", err)
        print(err)


if __name__ == "__main__":
    data_frame = import_data(r"./data/bank_data.csv")
    perform_eda(data_frame)
    data_frame = encoder_helper(
        data_frame, constants.cat_columns, constants.dv)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, constants.dv)
    train_models(x_train, x_test, y_train, y_test)
