
eda_folder = "./images/eda/"

eda_folder_archive = eda_folder + "archive/"

results_folder = "./images/results/"

results_folder_archive = results_folder + "archive/"

model_folder = "./models/"

model_folder_archive = model_folder + "archives/"

log_file_run = "./logs/run_log.log"

log_file_test = "./logs/test_churn_library.log"

data_folder = "./data/"

data_file = data_folder + "bank_data.csv"

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

dv = "Churn"

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
