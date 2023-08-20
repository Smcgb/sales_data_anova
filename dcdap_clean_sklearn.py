# clean the dcdap pet supplies data based on 'DCDAP - EDA Report.ipynb'

#import libraries for reports

#dataframes and arrays
import pandas as pd

#imputation and scaling
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler

# settings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
import warnings

def clean_data():
    """
    Reads the csv file into a pandas dataframe and cleans the data by:
    - setting all column names to lower case
    - remapping sales method
    - cleaning years
    - encoding data
    - scaling data
    - imputing data
    - inverse scaling data
    - setting predicted revenue
    - saving the cleaned data to a csv file
    """
    # read the csv file into a pandas dataframe
    df = pd.read_csv('product_sales.csv')

    # set all column names to lower case
    df.columns = df.columns.str.lower()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    # remap sales method
    map = {'email + call':'both', 'em + call':'both'}
    df['sales_method'].replace(map, inplace=True)

    # clean years
    df[df['years_as_customer'] > 39]['years_as_customer'] = \
        df[df['years_as_customer'] <= 39]['years_as_customer'].median()
    
    # convert categories
    for col in df.columns:
        if df[col].dtype == 'object':
            #convert to category
            df[col] = df[col].astype('category')

    # encode data
    df_encoded = df.copy(deep=True)
    df_encoded.drop(columns=['customer_id'], inplace=True)
    df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    # scale data
    scaler = StandardScaler()
    fit = scaler.fit(df_encoded)
    scaled_data = fit.transform(df_encoded)
    df_scaled = pd.DataFrame(scaled_data, columns=df_encoded.columns)

    # impute data
    imputer = KNN(verbose=False)
    data_knn = df_scaled.copy(deep=True)
    data_knn.iloc[:, :] = imputer.fit_transform(data_knn)

    # inverse scale data
    df_inverse = scaler.inverse_transform(data_knn)
    df_inverse = pd.DataFrame(df_inverse, columns=data_knn.columns)

    # set predicted revenue
    df['revenue'] = df_inverse['revenue']

    # to csv
    df.to_csv('product_sales_cleaned.csv', index=False)

    # suppress warnings
    warnings.filterwarnings("ignore")

def main():
    clean_data()

if __name__ == '__main__':
    main()
