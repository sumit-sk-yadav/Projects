import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def data_importer(file_name) -> pd.DataFrame:
    """Takes the file name of the csv file and outputs it as a pandas dataframe

    Args:
        file_name (string): the name of the data file present in the data folder

    Returns:
        Pandas Dataframe
    """
    file_path = os.path.join("data", f"{file_name}.csv")
    try:
        data = pd.read_csv(file_path)
        data = data.sample(17500, random_state=13)
        test_data = data.sample(2000, random_state=15)
        print("Data file imported successfully")
        return data, test_data
    except FileNotFoundError:
        print(f"No such file found: {file_path}")
        return None


def data_imputer(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Imputes the missing values in the given dataframe.
        Uses 'median' as the strategy for numerical column types
        Uses 'most_frequent' or mode as the strategy for categorical column types

    Args:
        dataframe (string): the name of the dataframe that needs the imputation

    Returns:
        Pandas Dataframe with imputed values
    """
    categorical_cols = dataframe.select_dtypes(include="object").columns
    numerical_cols = (
        dataframe.select_dtypes(exclude="object")
        .drop(["Premium Amount", "id"], axis=1)
        .columns
    )

    cat_imputer = SimpleImputer(strategy="most_frequent")
    dataframe[categorical_cols] = cat_imputer.fit_transform(dataframe[categorical_cols])

    num_imputer = SimpleImputer(strategy="median")
    dataframe[numerical_cols] = num_imputer.fit_transform(dataframe[numerical_cols])
    print("Data imputation complete")
    return dataframe


def data_encoder(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Encodes the categorical variables in the dataframe provided. One hot encodes them first and then converts them from boolean values to float

    Args:
        dataframe (pd.Dataframe): name of the dataframe to encode columns from

    Returns:
        pd.DataFrame: Pandas Dataframe with label encoded values for categorical variables
    """
    categorical_columns = dataframe.select_dtypes(include="object")

    dataframe = pd.get_dummies(
        dataframe, columns=categorical_columns.columns, drop_first=True
    )

    bool_cols = dataframe.select_dtypes(include="bool").columns

    dataframe[bool_cols] = dataframe[bool_cols].astype("float32")
    print("Data encoding complete")
    return dataframe


def date_time_converter(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts the given list of columns in the dataframe to the datetime format

    Args:
        dataframe (pd.Dataframe): data frame containing the data
        columns (list) : list of column names to convert to datetime format
    Returns:
        pd.DataFrame: dataframe where the specified columns are converted to datetime format
    """

    for column in columns:
        if column in dataframe:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")

            dataframe[f"{column}_year"] = dataframe[column].dt.year
            dataframe[f"{column}_month"] = dataframe[column].dt.month
            dataframe[f"{column}_weekday"] = dataframe[column].dt.weekday

        else:
            print(f"{column} not found in the dataframe {dataframe}")
    print("Date time conversion complete")
    return dataframe


def data_scaler(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Scales the columns in the list using the standard scaler from sklearn

    Args:
        dataframe (pd.DataFrame): dataframe that contains the columns to be scaled
    Returns:
        pd.DataFrame: dataframe with scaled numerical columns
    """
    scaler = StandardScaler()
    numerical_cols = (
        dataframe.select_dtypes(exclude="object")
        .drop(columns=["Premium Amount", "id"], axis=1)
        .columns
    )
    dataframe[numerical_cols] = scaler.fit_transform(dataframe[numerical_cols])

    print("Data scaling complete")
    return dataframe
