import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

class DataPrepKit:
   def __init__(self):
       self._df = None
       self._file_path = None

   def read_data(self, filepath):
       self._file_path = filepath
       file_extension = os.path.splitext(filepath)[1].lower()
       try:
           if file_extension == ".csv":
               self._df = pd.read_csv(filepath)
           elif file_extension == ".xlsx" or file_extension == ".xls":
               self._df = pd.read_excel(filepath)
           elif file_extension == ".json":
               self._df = pd.read_json(filepath)
           else:
               print(f"Unsupported file type: {file_extension}")
       except Exception as e:
           print(f"Error reading file: {e}")

   def summarize(self):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return

       print(self._df.info())
       print("\nHead of data:")
       print(self._df.head())
       print("\nKey statistical summaries:")
       print(self._df.describe())
       print("\nCategorical summaries\n")
       for col in self._df.columns:
           if not pd.core.dtypes.common.is_numeric_dtype(self._df[col]):
               print(f"Column: {col}")
               print("Most Frequent Value:", self._df[col].mode().iloc[0])
               print("Value Counts:\n", self._df[col].value_counts())

   def drop_duplicates(self):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       self._df.drop_duplicates(inplace=True)

   def drop_rows_cols(self, rows=None, columns=None):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       if rows is not None:
           self._df.drop(rows, inplace=True)
       if columns is not None:
           self._df.drop(columns, axis=1, inplace=True)

   def drop_empty_columns(self):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       non_null_counts = self._df.notna().sum()
       empty_columns = non_null_counts[non_null_counts == 0].index
       self._df = self._df.drop(columns=empty_columns)

   def impute_avg(self):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       numeric_cols = self._df.select_dtypes(include=[np.number]).columns
       for col in numeric_cols:
           col_mean = self._df[col].mean()
           self._df[col] = self._df[col].fillna(col_mean)

   def impute_zero(self):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       numeric_cols = self._df.select_dtypes(include=[np.number]).columns
       for col in numeric_cols:
           self._df[col] = self._df[col].fillna(0)

   def encode_categorical(self, columns, method):
       if self._df is None:
           print("No data loaded yet. Please read data first.")
           return
       if not set(columns).issubset(set(self._df.columns)):
           raise KeyError(f"{columns} are incorrect column names.")
       if method == "label":
           encoder = LabelEncoder()
       elif method == "ordinal":
           encoder = OrdinalEncoder()
       elif method == "one-hot":
           encoder = OneHotEncoder(handle_unknown='ignore')
       else:
           raise ValueError("Encoding method is not available")
       self._df[columns] = encoder.fit_transform(self._df[columns])

# Get data path from user
data_path = input("Enter the path to your data file: ")

# Create instance of DataPrepKit
data_prep = DataPrepKit()

# Read data
data_prep.read_data(data_path)

# Exploratory Data Analysis
print("\nExploratory Data Analysis:\n")
data_prep.summarize()

# Handle missing values
print("\nHandling missing values:\n")
numeric_cols = data_prep._df.select_dtypes(include=[np.number]).columns
categorical_cols = data_prep._df.select_dtypes(exclude=[np.number]).columns

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Ask user for imputation method for numeric columns
numeric_impute_method = input("Enter imputation method for numeric columns (avg/zero) [default: avg]: ").lower() or "avg"
if numeric_impute_method == "avg":
   data_prep.impute_avg()
elif numeric_impute_method == "zero":
   data_prep.impute_zero()
else:
   print("Invalid choice, using default (avg)")
   data_prep.impute_avg()

categorical_impute_method = input("Enter imputation method for categorical columns (mode/drop) [default: mode]: ").lower() or "mode"
if categorical_impute_method == "mode":
   data_prep._df[categorical_cols] = data_prep._df[categorical_cols].fillna(data_prep._df[categorical_cols].mode().iloc[0])
elif categorical_impute_method == "drop":
   data_prep._df.dropna(subset=categorical_cols, inplace=True)
else:
   print("Invalid choice, using default (mode)")
   data_prep._df[categorical_cols] = data_prep._df[categorical_cols].fillna(data_prep._df[categorical_cols].mode().iloc[0])

# Categorical Data Encoding
print("\nCategorical Data Encoding:\n")
encode_cols = input("Enter the categorical columns to encode (comma-separated): ").split(",")
encode_method = input("Enter encoding method (label/ordinal/one-hot) [default: one-hot]: ").lower() or "one-hot"
data_prep.encode_categorical(encode_cols, encode_method)

# Drop columns
drop_cols = input("Enter the columns to drop (comma-separated) [default: none]: ").split(",") or None
data_prep.drop_rows_cols(columns=drop_cols)

# Drop empty columns
drop_empty = input("Drop empty columns? (y/n) [default: y]: ").lower() or "y"
if drop_empty == "y":
   data_prep.drop_empty_columns()

print("\nData preprocessing complete!")