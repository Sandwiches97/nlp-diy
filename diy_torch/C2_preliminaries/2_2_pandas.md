# 2 Data Preprocessing

:label:`sec_pandas`




So far we have introduced a variety of techniques for manipulating data that are already stored in tensors. To apply deep learning to solving real-world problems, we often begin with preprocessing $\color{red}\text{raw data}$, rather than those nicely prepared data in the tensor format.

Among popular data analytic tools in Python, the `pandas` package is commonly used (pandas 经常用于处理原始数据). Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with tensors. So, we will briefly walk through steps for preprocessing raw data with `pandas` and converting them into the tensor format. We will cover more data preprocessing techniques in later chapters.

## 2.1 Reading the Dataset

As an example, we begin by ( **creating an artificial dataset that is stored in a csv (comma-separated values) file** ) `../data/house_tiny.csv`. Data stored in other formats may be processed in similar ways.

Below we write the dataset row by row into a csv file.

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

To [ **load the raw dataset from the created csv file** ], we import the `pandas` package and invoke the `read_csv` function. This dataset has four rows and three columns, where each row describes the number of rooms ("NumRooms"), the alley type ("Alley"), and the price ("Price") of a house.

```python
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
## 2.2 Handling Missing Data 处理缺失值

Note that "NaN" entries are missing values. To handle missing data, typical methods include

- *imputation, 插值法*, where imputation $\text{\colorbox{black}{\color{yellow}replaces}}$ missing values $\text{\colorbox{black}{\color{yellow}with}}$ substituted ones
- *deletion，删除法*, where deletion $\text{\colorbox{black}{\color{yellow}ignores}}$ missing values.

Here we will consider imputation.

By integer-location based indexing (位置索引) (`iloc`), we $\text{\colorbox{black}{\color{yellow}split}}$ `data` $\text{\colorbox{black}{\color{yellow}into}}$ `inputs` and `outputs`, where the former `inputs` takes the first two columns while the latter `outputs` only keeps the last column. For numerical values in `inputs` that are missing, we [ **$\text{\colorbox{black}{\color{yellow}replace}}$ the "NaN" entries $\text{\colorbox{black}{\color{yellow}with}}$ the $\color{red}\text{mean value}$ of the same column.** ]

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
**For categorical or $\text{discrete 离散数据}$ values in `inputs`, we consider "NaN" as a category.** ] Since the "Alley" column only takes two types of categorical values "Pave" and "NaN", `pandas` can automatically convert this column to two columns "Alley_Pave" and "Alley_nan". A row whose alley type is "Pave" will set values of "Alley_Pave" and "Alley_nan" to 1 and 0. A row with a missing alley type will set their values to 0 and 1.

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
## Conversion to the Tensor Format

Now that [ **all the entries in `inputs` and `outputs` are numerical, they can be converted to the tensor format.** ] Once data are in this format, they can be further manipulated with those tensor functionalities that we have introduced in 

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
## Summary

* Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with tensors.
* Imputation and deletion can be used to handle missing data.

## Exercises

Create a raw dataset with more rows and columns.

1. Delete the column with the most missing values.
2. Convert the preprocessed dataset to the tensor format.

[Discussions](https://discuss.d2l.ai/t/29)
