# Pandas Reference Guide

A practical reference guide for common Pandas operations and concepts.

---

## Table of Contents
1. [Pandas Overview](#pandas-overview)
2. [Series: Creation & Operations](#series-creation--operations)
3. [DataFrame: Creation & I/O](#dataframe-creation--io)
4. [Data Types (dtype)](#data-types-dtype)
5. [Object dtype: NumPy vs Pandas](#object-dtype-numpy-vs-pandas)
6. [Binary/BLOB Data Handling](#binaryblob-data-handling)
7. [DataFrame Inspection](#dataframe-inspection)
8. [Indexing & Selection](#indexing--selection)
9. [Column Access & Subsetting](#column-access--subsetting)
10. [Data Cleaning & Transformation](#data-cleaning--transformation)
11. [Column Operations](#column-operations)
12. [Concatenation](#concatenation)
13. [Merging/Joining](#mergingjoining)
14. [GroupBy](#groupby)
    - [Filtering Groups](#filtering-groups)
    - [GroupBy: transform vs aggregate vs apply](#groupby-transform-vs-aggregate-vs-apply)
15. [Handling Missing Values](#handling-missing-values)
    - [Imputation Strategies](#imputation-strategies)
16. [Data Reshaping](#data-reshaping)
    - [Melt](#melt)
    - [Pivot Table](#pivot-table)
    - [Binning](#binning)
17. [String Methods](#string-methods)
18. [DateTime in Pandas](#datetime-in-pandas)
19. [Data Analysis Functions](#data-analysis-functions)
    - [Unique Values](#unique-values)
    - [Value Counts](#value-counts)
    - [Aggregation Functions](#aggregation-functions)
    - [Duplicates](#duplicates)
20. [Exporting Data](#exporting-data)

---

## Pandas Overview

### What is Pandas?

Pandas is a powerful open-source data analysis and manipulation library for Python. It provides data structures and functions needed to work with structured data efficiently.

### Key Features

- **DataFrame**: 2D labeled data structure (like a spreadsheet or SQL table)
- **Series**: 1D labeled array
- Data reading/writing capabilities (CSV, Excel, SQL, JSON, etc.)
- Data cleaning and preprocessing tools
- Time series functionality
- Statistical analysis tools
- Data visualization integration

### Common Use Cases

- Data cleaning and transformation
- Exploratory data analysis
- Statistical modeling
- Financial analysis
- Scientific computing

### Basic Example

```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['NYC', 'LA', 'Chicago']}
df = pd.DataFrame(data)

# Read from CSV
# df = pd.read_csv('data.csv')
```

---

## Series: Creation & Operations

### What is a Series?

A Series is a 1D labeled array in Pandas, similar to a column in a spreadsheet or a single column from a DataFrame.

### Creating a Series

```python
import pandas as pd
import numpy as np

# From a NumPy array with custom index
votes = np.array([334, 211, 145, 160, 139])
votes_series = pd.Series(votes, index=['R1', 'R2', 'R3', 'R4', 'R5'], name='Votes')
print(votes_series)
# R1    334
# R2    211
# R3    145
# R4    160
# R5    139
# Name: Votes, dtype: int64
```

### Series Parameters

- **data**: Array-like, dict, or scalar value
- **index**: Array-like or Index (default: RangeIndex)
- **dtype**: Data type for the Series
- **name**: Name of the Series

### Common Series Operations

```python
# Access by index label
votes_series['R1']  # 334

# Access by position
votes_series.iloc[0]  # 334

# Filter by condition
votes_series[votes_series > 200]  # R1: 334, R2: 211

# Mathematical operations
votes_series * 2  # Double all values
votes_series.mean()  # Average value
```

### 🌍 Real-World Use Cases

- Storing single column data with custom labels
- Time series data with dates as index
- Survey responses with respondent IDs
- Sensor readings with timestamps

---

## DataFrame: Creation & I/O

### Creating a DataFrame from Dictionary

```python
import pandas as pd

# Create DataFrame from dictionary
rest_dict = {
    "Restaurant": ["R1", "R2", "R3"],
    "Votes": [334, 211, 145],
    "Cost": [1000, 1500, 1200]
}

rest_df = pd.DataFrame(rest_dict)
print(rest_df)
#   Restaurant  Votes  Cost
# 0         R1    334  1000
# 1         R2    211  1500
# 2         R3    145  1200
```

### Reading Data from CSV

```python
# Read CSV file
df = pd.read_csv("zomato_dataset.csv")

# Read with specific parameters
df = pd.read_csv(
    "data.csv",
    header=0,           # Row number for column names
    index_col=0,        # Column to use as index
    na_values=['NA', 'null'],  # Additional strings to recognize as NaN
    encoding='utf-8'
)
```

### Other I/O Operations

```python
# Read Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Read JSON
df = pd.read_json("data.json")

# Read SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)
```

### 🌍 Real-World Use Cases

- Loading datasets from various sources (CSV, Excel, databases)
- Creating DataFrames from API responses
- Combining data from multiple sources
- Data export for reporting

---

## Data Types (dtype)

### Object dtype

`dtype=object` means the array contains elements of different types or non-numeric data.

**Characteristics:**
- Each element can be a different Python object (strings, numbers, lists, etc.)
- NumPy stores references to Python objects instead of raw data
- Similar to a Python list but with NumPy array interface

### Common Scenarios

```python
import numpy as np

# Mixed types
arr = np.array([1, 'hello', 3.14, [1, 2, 3]], dtype=object)
# array([1, 'hello', 3.14, list([1, 2, 3])], dtype=object)

# Strings of different lengths
arr = np.array(['short', 'much longer string'], dtype=object)
# array(['short', 'much longer string'], dtype=object)

# Arrays within arrays
arr = np.array([[1, 2], [3, 4, 5]], dtype=object)
```

### Performance Implications

⚠️ **Important**: Object dtype has significant performance costs:

- **Slower**: No vectorized operations, element-by-element Python calls
- **More memory**: Stores Python object references
- **Limited functionality**: Can't use most NumPy optimizations

### When to Avoid

- Use specific dtypes when possible (`int64`, `float64`, `U10` for strings)
- Only use `object` dtype when you truly need mixed types or variable-length data

### Alternative for Strings

```python
# Fixed-length strings (better performance)
arr = np.array(['cat', 'dog', 'elephant'], dtype='U10')  # Unicode, max 10 chars
```

---

## Object dtype: NumPy vs Pandas

### Key Differences

Pandas handles `object` dtype more intelligently than NumPy:

- **Column-based**: Each Series can have its own dtype
- **Mixed types within columns**: Still possible but less common
- **String optimization**: Pandas has special handling for strings

### Pandas Object dtype in Practice

```python
import pandas as pd
import numpy as np

# Pandas DataFrame with mixed types
df = pd.DataFrame({
    'numbers': [1, 2, 3],           # int64
    'strings': ['a', 'b', 'c'],     # object (but optimized)
    'mixed': [1, 'hello', 3.14]     # object
})

print(df.dtypes)
# numbers      int64
# strings     object
# mixed       object
```

### Pandas String Handling

```python
# Pandas 2.0+ has dedicated string dtype
s = pd.Series(['cat', 'dog', 'elephant'], dtype='string')
# More efficient than object dtype

# Older versions still use object
s = pd.Series(['cat', 'dog', 'elephant'])  # dtype: object
```

### Performance Comparison

- **Better than NumPy**: Pandas has vectorized string operations
- **Still slower than numeric**: Object dtype operations are slower than int/float
- **Memory efficient**: Better memory management than NumPy objects

### Common Pandas Object dtypes

- Strings (default before pandas 2.0)
- Mixed data in columns
- Python objects like lists, dicts
- Categorical data (stored as object internally)

### Best Practices in Pandas

- Use `dtype='string'` for text data (pandas 2.0+)
- Use `category` for repeated strings
- Keep numeric columns as proper numeric dtypes
- Only use object when truly necessary

---

## Binary/BLOB Data Handling

### Can Pandas Handle BLOB?

Yes, Pandas DataFrames can handle BLOB (Binary Large Object) data, but with limitations.

### Method 1: Object dtype for binary data

```python
import pandas as pd

# Store binary data as bytes objects
df = pd.DataFrame({
    'id': [1, 2, 3],
    'image_data': [b'\x89PNG...', b'\x89PNG...', b'\x89PNG...'],
    'files': [open('file1.pdf', 'rb').read(), 
              open('file2.pdf', 'rb').read(), 
              open('file3.pdf', 'rb').read()]
})
```

### Method 2: Reading from databases

```python
import sqlite3

# Connect to database with BLOB columns
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT id, blob_column FROM table", conn)
# BLOB data comes back as bytes objects
```

### Method 3: Specialized approaches

```python
# For images - store as PIL objects
from PIL import Image
import io

def bytes_to_image(byte_data):
    return Image.open(io.BytesIO(byte_data))

df['image'] = df['blob_data'].apply(bytes_to_image)
```

### Limitations

⚠️ **Important considerations:**

- **Memory intensive**: Large BLOBs consume significant RAM
- **Performance**: Slower operations compared to numeric data
- **Display**: Binary data doesn't display nicely in DataFrames
- **Storage**: Not ideal for very large files

### Best Practices

- Store file paths/URLs instead of actual binary data when possible
- Use specialized libraries for specific binary formats (images, audio, etc.)
- Consider chunking for very large BLOB datasets
- Use appropriate database storage for production systems

### Alternative for Large Files

```python
# Store file references instead of actual data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'file_path': ['/path/to/file1.pdf', '/path/to/file2.pdf', '/path/to/file3.pdf']
})
```

---

## DataFrame Inspection

### Quick Data Overview

```python
# View first N rows (default: 5)
df.head(10)

# View last N rows (default: 5)
df.tail()

# View specific number of rows
df.head(3)   # First 3 rows
df.tail(2)   # Last 2 rows
```

### DataFrame Information

```python
# Get concise summary of DataFrame
df.info()
# Shows: column names, non-null counts, dtypes, memory usage
```

### Statistical Summary

```python
# Generate descriptive statistics
df.describe()
# Shows: count, mean, std, min, 25%, 50%, 75%, max for numeric columns

# Include object columns
df.describe(include='all')

# Specific columns
df[['votes', 'cost']].describe()
```

### Understanding the Statistics

**count**: Number of non-null (non-missing) values in each column

**mean**: Average value - sum of all values divided by count
- **Formula**: Σx / n (sum of values divided by number of values)
```python
# Example: [10, 20, 30, 40, 50]
# mean = (10 + 20 + 30 + 40 + 50) / 5 = 30
```

**std** (Standard Deviation): Measure of how spread out the values are from the mean
- **Formula**: √(Σ(x - mean)² / n) (square root of average squared distance from mean)
- Low std = values are close to the mean (consistent data)
- High std = values are spread out (variable data)
```python
# Example: [10, 20, 30, 40, 50] has higher std than [28, 29, 30, 31, 32]
```

**min**: Minimum (smallest) value in the column

**25%** (First Quartile/Q1): 25% of values are below this number
- **Calculation**: Median of the lower half of sorted data
- Separates the lowest 25% from the rest
- Also called the 25th percentile

**50%** (Second Quartile/Median): 50% of values are below this number
- **Calculation**: Middle value when data is sorted (or average of two middle values)
- Also called the 50th percentile or median
- Less affected by extreme values than mean

**75%** (Third Quartile/Q3): 75% of values are below this number
- **Calculation**: Median of the upper half of sorted data
- Separates the lowest 75% from the top 25%
- Also called the 75th percentile

**max**: Maximum (largest) value in the column

### Real-World Example

```python
# Restaurant ratings: [3.5, 4.0, 4.2, 4.5, 4.8]
# count: 5
# mean: 4.2 (average rating)
# std: 0.45 (ratings are fairly consistent)
# min: 3.5 (worst rating)
# 25%: 4.0 (25% of ratings are 4.0 or below)
# 50%: 4.2 (median rating)
# 75%: 4.5 (75% of ratings are 4.5 or below)
# max: 4.8 (best rating)
```

### 🌍 Real-World Use Cases

- Quick data quality check after loading
- Understanding data distribution
- Identifying missing values
- Memory usage optimization

---

## Indexing & Selection

### iloc: Integer Location (Implicit Indexing)

Uses integer positions for selection, similar to NumPy array indexing.

```python
# Single row (returns Series)
df.iloc[0]

# Multiple rows (returns DataFrame)
df.iloc[0:7]

# With step
df.iloc[0:7:2]  # Every 2nd row from 0 to 7

# Specific rows and columns
df.iloc[0:5, 0:3]  # First 5 rows, first 3 columns
```

### loc: Label-based (Explicit Indexing)

Uses the **index column** labels for row selection. Requires setting a custom index first.

```python
# Set custom index (this column becomes the index)
df_explicit = df.set_index("name")
df_explicit.head()

# Single row by index label
df_explicit.loc['Jalsa']

# Multiple rows by index labels
df_explicit.loc[['Jalsa', 'Grand Village']]

# Range of index labels (inclusive)
df_explicit.loc['Jalsa':'Grand Village']

# Row selection + Column selection
df_explicit.loc[['Jalsa', 'Grand Village'], ['votes', 'cost']]
# Returns specific rows AND specific columns
```

**Key points:**
- `.loc[]` uses **index column labels** for row selection (not any column)
- After row selection, add another `[]` to select specific columns
- Syntax: `df.loc[row_labels, column_names]`

### ⚠️ Note: Single [] vs Double [[]]

```python
# Assignment (setting values) - use single []
df.loc["Jalsa", "flag"] = 0  # Sets single cell to 0

# Selection (getting data) - double [[]] returns DataFrame
df.loc[["Jalsa"]]                    # Returns DataFrame (not Series)
df.loc[["Jalsa"], ["flag"]]          # Returns DataFrame with single column
df.loc["Jalsa"]                      # Returns Series
df.loc["Jalsa", "flag"]              # Returns scalar value
```

**When to use which:**
- **Single `[]`**: For assignment (`df.loc[row, col] = value`) or when you want a Series/scalar
- **Double `[[]]`**: For selection when you always want a DataFrame result

### Conditional Selection with loc

Use boolean conditions to filter rows, then select/assign columns.

```python
# Select rows where condition is true, then specific column
df.loc[df['Votes'] < 100, 'Rating']  # Returns Series of Ratings for rows with Votes < 100

# Assign value to rows meeting condition
df.loc[df['Votes'] < 100, 'Rating'] = 4.0  # Sets Rating to 4.0 for rows with Votes < 100

# Multiple conditions
df.loc[(df['Votes'] < 100) & (df['Cost'] > 500), 'Rating'] = 3.5

# Select multiple columns for filtered rows
df.loc[df['Votes'] < 100, ['Rating', 'Cost']]  # Returns DataFrame with those columns
```

### Conditional Selection with iloc

Use boolean array (same length as DataFrame) with iloc. **Note: iloc requires column positions, not names.**

```python
# Create boolean mask
mask = df['Votes'] < 100

# Use mask with iloc - must use column POSITION (integer), not name
df.iloc[mask, 2]  # Column at position 2 (0, 1, 2...)

# To use column name with iloc, get its position first
df.iloc[mask, df.columns.get_loc('Rating')] = 4.0

# ⚠️ Key difference from loc:
# loc:  df.loc[df['Votes'] < 100, 'Rating']  # Uses column NAME
# iloc: df.iloc[mask, 2]                     # Uses column POSITION
```

### Other Useful Variations

```python
# loc with callable
df.loc[lambda x: x['Votes'] > 100, 'Rating']  # Same as df.loc[df['Votes'] > 100, 'Rating']

# iloc with slice on columns
df.iloc[0:5, 2:5]  # Rows 0-4, columns 2-4

# loc with all columns for filtered rows
df.loc[df['Votes'] < 100]  # Returns all columns for filtered rows

# loc with column slice (if columns are sorted)
df.loc['A':'C', 'col1':'col3']  # Range of index labels and column labels
```

### Index Management

```python
# Set a column as index
df.set_index("name", inplace=True)

# Reset index to default integer index
df.reset_index(inplace=True)

# Reset index without keeping old index as column
df.reset_index(drop=True, inplace=True)

# Reset index after groupby (converts MultiIndex Series to DataFrame)
result = df.groupby(["salesman_id","customer_id"])["ord_no"].count().reset_index()
# Without reset_index: Returns Series with MultiIndex
# With reset_index: Returns DataFrame with columns

# Convert Series to DataFrame while preserving index
series.to_frame()
```

#### reset_index() Examples

##### df.reset_index(inplace=True)
**Moves index to column, creates new integer index**
```
Before:
        income
gender         
F       36000.0
M       39000.0

After:
  gender   income
0      F  36000.0
1      M  39000.0
```

##### df.reset_index(drop=True, inplace=True)
**Drops old index completely**
```
Before:
        income
gender         
F       36000.0
M       39000.0

After:
    income
0  36000.0
1  39000.0
```

##### series.to_frame()
**Converts Series to DataFrame while preserving index**
```
Before:
gender
F    36000.0
M    39000.0
Name: income, dtype: float64

After:
         income
gender         
F       36000.0
M       39000.0
```

### ⚠️ Important: iloc vs loc

- **iloc**: Position-based (0, 1, 2...) - end is exclusive
- **loc**: Label-based - end is inclusive
- Use iloc when you want to select by position
- Use loc when you want to select by label

### 🌍 Real-World Use Cases

- Selecting specific rows for analysis
- Time series selection by date range
- Random sampling of data
- Creating subsets for training/testing

---

## Column Access & Subsetting

### Accessing Columns

```python
# Dot notation (only for valid Python identifiers)
df.votes

# Bracket notation (safer, works for all column names)
df["votes"]

# Bracket notation for columns with spaces/special characters
df["approx_cost(for two people)"]
```

### ⚠️ Important: Dot vs Bracket Notation

**Use bracket notation when:**
- Column names have spaces or special characters
- Column names are Python keywords (e.g., "class", "def")
- Column names start with numbers
- You want to create a new column dynamically

**Use dot notation when:**
- Column names are valid Python identifiers
- You prefer cleaner syntax
- You're accessing existing columns (not creating new ones)

### Subsetting Multiple Columns

```python
# Select specific columns
subset_df = df[["name", "votes"]]
subset_df.head()

# Select columns by data type
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
```

### 🌍 Real-World Use Cases

- Creating focused datasets for analysis
- Feature selection for machine learning
- Data export with specific columns
- Reducing memory usage by selecting needed columns

---

## Data Cleaning & Transformation

### String Slicing and Cleaning

Common pattern for cleaning rating data or similar string formats.

```python
import numpy as np

# Example: Clean rating strings like "4.5/5" → 4.5
df["only_rating"] = df["rate"].str[0:-2].replace('N', np.nan).astype(float)
```

### Step-by-Step Breakdown

```python
df["only_rating"] = df["rate"].str[0:-2].replace('N', np.nan).astype(float)
```

**1. `df["rate"].str[0:-2]`**
- Takes the "rate" column (assumed to be strings)
- Slices from beginning to 2nd-to-last character
- Removes last 2 characters (likely rating suffix like "/5" or "★")

**2. `.replace('N', np.nan)`**
- Replaces any 'N' values with NaN (Not a Number)
- Handles specific missing value indicator

**3. `.astype(float)`**
- Converts the cleaned strings to float numbers

### Example Transformation

```python
# Original data:
df["rate"] = ['4.5/5', '3.0/5', 'N', '5.0/5']

# After str[0:-2]: ['4.5', '3.0', 'N', '5.0']
# After replace: ['4.5', '3.0', nan, '5.0']  
# After astype(float): [4.5, 3.0, nan, 5.0]
```

### Common Use Cases

- Movie ratings ("4.2/5" → 4.2)
- Product ratings ("3.8★" → 3.8)
- Removing units from measurements ("100kg" → 100)

### ⚠️ Important Note on `.replace()`

`.replace('N', np.nan)` only replaces the **exact character 'N'** with NaN. It does not handle:
- Other invalid values like 'NA', 'null', 'missing', '-'
- Empty strings ''
- Whitespace 'N '
- Case variations 'n'

### Comprehensive Missing Data Handling

```python
# Multiple replacements
df["rate"].replace(['N', 'NA', 'null', '', '-'], np.nan)

# More robust approach using pd.to_numeric
df["rate"] = pd.to_numeric(df["rate"], errors='coerce')
# This converts any non-numeric to NaN automatically
```

### Why Just 'N'?

- 'N' might be the specific code used in that dataset for "No rating"
- Could be a standardized missing value indicator in that particular data source
- Targeted replacement for known data quality issues

---

## Concatenation

### pd.concat for Series

Combine multiple Series along a specified axis.

```python
import pandas as pd
import numpy as np

votes = np.array([334, 211, 145, 160, 139])
cost = np.array([1100, 1200, 1000, 1400, 1700])

votes_series = pd.Series(votes, index=['R1', 'R2', 'R3', 'R4', 'R5'], name='Votes')
cost_series = pd.Series(cost, index=['R1', 'R2', 'R3', 'R4', 'R5'], name='Cost')

# Concatenate vertically (stack series)
result = pd.concat([votes_series, cost_series], axis=0)
# Returns a longer Series with all values

# Concatenate horizontally (create DataFrame)
result = pd.concat([votes_series, cost_series], axis=1)
# Returns a DataFrame with both series as columns
```

### pd.concat for DataFrames

```python
# Concatenate DataFrames vertically (add rows)
df_combined = pd.concat([df1, df2], axis=0)

# Concatenate DataFrames horizontally (add columns)
df_combined = pd.concat([df1, df2], axis=1)

# Ignore index when concatenating
df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)

# Concatenate with keys (creates MultiIndex)
df_combined = pd.concat([df1, df2], axis=0, keys=['dataset1', 'dataset2'])
```

### Key Parameters

- **axis**: 0 for vertical (rows), 1 for horizontal (columns)
- **ignore_index**: Reset index (default: False)
- **keys**: Add hierarchical index
- **join**: 'inner' or 'outer' (default: 'outer')

### 🌍 Real-World Use Cases

- Combining multiple datasets
- Adding new features to existing data
- Merging time series data
- Creating batch datasets for machine learning

---

## Merging/Joining

### pd.merge for DataFrames

Combine DataFrames based on common columns (similar to SQL joins).

```python
pd.merge(left_df, right_df, on='key', how='join_type')
```

### Join Types (how parameter)

#### Inner Join (default)
```python
pd.merge(df1, df2, on='id', how='inner')
# Only keeps rows where key exists in BOTH DataFrames
```
- Returns intersection of keys
- Rows without matching keys are dropped

#### Left Join
```python
pd.merge(df1, df2, on='id', how='left')
# Keeps all rows from left DataFrame, matches from right
```
- All rows from left DataFrame
- NaN for unmatched columns from right DataFrame

#### Right Join
```python
pd.merge(df1, df2, on='id', how='right')
# Keeps all rows from right DataFrame, matches from left
```
- All rows from right DataFrame
- NaN for unmatched columns from left DataFrame

#### Outer Join
```python
pd.merge(df1, df2, on='id', how='outer')
# Keeps all rows from BOTH DataFrames
```
- Union of all keys
- NaN where no match exists

### Join on Multiple Columns

```python
# Join on single column
pd.merge(df1, df2, on='id')

# Join on multiple columns
pd.merge(df1, df2, on=['id', 'date'])
# Matches rows where BOTH id AND date are equal
```

### Different Column Names

```python
# When columns have different names in each DataFrame
pd.merge(df1, df2, left_on='id', right_on='user_id')
```

### Join on Index

```python
# Join on index instead of columns
pd.merge(df1, df2, left_index=True, right_index=True)
```

### Example

```python
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [2, 3, 4], 'age': [20, 30, 40]})

# Inner: id 2, 3 (common to both)
# Left: id 1, 2, 3 (all from df1)
# Right: id 2, 3, 4 (all from df2)
# Outer: id 1, 2, 3, 4 (all from both)
```

### 🌍 Real-World Use Cases

- Combining customer data from different sources
- Merging transaction data with product information
- Joining sales data with geographical data
- Enriching datasets with additional features

---

## GroupBy

### What is GroupBy?

GroupBy splits data into groups based on column values, then applies aggregation functions to each group.

```python
# Basic syntax
df.groupby("column")["target_column"].aggregation_function()
```

### Single Group, Single Aggregation

```python
# Group by category, calculate mean rating for each category
sales_df.groupby("category")["rating"].mean()
# Result: Series with category as index, mean rating as values
```

### Single Group, Count

```python
# Group by user, count how many products each user bought
sales_df.groupby("user_name")["product_name"].count()
# Result: Series with user_name as index, count as values
```

### Multiple Groups, Single Aggregation

```python
# Group by category AND product_name (creates MultiIndex)
sales_df.groupby(["category", "product_name"])["rating"].mean()
# Result: Series with MultiIndex (category, product_name)
```

### Multiple Groups, Multiple Aggregations

```python
# Group by multiple columns, apply multiple aggregations
sales_df.groupby(["category", "product_name"])["rating"].agg(["mean", "max", "min", "count"])
# Result: DataFrame with MultiIndex (category, product_name) and columns: mean, max, min, count
```

**Example output:**
```
                              mean  max  min  count
category  product_name
Electronics Laptop          4.5   5.0  4.0      10
           Phone           4.2   4.8  3.5      15
Clothing   Shirt           4.0   4.5  3.8       8
```

### What if you just do groupby()?

```python
# Just creates GroupBy object - no results
sales_df.groupby("category")
# Returns: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>
```

**You must apply an aggregation function to see results:**

```python
# Get the GroupBy object
grouped = sales_df.groupby("category")

# Apply aggregation to get results
grouped.mean()           # Mean of all numeric columns
grouped.count()          # Count of non-null values
grouped["rating"].mean() # Mean of specific column
grouped.agg({"rating": "mean", "price": "sum"})  # Multiple aggregations
```

### Filtering Groups

#### Method 1: Filter After Aggregation using `.loc[lambda x: ...]`

Filters the **aggregated results** after computing statistics.

```python
# Example: Filter categories with an average rating >= 4
categories_high_rating = df.groupby('category')['rating'].mean().loc[lambda x: x > 4]

# Display the results
print("Categories with High Average Rating (Above 4):")
categories_high_rating
```

**How it works:**
- `df.groupby('category')['rating'].mean()` - Groups by category and calculates the mean rating for each
- `.loc[lambda x: x > 4]` - Filters the resulting Series to keep only categories where the mean rating is greater than 4
- The lambda function receives the aggregated Series and returns a boolean mask

**Use case:** When you want to filter the **aggregated results** (e.g., keep only categories that meet a criteria)

---

#### Method 2: Filter Groups using `.filter(lambda group: ...)`

Filters the **original data** based on group-level criteria.

```python
# Example: Filter users who have written more than 15 reviews
users_with_many_reviews = df.groupby('user_id').filter(
    lambda group: group['review_id'].count() > 15
)
```

**How it works:**
- `df.groupby('user_id')` - Groups by user_id
- `.filter(lambda group: ...)` - Keeps only entire groups that satisfy the condition
- The lambda receives each **group** (a DataFrame subset) and returns True/False
- If True, the entire group is kept; if False, the entire group is dropped
- Returns the **original DataFrame rows** (not aggregated)

**Use case:** When you want to filter the **original data** based on group-level criteria (e.g., keep all reviews from users who have written more than 15 reviews)

---

### When to Use Which Method?

| Method | Returns | When to Use |
|--------|---------|-------------|
| `.loc[lambda x: ...]` | Aggregated summary (Series/DataFrame) | You want summary statistics that meet a criteria (e.g., "show me categories with avg rating > 4") |
| `.filter(lambda group: ...)` | Original DataFrame rows | You want the full data filtered by group properties (e.g., "show me all reviews from users with > 15 reviews") |

**Key Difference:**
- **`.loc[lambda x: ...]`** → Filters the **aggregated result** (returns summary statistics)
- **`.filter(lambda group: ...)`** → Filters the **original rows** (returns full data, just with some groups removed)

### Other GroupBy Operations

```python
# Iterate through groups
for name, group in sales_df.groupby("category"):
    print(f"Category: {name}")
    print(group)

# Get specific group
grouped.get_group("Electronics")

# Get size of each group
grouped.size()

# Apply custom function
grouped.apply(lambda x: x.max() - x.min())
```

### Key Concepts

- `groupby("column")`: Creates groups based on unique values in that column
- `["column"]` after groupby: Selects which column to aggregate
- `.mean()`, `.count()`, etc.: Aggregation function applied to each group
- `.agg([...])`: Apply multiple aggregation functions at once
- Multiple columns in groupby: Creates hierarchical grouping (MultiIndex)
- **Important**: `groupby()` alone returns a GroupBy object - you need an aggregation function to get results

### 🌍 Real-World Use Cases

- Calculate average rating per product category
- Count transactions per customer
- Find max/min values per group
- Summarize sales data by region/time period
- Compare performance across different segments

---

## GroupBy: transform vs aggregate vs apply

### Understanding the Three Methods

When working with GroupBy objects, you have three main methods for applying operations: **aggregate**, **transform**, and **apply**. Each serves a different purpose.

### 1. Aggregate (agg) - Reduces Data

**Purpose**: Calculate summary statistics for each group. Returns one value per group.

**When to use**: When you want summary statistics (mean, sum, count, etc.) per group.

```python
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR'],
    'salary': [50000, 60000, 70000, 80000, 45000, 55000]
})

# Aggregate - reduces to one value per group
result = df.groupby('department')['salary'].agg('mean')
print(result)
# department
# HR      50000.0
# IT      75000.0
# Sales   55000.0
# Name: salary, dtype: float64

# Multiple aggregations
result = df.groupby('department')['salary'].agg(['mean', 'max', 'min'])
print(result)
#            mean    max    min
# department                    
# HR       50000  55000  45000
# IT       75000  80000  70000
# Sales    55000  60000  50000
```

**Key characteristics**:
- Returns Series or DataFrame with one row per group
- Reduces data size (fewer rows than original)
- Common aggregations: mean, sum, count, min, max, std

---

### 2. Transform - Maintains Shape

**Purpose**: Apply a function to each group but return a result with the same shape as original data.

**When to use**: When you want to add calculated values back to the original DataFrame (e.g., fill missing values, normalize within groups, add group-level statistics).

```python
# Transform - returns same shape as original
result = df.groupby('department')['salary'].transform('mean')
print(result)
# 0    55000.0
# 1    55000.0
# 2    75000.0
# 3    75000.0
# 4    50000.0
# 5    50000.0
# Name: salary, dtype: float64

# Add group mean as a new column
df['dept_mean_salary'] = df.groupby('department')['salary'].transform('mean')
print(df)
#   department  salary  dept_mean_salary
# 0      Sales   50000          55000.0
# 1      Sales   60000          55000.0
# 2         IT   70000          75000.0
# 3         IT   80000          75000.0
# 4         HR   45000          50000.0
# 5         HR   55000          50000.0

# Fill missing values with group median (Titanic example)
titanic_clean['age'] = titanic_clean.groupby(['sex', 'pclass'])['age'].transform(
    lambda x: x.fillna(x.median())
)
```

**Key characteristics**:
- Returns Series/DataFrame with same length as original
- Broadcasts group-level calculation to each row in the group
- Perfect for adding group statistics as new columns
- Common uses: fillna, normalization, ranking within groups

---

### 3. Apply - Flexible Custom Operations

**Purpose**: Apply any custom function to each group. Most flexible but can be slower.

**When to use**: When you need complex operations that aggregate/transform can't handle.

```python
# Apply - flexible custom operations
def custom_function(group):
    # Custom logic: return salary range for each department
    return group['salary'].max() - group['salary'].min()

result = df.groupby('department').apply(custom_function)
print(result)
# department
# HR      10000
# IT      10000
# Sales   10000
# dtype: int64

# Apply can return different shapes
def get_top_earner(group):
    return group.nlargest(1, 'salary')

result = df.groupby('department').apply(get_top_earner)
print(result)
#              department  salary
# department                         
# HR               HR       55000
# IT                IT      80000
# Sales          Sales      60000
```

**Key characteristics**:
- Most flexible - can return any shape
- Slower than aggregate/transform (not optimized)
- Use when you need complex custom logic
- Can return Series, DataFrame, or scalar

---

### Comparison Table

| Method | Purpose | Return Shape | Speed | Use Case |
|--------|---------|--------------|-------|----------|
| **aggregate** | Calculate summary statistics | One row per group | Fast | Get mean, sum, count per group |
| **transform** | Add group stats to original | Same as original | Fast | Fill missing values, normalize, add columns |
| **apply** | Custom complex operations | Variable | Slow | Complex logic not possible with agg/transform |

---

### Practical Examples

#### Example 1: Fill Missing Values (transform)
```python
# Fill missing ages with group-specific median
df['age'] = df.groupby(['gender', 'class'])['age'].transform(
    lambda x: x.fillna(x.median())
)
```

#### Example 2: Calculate Group Statistics (aggregate)
```python
# Get average salary per department
dept_stats = df.groupby('department')['salary'].agg(['mean', 'std', 'count'])
```

#### Example 3: Normalize Within Groups (transform)
```python
# Z-score normalization within each department
df['salary_zscore'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

#### Example 4: Rank Within Groups (transform)
```python
# Rank employees by salary within their department
df['salary_rank'] = df.groupby('department')['salary'].transform('rank')
```

#### Example 5: Custom Analysis (apply)
```python
# Find departments with salary > 2x the median
def high_salary_dept(group):
    median = group['salary'].median()
    return group[group['salary'] > 2 * median]

high_earners = df.groupby('department').apply(high_salary_dept)
```

---

### Quick Decision Guide

**Use `aggregate` when:**
- You want summary statistics (mean, sum, count, etc.)
- You need to reduce data to one row per group
- Performance is important

**Use `transform` when:**
- You want to add group-level statistics as new columns
- You need to fill missing values with group-specific values
- You want to normalize/rank within groups
- You need to maintain original DataFrame shape

**Use `apply` when:**
- You need complex custom logic
- Your operation doesn't fit aggregate/transform patterns
- You need to return data in a non-standard shape
- Performance is not critical

---

## Handling Missing Values

### isna() vs isnull()

Both `isna()` and `isnull()` are **identical functions** in Pandas - they are aliases of each other. They detect missing values (NaN, None, NaT) in the DataFrame.

```python
# Both produce the same result
df.isna()
df.isnull()
```

### What do they return?

Both functions return a **boolean DataFrame** of the same shape as the original:
- `True` where the value is missing (NaN, None, NaT)
- `False` where the value is present

```python
# Example
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': ['x', None, 'z', 'w'],
    'C': [1.0, 2.0, 3.0, np.nan]
})

df.isna()
#        A      B      C
# 0  False  False  False
# 1  False   True  False
# 2   True  False  False
# 3  False  False   True
```

### df.isnull().sum() / df.isna().sum()

Adding `.sum()` to these functions counts the number of missing values in each column.

```python
# Count missing values per column
df.isnull().sum()
# or
df.isna().sum()

# Result:
# A    1
# B    1
# C    1
# dtype: int64
```

**How it works:**
- `df.isna()` returns a boolean DataFrame (True/False)
- `.sum()` treats `True` as 1 and `False` as 0
- Sums along each column (default axis=0)
- Returns a Series with column names as index and missing value counts as values

### Total Missing Values in Entire DataFrame

```python
# Total missing values across all columns
df.isna().sum().sum()  # Returns a single number

# Or using .values.sum()
df.isna().values.sum()
```

### Missing Values per Row

```python
# Count missing values per row (axis=1)
df.isna().sum(axis=1)

# Result:
# 0    0
# 1    1
# 2    1
# 3    1
# dtype: int64
```

### Common Use Cases

```python
# Check if any column has missing values
df.isna().any()

# Check if DataFrame has any missing values at all
df.isna().any().any()

# Get percentage of missing values per column
df.isna().sum() / len(df) * 100

# Filter columns with missing values
df.columns[df.isna().any()]

# Filter rows with any missing values
df[df.isna().any(axis=1)]
```

### Key Points

- `isna()` and `isnull()` are **exactly the same** - use either one
- `df.isna().sum()` counts missing values **per column**
- `df.isna().sum(axis=1)` counts missing values **per row**
- `df.isna().sum().sum()` gives **total missing values** in entire DataFrame
- These functions only detect NaN, None, and NaT - not custom missing value indicators like 'NA', 'null', etc.

### 💡 Interesting Fact

`isnull()` is just an alias for `isna()`! Both functions are identical in every way. The reason both exist:
- `isnull()` was the original name (for compatibility with R users)
- `isna()` was added later as the preferred name (more consistent with NumPy's `isnan`)
- Both are kept for backward compatibility

**Recommendation:** Use `isna()` as it's the more modern and consistent name, but either works perfectly.

### 🌍 Real-World Use Cases

- Data quality assessment after loading data
- Identifying columns that need imputation
- Deciding which rows to drop due to missing data
- Monitoring data completeness over time

### Imputation Strategies

Filling missing values with mean is a simple approach but not always ideal.

**When mean works:** Normally distributed data, few missing values (<5%), random missingness

**Better alternatives:**
```python
# Median for skewed data
df['price'].fillna(df['price'].median(), inplace=True)

# Forward fill for time series
df['temperature'].fillna(method='ffill', inplace=True)

# New category for categorical data
df['category'].fillna('Unknown', inplace=True)
```

---

## Data Reshaping

### Melt

`melt()` transforms wide-format data to long-format by unpivoting columns into rows.

#### Example

```python
df = pd.DataFrame({
    'student_id': [1, 2],
    'math': [85, 92],
    'science': [90, 88]
})

# Melt to long format
df_long = df.melt(
    id_vars=['student_id'],
    value_vars=['math', 'science'],
    var_name='subject',
    value_name='score'
)

# Result:
#    student_id  subject  score
# 0           1     math     85
# 1           2     math     92
# 2           1  science     90
# 3           2  science     88
```

#### Parameters

- **`id_vars`**: Columns to keep unchanged
- **`value_vars`**: Columns to unpivot
- **`var_name`**: Name for new column containing old column names
- **`value_name`**: Name for new column containing values

#### About the Index

After melting, DataFrame gets a new index (0, 1, 2, 3...) because each original row becomes multiple rows.

### Pivot Table

`pivot_table()` creates a spreadsheet-style pivot table as a DataFrame.

#### Basic Syntax

```python
df.pivot_table(values='value_column', index='row_column', columns='col_column', aggfunc='mean')
```

#### Parameters

- **`values`**: Column to aggregate
- **`index`**: Column for rows
- **`columns`**: Column for columns
- **`aggfunc`**: Aggregation function (default: 'mean')

#### Example

```python
df = pd.DataFrame({
    'category': ['Electronics', 'Clothing', 'Electronics', 'Clothing'],
    'product': ['Laptop', 'Shirt', 'Phone', 'Pants'],
    'sales': [1000, 500, 800, 300]
})

# Create pivot table
pivot = df.pivot_table(
    values='sales',
    index='category',
    columns='product',
    aggfunc='sum'
)

# Result:
# product     Laptop  Phone   Pants  Shirt
# category
# Clothing      NaN    NaN   300.0  500.0
# Electronics 1000.0  800.0    NaN    NaN
```

#### Common Aggregation Functions

```python
# Different aggregation functions
df.pivot_table(values='sales', index='category', aggfunc='sum')
df.pivot_table(values='sales', index='category', aggfunc='mean')
df.pivot_table(values='sales', index='category', aggfunc='count')

# Multiple aggregation functions
df.pivot_table(values='sales', index='category', aggfunc=['sum', 'mean', 'count'])
```

#### Multiple Index/Columns

```python
# Multiple rows and columns
df.pivot_table(
    values='sales',
    index=['category', 'subcategory'],
    columns=['region', 'year'],
    aggfunc='sum'
)
```

#### Handling Missing Values

```python
# Fill missing values
df.pivot_table(values='sales', index='category', columns='product', fill_value=0)

# Drop missing values
df.pivot_table(values='sales', index='category', columns='product', dropna=True)
```

#### When to Use Pivot Table

- **Data Analysis**: Summarize data across categories
- **Reporting**: Create summary tables
- **Business Intelligence**: Create dashboard metrics
- **Data Exploration**: Understand patterns in data

### Binning

Binning groups continuous numerical data into discrete categories or bins.

#### What it Does

Converts continuous values like:
```
[23, 45, 67, 89, 12, 34, 56, 78, 90]
```

Into categorical bins like:
```
['Young', 'Adult', 'Senior', 'Young', 'Child', 'Adult', 'Adult', 'Senior', 'Senior']
```

#### Methods

##### pd.cut() - Equal-width bins

```python
import pandas as pd

ages = [23, 45, 67, 89, 12, 34, 56, 78, 90]

# Create equal-width bins
bins = [0, 18, 35, 50, 65, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']

age_groups = pd.cut(ages, bins=bins, labels=labels)
print(age_groups)
# ['Young Adult', 'Adult', 'Middle Age', 'Senior', 'Child', 'Young Adult', 'Adult', 'Senior', 'Senior']
```

##### pd.qcut() - Equal-frequency bins

```python
# Create bins with equal number of data points
age_groups = pd.qcut(ages, q=3, labels=['Young', 'Middle', 'Old'])
print(age_groups)
# ['Young', 'Middle', 'Old', 'Old', 'Young', 'Young', 'Middle', 'Old', 'Old']
```

#### Key Parameters

##### pd.cut()
- **`x`**: Array-like data to bin
- **`bins`**: Number of bins or bin edges
- **`labels`**: Names for bins (optional)
- **`right`**: Include right edge (default: True)

##### pd.qcut()
- **`x`**: Array-like data to bin
- **`q`**: Number of quantiles (bins)
- **`labels`**: Names for bins (optional)

#### Example with DataFrame

```python
df = pd.DataFrame({
    'age': [23, 45, 67, 89, 12, 34, 56, 78, 90],
    'income': [30000, 50000, 80000, 120000, 15000, 35000, 60000, 90000, 130000]
})

# Age binning
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100],
                       labels=['Young', 'Adult', 'Middle', 'Senior'])

# Income binning
df['income_level'] = pd.qcut(df['income'], q=4,
                           labels=['Low', 'Medium', 'High', 'Very High'])

print(df)
#    age  income age_group income_level
# 0   23   30000     Young         Low
# 1   45   50000     Adult      Medium
# 2   67   80000    Middle        High
# 3   89  120000    Senior   Very High
```

#### When to Use Binning

- **Feature Engineering**: Convert continuous features to categorical
- **Analysis**: Group data into meaningful categories
- **Visualization**: Create histograms or categorical plots
- **Machine Learning**: Some algorithms work better with categorical data

#### cut() vs qcut()

| Method | How it works | When to use |
|--------|---------------|-------------|
| `pd.cut()` | Equal width bins | When range matters more than distribution |
| `pd.qcut()` | Equal frequency bins | When you want balanced groups |

---

## String Methods

### str.contains()

`str.contains()` checks if each element in a Series contains a specific substring or pattern, returning a boolean Series.

#### Basic Usage

```python
# Check if strings contain a substring
df['column'].str.contains('substring')
```

#### Returns

- **True** if the substring/pattern is found
- **False** if not found
- **NaN** if the original value is NaN (unless `na=False`)

#### Examples

##### Simple Substring Search

```python
df = pd.DataFrame({
    'product_name': ['iPhone 13', 'Samsung Galaxy', 'Google Pixel', 'iPhone 12'],
    'description': ['Premium smartphone', 'Android device', 'Camera phone', 'Old model']
})

# Find all iPhone products
df[df['product_name'].str.contains('iPhone')]
# Returns rows with 'iPhone 13' and 'iPhone 12'
```

##### Case Insensitive

```python
# Case insensitive search
df['product_name'].str.contains('iphone', case=False)
# Returns True for both 'iPhone 13' and 'iPhone 12'
```

##### Multiple Patterns (Regex)

```python
# Find products with 'phone' OR 'pixel'
df['product_name'].str.contains('phone|pixel', case=False, regex=True)
```

##### Start/End Anchors

```python
# Products starting with 'iPhone'
df['product_name'].str.contains('^iPhone', regex=True)

# Products ending with 'phone'
df['product_name'].str.contains('phone$', regex=True)
```

##### Handling NaN Values

```python
# Treat NaN as False instead of NaN
df['product_name'].str.contains('iPhone', na=False)
```

#### Common Parameters

- **`pat`**: Pattern to search for (string or regex)
- **`case`**: Case sensitive (default: True)
- **`flags`**: Regex flags
- **`na`**: Value for NaN results (default: NaN)
- **`regex`**: Treat pattern as regex (default: True)

#### Practical Use Cases

```python
# Filter emails
emails = df['email'].str.contains('@gmail.com|@yahoo.com', regex=True)

# Find products with specific features
phones = df['description'].str.contains('camera|battery', regex=True)

# Clean data - identify problematic entries
issues = df['text'].str.contains('[^a-zA-Z0-9 ]', regex=True)  # Special chars
```

#### Key Points

- Returns boolean mask for filtering
- Supports regex patterns for complex searches
- Use `case=False` for case insensitive matching
- Use `na=False` to handle NaN values gracefully

---

## DateTime in Pandas

### Converting to DateTime

`pd.to_datetime()` converts strings or other formats to datetime objects for time-based operations.

#### Basic Conversion

```python
# Convert string column to datetime
df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
```

#### Common Formats

```python
# Various datetime formats that pandas recognizes
dates = pd.to_datetime([
    "2023-01-15",           # YYYY-MM-DD
    "15/01/2023",           # DD/MM/YYYY
    "2023-01-15 14:30:00",  # With time
    "Jan 15, 2023",         # Month name
    "20230115"              # YYYYMMDD
])
```

#### Custom Format

```python
# Specify custom format
pd.to_datetime(df["date"], format="%d/%m/%Y")
```

#### Handling Multiple Columns

```python
# Combine separate date and time columns
df["datetime"] = pd.to_datetime(df[["year", "month", "day"]])
```

### DateTime Operations

#### Extract Components

```python
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_name"] = df["datetime"].dt.day_name()  # Monday, Tuesday, etc.
df["month_name"] = df["datetime"].dt.month_name()  # January, February, etc.
```

#### Time Differences

```python
# Calculate time differences
df["days_since_order"] = (pd.Timestamp.now() - df["order_timestamp"]).dt.days

# Difference between two datetime columns
df["processing_time"] = df["completed_date"] - df["start_date"]
```

#### Filtering by Time

```python
# Filter by date range
recent_orders = df[df["order_timestamp"] >= "2023-01-01"]

# Filter by time of day
morning_orders = df[df["order_timestamp"].dt.hour < 12]

# Filter by day of week
weekday_orders = df[df["order_timestamp"].dt.dayofweek < 5]  # Monday=0, Sunday=6
```

#### Format DateTime

```python
# Format datetime as custom strings
df["month_year"] = df["order_timestamp"].dt.strftime("%B %Y")  # "January 2023"
df["numeric_month_year"] = df["order_timestamp"].dt.strftime("%m %Y")  # "01 2023"

# Extract week number
df["week"] = df["order_timestamp"].dt.isocalendar().week
```


### Key Points

- Convert strings to datetime for time-based analysis
- Use `.dt` accessor to extract datetime components
- Supports various date formats automatically
- Essential for time series analysis and filtering

---

## Column Operations

### Rename Columns

#### Single Column

```python
df.rename(columns={'old_name': 'new_name'})
```

#### Multiple Columns

```python
df.rename(columns={
    'old_name1': 'new_name1',
    'old_name2': 'new_name2',
    'rate': 'rating'
})
```

#### In-place Modification

```python
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### Alternative Methods

#### Direct Assignment

```python
df.columns = ['col1', 'col2', 'col3']  # Must rename all columns
```

#### List Comprehension

```python
# Convert to lowercase
df.columns = [col.lower() for col in df.columns]

# Replace spaces with underscores
df.columns = [col.replace(' ', '_') for col in df.columns]
```

#### String Methods on Columns

```python
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')
```

### Common Patterns

```python
# Clean column names comprehensively
df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

# Add prefix to all columns
df.add_prefix('new_')

# Add suffix to all columns
df.add_suffix('_old')
```

---

## Data Analysis Functions

### Unique Values

#### Get Unique Values

```python
# Unique values from a column
df['column_name'].unique()

# Count of unique values
df['column_name'].nunique()

# All unique values across entire DataFrame
df.stack().unique()
```

### Aggregation Functions

#### agg (Aggregate) Multiple Functions

```python
# Apply multiple aggregation functions to single column
df["cost_for_one"].agg(["mean", "sum", "count", "min", "max"])
# Returns Series with all statistics

# Apply multiple aggregation functions to multiple columns
df[["cost_for_one", "cost_for_two"]].agg(["mean", "sum", "count", "min", "max"])
# Returns DataFrame with statistics for both columns
```

**Why use agg?**
- Apply multiple functions at once instead of calling them separately
- Works for single OR multiple columns
- More efficient and cleaner code

#### Sorting

```python
# Sort by single column
df.sort_values(by="only_rating", ascending=False)

# Sort by multiple columns with different order
df.sort_values(by=["only_rating", "Votes"], ascending=[False, True])
# Sorts by only_rating (descending), then by Votes (ascending)
```

### Value Counts

#### Basic Usage

```python
# Count occurrences of each value
df['column_name'].value_counts()
```

#### Useful Parameters

```python
# Include NaN values in counts
df['column_name'].value_counts(dropna=False)

# Sort by index instead of count
df['column_name'].value_counts().sort_index()

# Percentage instead of count
df['column_name'].value_counts(normalize=True)
```

### Multiple Columns

```python
# Value counts for multiple columns (combinations)
df[['col1', 'col2']].value_counts()

# Unique combinations
df[['col1', 'col2']].drop_duplicates()
```

### Advanced Operations

```python
# Limit results to top N
df['column'].value_counts().head(10)

# Binning continuous data
df['numeric_column'].value_counts(bins=5)

# Custom sorting
df['column'].value_counts().sort_values(ascending=False)
```

### Duplicates

#### Check for Duplicates

```python
# Check if any rows are duplicated (returns boolean Series)
df.duplicated()

# Count number of duplicate rows
df.duplicated().sum()

# Mark all duplicates (including first occurrence)
df.duplicated(keep=False)

# Mark duplicates except last occurrence
df.duplicated(keep='last')
```

#### Remove Duplicates

```python
# Remove duplicate rows (keep first occurrence)
df.drop_duplicates()

# Remove duplicates, keep last occurrence
df.drop_duplicates(keep='last')

# Remove all duplicates
df.drop_duplicates(keep=False)

# Remove duplicates based on specific columns
sample_df.drop_duplicates(subset=["Restaurant"], keep="first")
# Removes rows with duplicate Restaurant names (keeps first occurrence)

sample_df.drop_duplicates(subset=["Restaurant", "Votes"], keep="first")
# Removes rows where BOTH Restaurant AND Votes are duplicates (keeps first occurrence)

# Remove duplicates in-place
df.drop_duplicates(inplace=True)
```

#### Check for Duplicates in Specific Column

```python
# Check duplicates in a single column
sample_df["Restaurant"].duplicated()
# Returns boolean Series: True for rows with duplicate Restaurant values (after first occurrence)

# Get duplicate values from a column
sample_df[sample_df["Restaurant"].duplicated()]
# Returns rows where Restaurant value appears more than once (excluding first occurrence)

# ⚠️ Note: drop_duplicates removes duplicate ROWS, not columns
```

### Common Patterns

```python
# Get most frequent value (mode)
most_common = df['column'].mode()[0]

# Summary of unique values per column
df.nunique()

# Get value counts as DataFrame
df['column'].value_counts().reset_index()
```

### 🌍 Real-World Use Cases

```python
# Analyze customer distribution by region
customer_counts = df['region'].value_counts()

# Find most popular product categories
top_categories = df['category'].value_counts().head(5)

# Check data quality - identify rare values
rare_values = df['status'].value_counts()[df['status'].value_counts() < 10]
```

---

## Exporting Data

### Export to CSV

```python
# Export DataFrame to CSV
agg_location.to_csv("agg_data.csv", index=False)

# Export with specific parameters
df.to_csv(
    "output.csv",
    index=False,          # Don't write row names
    header=True,          # Write column names
    encoding='utf-8',
    sep=','               # Separator (default: comma)
)
```

### Other Export Formats

```python
# Export to Excel
df.to_excel("output.xlsx", sheet_name="Data", index=False)

# Export to JSON
df.to_json("output.json", orient='records')

# Export to SQL
import sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

### 🌍 Real-World Use Cases

- Saving processed data for later use
- Creating reports for stakeholders
- Exporting aggregated data
- Data backup and archival

---

## Summary

This guide covers essential Pandas operations for:

- **Data structures**: Series and DataFrame creation
- **I/O operations**: Reading and writing data from various sources
- **Data types**: Understanding object dtype and performance implications
- **Binary data**: Handling BLOB data with appropriate methods
- **DataFrame inspection**: Quick data overview and statistics
- **Indexing & selection**: iloc, loc, and index management
- **Column operations**: Access, subsetting, and renaming
- **Data cleaning**: String manipulation and missing value handling
- **Concatenation**: Combining Series and DataFrames
- **Data analysis**: Unique values and value counts for exploratory analysis
- **Exporting data**: Saving processed data to various formats

Use this guide as a quick reference for common Pandas operations and best practices.
