# EDA Reference Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Data](#understanding-data)
3. [Feature Engineering](#feature-engineering)
4. [EDA Univariate](#eda-univariate)
   - [Distributions](#distributions)
   - [Understanding Distribution Curves](#understanding-distribution-curves)
   - [Standard Deviation and Empirical Rule](#standard-deviation-and-empirical-rule)
   - [Z-Score and Z-Table](#z-score-and-z-table)
   - [Solving Normal Distribution Questions](#solving-normal-distribution-questions)
   - [Log-Normal Transformation](#log-normal-transformation)
   - [Probability Density Function (PDF)](#probability-density-function-pdf)
   - [Statistical Measures](#statistical-measures)
   - [Plots for Univariate Analysis](#plots-for-univariate-analysis)

---

## Introduction

### What is EDA?

#### Core Concept
**Exploratory Data Analysis (EDA)** is the process of analyzing datasets to summarize their main characteristics, often using visual methods. It's a critical first step in data science to understand patterns, spot anomalies, test hypotheses, and check assumptions.

#### How It Works
EDA involves:
- Examining data structure and types
- Checking for missing values and outliers
- Visualizing distributions and relationships
- Identifying patterns and trends
- Formulating hypotheses for further analysis

#### Key Points
- **Purpose**: Understand data before applying complex models
- **Tools**: Pandas, NumPy, Matplotlib, Seaborn
- **Iterative**: Often requires multiple passes through the data
- **Foundation**: Essential for building accurate machine learning models

#### 🌍 Real-World Use Cases
**Real-world example:**
Before building a customer churn prediction model, EDA helps identify which features (age, purchase history, engagement metrics) are most correlated with churn, guiding feature selection and model design.

#### ⚠️ Important Note
EDA is not about confirming pre-existing assumptions - it's about letting the data speak and discovering unexpected patterns that may challenge initial hypotheses.

---

## Understanding Data

### Initial Data Inspection

#### Core Concept
The first step in EDA is understanding the structure, types, and quality of your data before making any transformations.

#### How It Works
1. **Use `info()`** to get an overview of the dataset
2. **Observe output** for null values and data types
3. **Document findings** in notebook using markdown
4. **Check uniqueness** with `nunique()`
5. **Perform initial analysis** without data imputation
6. **Remove unnecessary columns** like serial numbers

#### Example
```python
# Get data overview
data.info()

# Check unique values in each column
data.nunique()

# Remove unnecessary columns
data = data.drop(columns=['S.No', 'ID'])
```

#### Key Points
- **`info()`**: Shows column names, non-null counts, and data types
- **`nunique()`**: Reveals cardinality of each column (helpful for identifying categorical vs numerical)
- **No imputation yet**: Part B of EDA will handle missing values
- **Document everything**: Use markdown cells to note observations

#### ⚠️ Important Note
Always perform initial analysis on the original data distribution. Data imputation should only happen after understanding the raw data patterns.

---

## Feature Engineering

### Creating and Cleaning Features

#### Core Concept
Feature engineering transforms raw data into meaningful features that improve model performance.

#### How It Works
1. **Create new features** from existing ones
2. **Clean text data** using various techniques
3. **Identify feature types** (numerical vs categorical)

#### Text Normalization Techniques

##### Using Regex
```python
import re

def clean_text(text):
    # Convert to lowercase
    text = text.str.lower()
    # Remove special characters
    text = text.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    # Remove extra spaces
    text = text.str.strip()
    return text

data['cleaned_column'] = clean_text(data['raw_column'])
```

##### Manual Exact Mapping
```python
# Create mapping dictionary
mapping = {
    'male': 'M',
    'female': 'F',
    'Male': 'M',
    'Female': 'F'
}

data['gender'] = data['gender'].map(mapping)
```

##### Normalization + Exact Match
```python
# Normalize first, then exact match
data['normalized'] = data['column'].str.lower().str.strip()
data['category'] = data['normalized'].map(category_mapping)
```

##### Fuzzy Matching
```python
from fuzzywuzzy import process

# Find closest match from a list
def fuzzy_match(value, choices, threshold=80):
    match, score = process.extractOne(value, choices)
    return match if score >= threshold else value

data['standardized'] = data['column'].apply(lambda x: fuzzy_match(x, standard_list))
```

#### Identifying Feature Types
```python
# Separate numerical and categorical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object', 'category']).columns

print("Numerical:", numerical_features.tolist())
print("Categorical:", categorical_features.tolist())
```

#### Key Points
- **Regex**: Powerful for pattern-based text cleaning
- **Exact mapping**: Best when you have a known set of values
- **Fuzzy matching**: Useful for typos and variations
- **Feature type identification**: Critical for choosing the right analysis methods

#### 💡 Tip
Always keep the original column when creating engineered features. This allows you to compare and revert if needed.

---

## EDA Univariate

### Analyzing Single Variables

#### Core Concept
Univariate analysis examines each variable individually to understand its distribution, central tendency, and spread.

### Distributions

#### Symmetric Distributions (Mean = Median = Mode)

##### Core Concept
**Symmetric distributions** are perfectly balanced around their center. In these distributions, the three measures of central tendency coincide at the same point.

##### Normal (Gaussian) Distribution
The most common symmetric distribution, also known as the bell curve.

**Key characteristics:**
- **Mean = Median = Mode**: All three measures of central tendency are identical
- **Symmetric**: Left and right sides are mirror images
- **Asymptotic**: The curve never touches the x-axis (extends to infinity)
- **Unimodal**: Has a single peak

**Visual representation:**
```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   μ-3σ μ-σ  μ  μ+σ μ+3σ
```

##### Other Symmetric Distributions

**Uniform Distribution:**
- All values have equal probability
- Mean = Median = Mode (for continuous uniform)
- Shape: Flat rectangle
```
┌─────────────────────────────┐
│                             │
└─────────────────────────────┘
```

**Laplace Distribution:**
- Similar to normal but with sharper peak
- Mean = Median = Mode
- Heavier tails than normal distribution

#### 🔑 Key Point
When mean = median = mode, the distribution is symmetric. This symmetry simplifies statistical analysis and is a key assumption for many statistical tests.

#### Skewed Distribution
When data is not symmetrically distributed.

**Right (Positive) Skew:**
```
    ╭─╮
   ╭─╯ ╰─╮
  ╭─╯     ╰─╮
 ╭─╯         ╰───────
╭─╯
```
- Tail extends to the right
- Mean > Median > Mode
- Common in income data, wait times

**Left (Negative) Skew:**
```
           ╭─╮
      ╭─────╯ ╰─╮
    ╭─╯         ╰─╮
  ╭─╯             ╰─╮
╭─╯                 ╰─╮
```
- Tail extends to the left
- Mean < Median < Mode
- Common in test scores, age at death

#### Key Points
- **Skewness**: Measure of asymmetry (0 = symmetric, >0 = right skew, <0 = left skew)
- **Impact**: Affects statistical tests and model assumptions
- **Transformation**: Log/square root can help normalize skewed data

---

### Understanding Distribution Curves

#### Core Concept
A distribution curve is a visual representation of how data values are spread across different ranges. Each point on the curve tells us something about the probability or frequency of data occurring at that value.

#### What the Axes Mean

**X-Axis (Horizontal):**
- Represents the **data values** (the variable being measured)
- Shows the range of possible values
- For example: height, weight, test scores, income
- The center (μ) is the mean of the distribution

**Y-Axis (Vertical):**
- Represents the **probability density** or **frequency**
- Height of the curve at any point indicates how likely that value is
- Higher points = more data values cluster there
- Lower points = fewer data values occur there

#### What Each Point on the Curve Means

**Peak (Highest Point):**
- Represents the **mode** (most frequent value)
- In normal distribution, this is also the mean and median
- Maximum probability density

**Points Away from Peak:**
- Lower height = lower probability of occurrence
- As you move away from center, values become less likely
- The curve never touches zero (asymptotic) - theoretically, any value is possible

**Area Under the Curve:**
- Total area = 1 (or 100% of the data)
- Area between two points = probability of data falling in that range
- This is why we can say "68% of data is within 1 standard deviation"

#### Visual Example: Normal Distribution Curve

```
Probability Density (Y-axis)
    ↑
    │       ╭───────╮
    │      ╭─╯       ╰─╮
    │     ╭─╯           ╰─╮
    │    ╭─╯               ╰─╮
    │   ╭─╯                   ╰─╮
────┴───╯─────────────────────────╰───→ Data Values (X-axis)
    μ-3σ μ-σ      μ      μ+σ μ+3σ
         ↑        ↑        ↑
      Less     Most     Less
      likely   likely   likely
```

**Interpretation:**
- Point at μ (center): Highest probability - most data values here
- Point at μ+σ: Lower probability than center
- Point at μ+3σ: Very low probability - rare values

#### 🔑 Key Point
The curve is not just a shape - it's a mathematical function where the height at any x-value tells you the relative likelihood of observing that value in your data.

#### 🌍 Real-World Use Cases
**Example: Test Scores**
If you plot student test scores:
- X-axis: Test scores (0-100)
- Y-axis: How many students got each score
- Peak around 75: Most students scored near 75
- Low at 20 and 95: Few students scored very low or very high

#### ⚠️ Important Note
The curve shows probability density, not actual counts. To get actual counts, you multiply the density by the total number of data points and the width of the interval.

---

### Standard Deviation and Empirical Rule

#### Core Concept
**Standard Deviation (σ)** measures how spread out data is from the mean. It's the average distance of data points from the center. The **Empirical Rule** (68-95-99.7 rule) describes how data is distributed in a normal distribution.

#### Mean (μ)

##### Formula
```
μ = (x₁ + x₂ + x₃ + ... + xₙ) / n
```

##### Formula Breakdown
- **μ (mu)**: The population mean (average)
- **x₁, x₂, x₃, ... xₙ**: Individual data values
- **n**: Total number of data points
- **Σ**: Summation (add all values together)

##### Why It's Used
- **Central tendency**: Represents the "typical" value
- **Balance point**: The point where the data would balance if placed on a scale
- **Baseline**: Used as reference for measuring spread

#### Variance (σ²)

##### Formula
```
σ² = Σ(xᵢ - μ)² / n
```

##### Formula Breakdown
- **σ²**: Population variance (sigma squared)
- **xᵢ**: Each individual data point
- **μ**: The mean
- **(xᵢ - μ)**: Deviation of each point from the mean
- **(xᵢ - μ)²**: Squared deviation (eliminates negative values)
- **Σ**: Sum of all squared deviations
- **n**: Number of data points

##### Why It's Used
- **Spread measurement**: Quantifies how dispersed data is
- **Squared units**: Makes deviations positive and emphasizes larger deviations
- **Foundation**: Variance is the building block for standard deviation

#### Standard Deviation (σ)

##### Formula
```
σ = √(Σ(xᵢ - μ)² / n)
```

##### Formula Breakdown
- **σ**: Population standard deviation (sigma)
- **√**: Square root (converts back to original units)
- **Σ(xᵢ - μ)² / n**: The variance
- **Result**: Average distance from the mean in original units

##### Why It's Used
- **Interpretability**: In same units as original data (unlike variance)
- **Spread indicator**: Larger σ = more spread out data
- **Outlier detection**: Points beyond ±3σ are typically outliers
- **Comparison**: Allows comparing spread across different datasets

#### The Empirical Rule (68-95-99.7 Rule)

For a normal distribution:

```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   μ-3σ μ-σ  μ  μ+σ μ+3σ
   └────┴────┴────┴────┴────┘
     99.7%   95%   99.7%
```

**Rule Breakdown:**
- **68%** of data falls within **±1σ** (one standard deviation) of the mean
- **95%** of data falls within **±2σ** (two standard deviations) of the mean
- **99.7%** of data falls within **±3σ** (three standard deviations) of the mean

#### What Sigma (σ) Represents

**σ as a Unit of Measurement:**
- **1σ**: One standard deviation unit from the mean
- **2σ**: Two standard deviation units from the mean
- **3σ**: Three standard deviation units from the mean

**Use Cases of Sigma:**
- **Quality control**: In manufacturing, 6σ means only 3.4 defects per million
- **Risk assessment**: In finance, σ measures volatility of returns
- **Process capability**: Determines if a process meets specifications
- **Confidence intervals**: Used to calculate margins of error

#### Example Calculation

```python
import numpy as np

# Sample data: test scores
scores = [72, 85, 90, 78, 82, 88, 76, 91, 84, 79]

# Calculate mean
mean = np.mean(scores)  # μ = 82.5

# Calculate standard deviation
std_dev = np.std(scores)  # σ ≈ 6.2

# Apply Empirical Rule
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"68% of scores between: {mean - std_dev:.1f} and {mean + std_dev:.1f}")
print(f"95% of scores between: {mean - 2*std_dev:.1f} and {mean + 2*std_dev:.1f}")
print(f"99.7% of scores between: {mean - 3*std_dev:.1f} and {mean + 3*std_dev:.1f}")
```

**Output:**
```
Mean: 82.5
Standard Deviation: 6.20
68% of scores between: 76.3 and 88.7
95% of scores between: 70.1 and 94.9
99.7% of scores between: 63.9 and 101.1
```

#### 🔑 Key Point
The Empirical Rule ONLY applies to normal distributions. For skewed distributions, these percentages don't hold.

#### 🌍 Real-World Use Cases
**Manufacturing Quality Control:**
- If product weights have μ=100g, σ=2g
- 68% of products weigh between 98-102g
- 95% weigh between 96-104g
- Products outside 94-106g (±3σ) are considered defective

**Finance:**
- Stock returns with μ=8%, σ=15%
- 68% chance returns are between -7% and +23%
- 95% chance returns are between -22% and +38%

#### ⚠️ Important Note
The formulas above are for **population** parameters. For **sample** data, use:
- Sample mean: x̄ (same formula)
- Sample variance: s² = Σ(xᵢ - x̄)² / (n-1)  (divide by n-1, not n)
- Sample std dev: s = √s²

The (n-1) is called **Bessel's correction** and provides an unbiased estimate.

#### 💡 Tip
When working with real data, you almost always use sample statistics (n-1) unless you have the entire population. Python's `numpy.std()` uses population formula by default - use `ddof=1` for sample: `np.std(data, ddof=1)`

---

### Z-Score and Z-Table

#### Core Concept
**Z-score** (also called standard score) measures how many standard deviations a data point is from the mean. It standardizes different normal distributions so they can be compared and analyzed using a single **Z-table**.

#### Z-Score Formula

```
Z = (X - μ) / σ
```

##### Formula Breakdown
- **Z**: The z-score (number of standard deviations from mean)
- **X**: The individual data value
- **μ**: The population mean
- **σ**: The population standard deviation

##### What Z-Score Tells You
- **Z = 0**: The value equals the mean
- **Z > 0**: The value is above the mean
- **Z < 0**: The value is below the mean
- **|Z| = 1**: One standard deviation from mean
- **|Z| = 2**: Two standard deviations from mean
- **|Z| = 3**: Three standard deviations from mean

#### Example Calculations

**Example 1: μ=65, σ=2.5, X=70**
```
Z = (70 - 65) / 2.5 = 5 / 2.5 = 2
```
70 is 2 standard deviations above the mean.

**Example 2: μ=65, σ=2.5, X=60**
```
Z = (60 - 65) / 2.5 = -5 / 2.5 = -2
```
60 is 2 standard deviations below the mean.

**Example 3: μ=100, σ=15, X=130**
```
Z = (130 - 100) / 15 = 30 / 15 = 2
```
130 is 2 standard deviations above the mean.

#### Z-Table

##### What It Is
A Z-table (standard normal table) provides the cumulative probability P(Z ≤ z) - the area under the standard normal curve to the left of a given z-score.

##### How to Read a Z-Table

**Structure:**
- **Rows**: First two digits of z-score (e.g., 1.9, 2.0)
- **Columns**: Third digit of z-score (e.g., 0.00, 0.01, 0.02)
- **Intersection**: The cumulative probability

**Example: Finding P(Z ≤ 1.96)**
1. Go to row "1.9"
2. Go to column "0.06"
3. Intersection = 0.9750
4. This means 97.5% of data falls below z=1.96

##### Common Z-Scores to Remember

| Z-Score | Probability | Percentage |
|---------|-------------|------------|
| 0.00 | 0.5000 | 50.0% |
| 1.00 | 0.8413 | 84.13% |
| 1.96 | 0.9750 | 97.5% |
| 2.00 | 0.9772 | 97.72% |
| 2.33 | 0.9901 | 99.01% |
| 2.58 | 0.9951 | 99.51% |
| 3.00 | 0.9987 | 99.87% |

#### Why Use Z-Scores?

##### Standardization
- **Compare different distributions**: A score of 85 in one test vs 90 in another
- **Universal scale**: All normal distributions become standard normal (μ=0, σ=1)
- **Single reference**: One Z-table works for all normal distributions

##### Example: Comparing Test Scores
```python
# Student A: Score 85 in Test 1 (μ=70, σ=10)
z_a = (85 - 70) / 10 = 1.5

# Student B: Score 90 in Test 2 (μ=80, σ=5)
z_b = (90 - 80) / 5 = 2.0

# Student B performed better relative to their class
# (2.0σ above mean vs 1.5σ above mean)
```

#### Using Python for Z-Scores

```python
import scipy.stats as stats
import numpy as np

# Calculate z-score manually
x = 70
mu = 65
sigma = 2.5
z = (x - mu) / sigma  # z = 2.0

# Using scipy
z_scipy = stats.zscore([x], ddof=0)[0]  # z = 2.0

# Find probability from z-score
prob = stats.norm.cdf(z)  # P(Z ≤ 2) = 0.9772

# Find z-score from probability
z_from_prob = stats.norm.ppf(0.975)  # z = 1.96
```

#### 🔑 Key Point
Z-scores transform any normal distribution into the standard normal distribution (μ=0, σ=1). This allows us to use a single Z-table for all normal distribution problems.

#### 🌍 Real-World Use Cases
**Medical Testing:**
- Blood test results are often reported as z-scores
- Z-score of +2 means result is 2 standard deviations above normal
- Helps identify abnormal values regardless of the specific test

**Finance:**
- Stock returns converted to z-scores to compare volatility
- Z-score of -3 indicates extreme negative performance

**Quality Control:**
- Process measurements converted to z-scores
- Values beyond ±3σ trigger investigation

#### ⚠️ Important Note
Z-scores assume your data follows a normal distribution. For skewed or non-normal data, z-scores may not be meaningful.

---

### Solving Normal Distribution Questions

#### Core Concept
Solving normal distribution problems involves converting raw values to z-scores, using the Z-table to find probabilities, and applying probability rules to get the final answer.

#### Step-by-Step Approach

**Step 1: Identify Parameters**
- Extract μ (mean) and σ (standard deviation) from the problem
- Identify the X values or range you need to find probability for

**Step 2: Convert to Z-Scores**
- Use Z = (X - μ) / σ for each boundary value
- This standardizes the problem to the standard normal distribution

**Step 3: Use Z-Table**
- Look up each z-score in the Z-table
- Get the cumulative probability P(Z ≤ z)

**Step 4: Calculate Final Answer**
- Apply probability rules based on the question type
- Add, subtract, or complement the probabilities as needed

#### Types of Problems

##### Type 1: P(X < a) - Less Than

**Question:** Find P(X < 70) where μ=65, σ=2.5

**Solution:**
```
Step 1: Z = (70 - 65) / 2.5 = 2
Step 2: Look up Z=2 in table → 0.9772
Answer: P(X < 70) = 97.72%
```

**Visual:**
```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   57.5  60   65   70  72.5
                    ↑
                  Shaded
```

##### Type 2: P(X > a) - Greater Than

**Question:** Find P(X > 70) where μ=65, σ=2.5

**Solution:**
```
Step 1: Z = (70 - 65) / 2.5 = 2
Step 2: Look up Z=2 in table → 0.9772
Step 3: P(X > 70) = 1 - P(X < 70) = 1 - 0.9772 = 0.0228
Answer: P(X > 70) = 2.28%
```

**Visual:**
```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   57.5  60   65   70  72.5
                    ↑
                  Shaded
```

##### Type 3: P(a < X < b) - Between Two Values

**Question:** Find P(60 < X < 70) where μ=65, σ=2.5

**Solution:**
```
Step 1: Z₁ = (60 - 65) / 2.5 = -2
Step 2: Z₂ = (70 - 65) / 2.5 = 2
Step 3: P(Z < -2) = 0.0228
Step 4: P(Z < 2) = 0.9772
Step 5: P(-2 < Z < 2) = 0.9772 - 0.0228 = 0.9544
Answer: P(60 < X < 70) = 95.44%
```

**Visual:**
```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   57.5  60   65   70  72.5
        ↑────────↑
        Shaded
```

##### Type 4: P(X < a) or P(X > a) with Empirical Rule

**Question:** Find P(60 < X < 72.5) where μ=65, σ=2.5

**Solution (Using Empirical Rule):**
```
Step 1: 60 = μ - 2σ, 72.5 = μ + 3σ
Step 2: Range is from -2σ to +3σ
Step 3: P(μ-2σ to μ) = 95/2 = 47.5%
Step 4: P(μ to μ+3σ) = 99.7/2 = 49.85%
Step 5: Total = 47.5% + 49.85% = 97.35%
Answer: P(60 < X < 72.5) = 97.35%
```

**Visual:**
```
       ╭───────╮
      ╭─╯       ╰─╮
     ╭─╯           ╰─╮
    ╭─╯               ╰─╮
───╯─────────────────────╰───
   57.5  60   65   70  72.5
        ↑──────────────↑
        Shaded
```

#### Python Solution

```python
import scipy.stats as stats

# Parameters
mu = 65
sigma = 2.5

# Problem 1: P(X < 70)
x1 = 70
z1 = (x1 - mu) / sigma
prob1 = stats.norm.cdf(z1)
print(f"P(X < 70) = {prob1:.4f}")  # 0.9772

# Problem 2: P(X > 70)
prob2 = 1 - stats.norm.cdf(z1)
print(f"P(X > 70) = {prob2:.4f}")  # 0.0228

# Problem 3: P(60 < X < 70)
x_lower = 60
x_upper = 70
z_lower = (x_lower - mu) / sigma
z_upper = (x_upper - mu) / sigma
prob3 = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
print(f"P(60 < X < 70) = {prob3:.4f}")  # 0.9545

# Problem 4: P(60 < X < 72.5)
x_upper2 = 72.5
z_upper2 = (x_upper2 - mu) / sigma
prob4 = stats.norm.cdf(z_upper2) - stats.norm.cdf(z_lower)
print(f"P(60 < X < 72.5) = {prob4:.4f}")  # 0.9735
```

#### Quick Reference Table

| Question Type | Formula | Example |
|---------------|---------|---------|
| P(X < a) | P(Z < z) where z = (a-μ)/σ | P(X < 70) = P(Z < 2) = 0.9772 |
| P(X > a) | 1 - P(Z < z) where z = (a-μ)/σ | P(X > 70) = 1 - 0.9772 = 0.0228 |
| P(a < X < b) | P(Z < z₂) - P(Z < z₁) | P(60 < X < 70) = 0.9772 - 0.0228 = 0.9544 |
| P(X < a) or P(X > b) | P(Z < z₁) + (1 - P(Z < z₂)) | P(X < 60 or X > 70) = 0.0228 + 0.0228 = 0.0456 |

#### 🔑 Key Point
Always convert to z-scores first. The Z-table gives you P(Z ≤ z), which is the area to the LEFT of the z-score. For "greater than" problems, use the complement rule: P(Z > z) = 1 - P(Z ≤ z).

#### 💡 Tip
For quick estimates, use the Empirical Rule:
- Within ±1σ: ~68%
- Within ±2σ: ~95%
- Within ±3σ: ~99.7%

For precise answers, always use z-scores and the Z-table.

#### 🚨 Common Pitfall
Don't forget to check if the question asks for "less than" or "greater than". P(X > a) requires subtracting from 1, not just looking up the z-score.

---

### Log-Normal Transformation

#### Core Concept
**Log-normal transformation** applies the logarithm function to data values. It converts right-skewed (log-normal) distributions into more symmetric, normal-like distributions by compressing large values and expanding small values.

#### What is Log-Normal Distribution?

A **log-normal distribution** is a distribution where the logarithm of the variable follows a normal distribution. The original data is right-skewed with a long tail to the right.

**Characteristics:**
- **Right-skewed**: Long tail extending to the right
- **Positive values only**: Cannot contain zeros or negative numbers
- **Multiplicative growth**: Results from multiplicative processes
- **Common in real-world data**: Income, prices, population, stock returns

**Visual comparison:**
```
Original (Right-Skewed)      After Log Transform
    ╭─╮                           ╭───────╮
   ╭─╯ ╰─╮                      ╭─╯       ╰─╮
  ╭─╯     ╰─╮                   ╭─╯           ╰─╮
 ╭─╯         ╰───────          ╭─╯               ╰─╮
╭─╯                               ╰─────────────────
```

#### Why Use Log Transformation in EDA?

##### 1. Normalize Skewed Data
**Problem:** Many real-world datasets are right-skewed
- Income data (few very rich, many poor)
- Product prices (few expensive, many cheap)
- Population sizes (few large cities, many small)

**Solution:** Log transformation makes distribution more symmetric
- Compresses large values (e.g., 1000 → 6.9)
- Expands small values (e.g., 10 → 2.3)
- Reduces skewness towards normal distribution

##### 2. Stabilize Variance
**Problem:** In skewed data, variance often increases with the mean
- Higher values have more variability
- Violates assumption of constant variance (homoscedasticity)

**Solution:** Log transformation makes variance more constant
- Equalizes spread across the range
- Helps meet statistical test assumptions

##### 3. Linearize Relationships
**Problem:** Multiplicative relationships are hard to model
- Y = a × X^b (power law)
- Y = a × e^(bX) (exponential)

**Solution:** Log transformation converts to additive (linear)
- log(Y) = log(a) + b × log(X)
- log(Y) = log(a) + b × X
- Makes patterns easier to detect and model

##### 4. Reduce Outlier Influence
**Problem:** Extreme values dominate analysis
- Single outlier can skew mean significantly
- Outliers have disproportionate impact

**Solution:** Log transformation compresses outliers
- Large outliers become less extreme
- Analysis becomes more robust

#### How to Apply Log Transformation

##### Natural Logarithm (ln)
```python
import numpy as np

# Apply natural log
df['log_price'] = np.log(df['price'])

# Equivalent to:
df['log_price'] = np.log1p(df['price'])  # log(1+x) handles zeros
```

##### Log Base 10
```python
# Apply log base 10
df['log10_price'] = np.log10(df['price'])
```

##### Handling Zeros
```python
# Method 1: Add 1 before log (common)
df['log_price'] = np.log1p(df['price'])  # log(1 + x)

# Method 2: Add small constant
df['log_price'] = np.log(df['price'] + 0.01)

# Method 3: Use sign-preserving log
def log_with_sign(x):
    return np.sign(x) * np.log1p(np.abs(x))
```

#### Example: Income Data Transformation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample income data (right-skewed)
income = np.array([30000, 35000, 40000, 45000, 50000, 
                   55000, 60000, 70000, 80000, 100000,
                   150000, 200000, 500000, 1000000, 5000000])

# Create DataFrame
df = pd.DataFrame({'income': income})

# Apply log transformation
df['log_income'] = np.log(df['income'])

# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original distribution
sns.histplot(df['income'], kde=True, ax=axes[0])
axes[0].set_title('Original Income (Right-Skewed)')
axes[0].set_xlabel('Income ($)')

# Log-transformed distribution
sns.histplot(df['log_income'], kde=True, ax=axes[1])
axes[1].set_title('Log-Transformed Income (More Normal)')
axes[1].set_xlabel('Log(Income)')

plt.tight_layout()
plt.show()

# Compare statistics
print("Original Income:")
print(f"  Mean: ${df['income'].mean():,.0f}")
print(f"  Median: ${df['income'].median():,.0f}")
print(f"  Std Dev: ${df['income'].std():,.0f}")
print(f"  Skewness: {df['income'].skew():.2f}")

print("\nLog-Transformed Income:")
print(f"  Mean: {df['log_income'].mean():.2f}")
print(f"  Median: {df['log_income'].median():.2f}")
print(f"  Std Dev: {df['log_income'].std():.2f}")
print(f"  Skewness: {df['log_income'].skew():.2f}")
```

**Output:**
```
Original Income:
  Mean: $643,333
  Median: $60,000
  Std Dev: $1,282,835
  Skewness: 2.87 (highly right-skewed)

Log-Transformed Income:
  Mean: 11.23
  Median: 11.00
  Std Dev: 1.58
  Skewness: 0.45 (much more symmetric)
```

#### When to Use Log Transformation

**Use when:**
- Data is right-skewed (positive skewness > 1)
- Data contains only positive values
- Variance increases with the mean
- You want to reduce outlier influence
- Relationship appears multiplicative or exponential
- Statistical tests assume normal distribution

**Do NOT use when:**
- Data contains zeros or negative numbers (without adjustment)
- Data is already normally distributed
- Interpretability of original scale is critical
- Data has a left-skew (log makes it worse)
- You need to preserve additive relationships

#### Common Use Cases

**Finance:**
- Stock prices (log returns are normally distributed)
- Income and wealth data
- Market capitalization

**Economics:**
- GDP of countries
- Population sizes
- Company revenues

**Science:**
- Bacterial growth (exponential)
- Reaction rates
- Signal processing

**E-commerce:**
- Product prices
- Order values
- Customer spending

#### Interpreting Log-Transformed Data

**Back-transformation:**
```python
# To get back to original scale
original_value = np.exp(log_value)  # for natural log
original_value = 10 ** log_value    # for log base 10
```

**Interpretation:**
- **Difference in log values ≈ percentage change**
- log(200) - log(100) = 5.30 - 4.61 = 0.69 ≈ 69% increase
- A 1-unit increase in log(X) ≈ 2.72x increase in X (for natural log)

#### 🔑 Key Point
Log transformation is a powerful tool for handling right-skewed data, but always check the distribution before and after transformation to ensure it's appropriate for your analysis.

#### 💡 Tip
Always use `np.log1p(x)` instead of `np.log(x)` when your data might contain zeros. `log1p(x)` computes log(1+x) which is numerically stable and handles zeros gracefully.

#### ⚠️ Important Note
After log transformation, interpretations change. A linear model on log-transformed data describes multiplicative relationships on the original scale, not additive ones.

#### 🌍 Real-World Use Cases
**House Price Prediction:**
- Original prices: Right-skewed (few very expensive houses)
- After log transform: More normal distribution
- Better model performance and more reliable predictions

**Customer Lifetime Value:**
- Original CLV: Highly skewed (few high-value customers)
- Log transform: Makes segmentation and analysis easier
- Helps identify meaningful customer segments

---

### Probability Density Function (PDF)

#### Core Concept
The **Probability Density Function (PDF)** is a mathematical function that describes the likelihood of a continuous random variable taking on a particular value. It's the function that creates the smooth curve we see in distribution plots.

#### What It Does
- **Defines the curve shape**: The PDF equation determines the exact shape of the distribution curve
- **Calculates probability density**: For any x-value, PDF(x) gives the height of the curve at that point
- **Area under curve = 1**: The total probability is always 1 (100%)

#### Key Points
- **Not probability itself**: PDF gives density, not direct probability
- **Probability from area**: To get probability, you integrate (find area under curve) between two points
- **Different for each distribution**: Normal, uniform, exponential all have different PDF formulas
- **Library-based**: In practice, we use libraries (SciPy, NumPy) rather than manual calculations

#### Example: Using Libraries

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for normal distribution
mu = 0      # mean
sigma = 1   # standard deviation

# Create range of x values
x = np.linspace(-4, 4, 100)

# Calculate PDF using scipy
pdf_values = stats.norm.pdf(x, loc=mu, scale=sigma)

# Plot the PDF
plt.plot(x, pdf_values)
plt.title('Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()
```

#### 🔑 Key Point
You rarely need to memorize PDF formulas. Libraries like `scipy.stats` handle the calculations. Focus on understanding what the curve represents and how to interpret it.

#### 💡 Tip
When you see a smooth distribution curve in Python (using `sns.kdeplot()` or similar), you're looking at a PDF visualization - the library is computing and plotting the PDF for you.

---

### Statistical Measures

#### Central Tendency
| Measure | Description | When to Use |
|---------|-------------|-------------|
| **Mean** | Average of all values | Normal distribution, no outliers |
| **Median** | Middle value when sorted | Skewed data, outliers present |
| **Mode** | Most frequent value | Categorical data, identifying peaks |

#### Spread
| Measure | Description | Formula |
|---------|-------------|---------|
| **Standard Deviation** | Average distance from mean | √(Σ(x-μ)²/n) |
| **Variance** | Squared standard deviation | σ² |
| **Range** | Max - Min | Max - Min |
| **IQR** | Range of middle 50% | Q3 - Q1 |

#### Visual Representation
```
Normal Distribution with Statistics:

       ╭───────╮
      ╭─╯       ╰─╮      Mode = Median = Mean
     ╭─╯           ╰─╮
    ╭─╯      σ      ╰─╮
───╯─────────────────────╰───
   μ-3σ μ-σ  μ  μ+σ μ+3σ
        ↑   ↑   ↑
      -1σ  μ  +1σ
```

### Plots for Univariate Analysis

#### KDE Plot (Kernel Density Estimation)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize distribution and detect skewness
sns.kdeplot(data['column'])
plt.title('Distribution with KDE')
plt.show()
```

**Use for:**
- Understanding distribution shape
- Detecting skewness
- Identifying multiple modes (peaks)

#### Box Plot
```python
# Detect outliers
sns.boxplot(x=data['column'])
plt.title('Box Plot for Outlier Detection')
plt.show()
```

**Box plot components:**
```
    ┌─── Whisker (max non-outlier)
    │
    ├─ Q3 (75th percentile)
    │
    ├─ Median (50th percentile)
    │
    ├─ Q1 (25th percentile)
    │
    └─── Whisker (min non-outlier)

    ○ Outliers (beyond whiskers)
```

**Use for:**
- Detecting outliers
- Comparing distributions
- Seeing spread and central tendency

#### Histogram
```python
# Visualize frequency distribution
plt.hist(data['column'], bins=30)
plt.title('Histogram')
plt.show()
```

#### Key Points
- **KDE**: Smooth curve showing distribution density
- **Boxplot**: Quick outlier detection and summary statistics
- **Histogram**: Shows actual frequency counts
- **Combine**: Use multiple plots for comprehensive understanding

#### 🚨 Common Pitfall
Don't rely on a single plot. Always use multiple visualization methods to get a complete picture of your data.

#### 💡 Tip
For skewed distributions, consider using median instead of mean for summary statistics, as mean is heavily influenced by outliers.

---

## Summary
- **Understanding Data**: Start with `info()`, `nunique()`, and document findings before any transformations
- **Feature Engineering**: Create meaningful features through text normalization, mapping, and proper feature type identification
- **Distributions**: Understand symmetric distributions (where mean=median=mode), normal distribution characteristics, and skewed distributions
- **Distribution Curves**: Interpret what the axes mean - X-axis shows data values, Y-axis shows probability density/frequency
- **Standard Deviation & Empirical Rule**: Master mean, variance, and standard deviation formulas with component breakdowns; apply the 68-95-99.7 rule for normal distributions
- **Z-Score & Z-Table**: Convert values to z-scores using Z = (X-μ)/σ; use Z-table to find probabilities; standardize different distributions for comparison
- **Solving Normal Distribution Questions**: Follow 4-step approach - identify parameters, convert to z-scores, use Z-table, calculate final answer; handle different problem types (less than, greater than, between values)
- **Log-Normal Transformation**: Apply log transformation to normalize right-skewed data, stabilize variance, linearize relationships, and reduce outlier influence; use when data is positively skewed and contains only positive values
- **PDF**: Probability Density Function defines curve shapes; use libraries like scipy.stats for calculations
- **Univariate Analysis**: Use distributions, statistical measures, and plots (KDE, boxplot, histogram) to understand individual variables
- **Key Principle**: Always analyze original data first, impute missing values later (Part B)
