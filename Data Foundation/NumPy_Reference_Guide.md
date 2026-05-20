# NumPy Reference Guide

## Table of Contents
1. [Matrix Multiplication](#matrix-multiplication)
2. [Broadcasting](#broadcasting)
3. [np.vectorize](#npvectorize)
4. [Array Splitting Functions](#array-splitting-functions)
5. [Understanding the Axis Parameter](#understanding-the-axis-parameter)
6. [Stacking Arrays](#stacking-arrays)
7. [Array Concatenation](#array-concatenation)
8. [Advanced Indexing](#advanced-indexing)

---

## Matrix Multiplication

### Dimension Compatibility Rule

Matrix multiplication A × B is defined **only when**:
- If A is an m × n matrix
- And B is a p × q matrix
- Then n must equal p (columns of A = rows of B)
- The result is an m × q matrix

### Computation Rule

To compute the element at position (i, j) in the result matrix:
- Take row i from the first matrix
- Take column j from the second matrix
- Compute the dot product: multiply corresponding elements and sum them

**Example:**
```
A = [a₁₁ a₁₂ a₁₃]    B = [b₁₁ b₁₂]
    [a₂₁ a₂₂ a₂₃]        [b₂₁ b₂₂]
                         [b₃₁ b₃₂]

Result (2×2):
c₁₁ = a₁₁·b₁₁ + a₁₂·b₂₁ + a₁₃·b₃₁
c₁₂ = a₁₁·b₁₂ + a₁₂·b₂₂ + a₁₃·b₃₂
c₂₁ = a₂₁·b₁₁ + a₂₂·b₂₁ + a₂₃·b₃₁
c₂₂ = a₂₁·b₁₂ + a₂₂·b₂₂ + a₂₃·b₃₂
```

### Key Properties

- **Non-commutative**: A × B ≠ B × A (generally)
- **Associative**: (A × B) × C = A × (B × C)
- **Distributive**: A × (B + C) = A × B + A × C
- **Identity matrix**: A × I = I × A = A (where I is the identity matrix)
- **Not defined** if inner dimensions don't match

---

## Broadcasting

### Core Rules

Broadcasting allows operations on arrays of different shapes without explicit replication.

**Two dimensions are compatible when:**
1. They are **equal**, OR
2. One of them is **1**, OR
3. One of them doesn't exist

### The Algorithm (Step-by-Step)

To check if two shapes can broadcast:

1. **Align shapes from right to left** (pad with 1s on the left if needed)
2. **Compare each dimension pair**
3. **Compatible if**: dimensions are equal OR one is 1 OR one doesn't exist
4. **Incompatible if**: dimensions are different AND neither is 1

```
Shape A: (a₁, a₂, ..., aₙ)
Shape B: (b₁, b₂, ..., bₘ)

Step 1: Pad the shorter shape with 1s on the left
Step 2: Compare from rightmost to leftmost
Step 3: Output shape = maximum of each dimension pair
```

### Compatibility Table with Examples

| Shape A | Shape B | Result | Output Shape | Example |
|---------|---------|--------|--------------|---------|
| (3,) | (3,) | ✓ | (3,) | `[1,2,3] + [4,5,6] = [5,7,9]` |
| (3,) | (1,) | ✓ | (3,) | `[1,2,3] + [10] = [11,12,13]` |
| (3,) | () | ✓ | (3,) | `[1,2,3] + 5 = [6,7,8]` |
| (2,3) | (3,) | ✓ | (2,3) | `[[1,2,3],[4,5,6]] + [10,20,30]` |
| (2,3) | (1,3) | ✓ | (2,3) | `[[1,2,3],[4,5,6]] + [[10,20,30]]` |
| (2,3) | (2,1) | ✓ | (2,3) | `[[1,2,3],[4,5,6]] + [[10],[20]]` |
| (2,3) | (1,1) | ✓ | (2,3) | `[[1,2,3],[4,5,6]] + [[10]]` |
| (3,1) | (1,3) | ✓ | (3,3) | `[[1],[2],[3]] + [[10,20,30]]` |
| (2,3) | (2,3) | ✓ | (2,3) | Element-wise operation |
| (2,3) | (3,2) | ✗ | Error | Incompatible shapes |
| (2,3) | (4,) | ✗ | Error | Incompatible shapes |
| (2,3,4) | (3,4) | ✓ | (2,3,4) | 3D with 2D broadcasting |
| (2,3,4) | (1,4) | ✓ | (2,3,4) | 3D with 1D broadcasting |
| (2,3,4) | (2,1,4) | ✓ | (2,3,4) | 3D with 3D broadcasting |

### Quick Test Method

Write shapes aligned right, compare each column:

**Example 1: (2,3) with (3,)**
```
  2   3
      3
  ↓   ↓
  2   3  ✓ (both equal in last dim)
```
**Result: (2,3)**

**Example 2: (3,1) with (1,3)**
```
  3   1
  1   3
  ↓   ↓
  3   3  ✓ (both have 1s, broadcast to max)
```
**Result: (3,3)**

**Example 3: (2,3) with (3,2)**
```
  2   3
  3   2
  ↓   ↓
  ✗   ✗  (2≠3 and neither is 1)
```
**Result: Error**

### Mental Shortcut

**Ask yourself for each dimension (right to left):**
- "Are they the same?" → Yes = OK
- "Is one of them 1?" → Yes = OK
- "Is one missing?" → Yes = OK (treat as 1)
- "Neither?" → **Error**

The output shape is the **maximum** of each dimension pair.

### Key Principle

Broadcasting is **virtual** - no actual data copying occurs. The operation is computed as if the arrays were expanded, but memory efficiency is maintained.

---

## np.vectorize

### What It Is

`np.vectorize` is a convenience function that transforms a Python function into a function that can operate on NumPy arrays element-wise. It's essentially a **loop wrapper**, not true vectorization.

### How It Works

```python
import numpy as np

# Regular Python function (works on scalars)
def add_one(x):
    return x + 1

# Vectorize it
add_one_vec = np.vectorize(add_one)

# Now works on arrays
arr = np.array([1, 2, 3, 4])
result = add_one_vec(arr)  # [2, 3, 4, 5]
```

### When to Use It

**Use when:**
- You have a custom Python function that doesn't support array operations
- The function contains conditional logic, if/else statements
- The function uses Python-only operations (strings, objects, complex logic)
- You need element-wise application but can't rewrite the function
- You want to preserve/force specific output data types

**Example with conditions:**
```python
def categorize(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"

categorize_vec = np.vectorize(categorize)
arr = np.array([-5, 0, 10])
result = categorize_vec(arr)  # ['negative', 'zero', 'positive']
```

### Preserving Output Types

`np.vectorize` allows you to explicitly specify the output data type:

```python
def to_string(x):
    return str(x)

# Without specifying otype - returns object array
to_string_vec1 = np.vectorize(to_string)
arr = np.array([1, 2, 3])
result1 = to_string_vec1(arr)
print(result1.dtype)  # object

# With otype specified - returns string array
to_string_vec2 = np.vectorize(to_string, otypes=[str])
result2 = to_string_vec2(arr)
print(result2.dtype)  # <U1 or similar (proper string type)
```

**Multiple output types:**
```python
def compute_stats(x):
    return x * 2, x + 1

stats_vec = np.vectorize(compute_stats, otypes=[float, int])
arr = np.array([1.5, 2.5, 3.5])
doubled, incremented = stats_vec(arr)

print(doubled.dtype)      # float64
print(incremented.dtype)  # int64
```

### When NOT to Use It

**Avoid when:**
- Performance is critical - it's **still a Python loop**, not true vectorization
- The function can be rewritten with NumPy operations
- You're working with large arrays

### Performance Comparison

```python
import numpy as np

arr = np.random.rand(1000000)

# Method 1: Native NumPy (fastest)
def native_numpy(x):
    return x * 2 + 1

# Method 2: np.vectorize (slow)
def python_func(x):
    return x * 2 + 1
vectorized = np.vectorize(python_func)

# Method 3: Explicit loop (similar to vectorize)
def explicit_loop(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2 + 1
    return result
```

**Typical timings:**
- Native NumPy: ~1-2 ms
- np.vectorize: ~100-200 ms
- Explicit loop: ~100-200 ms

### Multi-threading

**No**, `np.vectorize` does **not** use multi-threading. It's a single-threaded Python loop.

**Native NumPy operations** (true vectorization) CAN use multi-threading:
```python
# These may use multi-threaded BLAS/LAPACK libraries
np.dot(A, B)           # Matrix multiplication
np.matmul(A, B)       # Matrix multiplication
np.sum(large_array)   # Reduction operations
np.linalg.svd(A)      # Linear algebra operations
```

### Better Alternatives

```python
# Instead of vectorize, try:
arr = np.array([1, 2, 3, 4])

# 1. Native NumPy operations
result = arr * 2 + 1

# 2. np.where for conditions
result = np.where(arr < 0, "negative", "positive")

# 3. np.select for multiple conditions
conditions = [arr < 0, arr == 0, arr > 0]
choices = ["negative", "zero", "positive"]
result = np.select(conditions, choices)
```

### Summary

| Method | Multi-threading | Speed | Use Case |
|--------|-----------------|-------|----------|
| Native NumPy | Sometimes (BLAS) | Fastest | Always prefer |
| np.vectorize | No | Slow | Convenience & type control |
| Explicit loop | No | Slow | Same as vectorize |

**Use `np.vectorize` for code readability and type control, not performance.**

---

## Array Splitting Functions

### Overview

| Function | Direction | Axis | Use Case |
|----------|-----------|------|----------|
| `np.split` | General | Any | Split along any axis |
| `np.hsplit` | Horizontal | 1 | Split columns |
| `np.vsplit` | Vertical | 0 | Split rows |
| `np.dsplit` | Depth | 2 | Split depth (3D only) |
| `np.array_split` | General | Any | Uneven splits allowed |

### np.split (General)

Splits array along specified axis.

```python
import numpy as np

arr = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Split into 3 equal parts along axis 0 (rows)
result = np.split(arr, 3, axis=0)
# [array([[0, 1, 2, 3]]),
#  array([[4, 5, 6, 7]]),
#  array([[8, 9, 10, 11]])]

# Split at specific indices along axis 1 (columns)
result = np.split(arr, [2, 3], axis=1)
# [array([[0, 1], [4, 5], [8, 9]]),
#  array([[2], [6], [10]]),
#  array([[3], [7], [11]])]
```

### np.hsplit (Horizontal Split)

Splits array horizontally (columns, axis=1).

```python
arr = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Split into 2 equal parts (columns)
result = np.hsplit(arr, 2)
# [array([[0, 1], [4, 5], [8, 9]]),
#  array([[2, 3], [6, 7], [10, 11]])]

# Split at specific column indices
result = np.hsplit(arr, [1, 3])
# [array([[0], [4], [8]]),
#  array([[1, 2], [5, 6], [9, 10]]),
#  array([[3], [7], [11]])]
```

**Use case:** Separating features in a dataset
```python
data = np.array([[1, 2, 3, 'A'],
                 [4, 5, 6, 'B'],
                 [7, 8, 9, 'C']])

features, labels = np.hsplit(data, [3])
# features: [[1,2,3], [4,5,6], [7,8,9]]
# labels: [['A'], ['B'], ['C']]
```

### np.vsplit (Vertical Split)

Splits array vertically (rows, axis=0).

```python
arr = np.arange(12).reshape(4, 3)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# Split into 2 equal parts (rows)
result = np.vsplit(arr, 2)
# [array([[0, 1, 2], [3, 4, 5]]),
#  array([[6, 7, 8], [9, 10, 11]])]

# Split at specific row indices
result = np.vsplit(arr, [1, 3])
# [array([[0, 1, 2]]),
#  array([[3, 4, 5], [6, 7, 8]]),
#  array([[9, 10, 11]])]
```

**Use case:** Train/test split
```python
data = np.random.rand(100, 5)
train, test = np.vsplit(data, [80])
# train: 80 rows, test: 20 rows
```

### ⚠️ Important Note

**All split functions (`np.split`, `np.vsplit`, `np.hsplit`, `np.dsplit`, `np.array_split`) return a LIST of arrays, not a single array.**

If you need to work with the result as a single array, you must concatenate it back:

```python
# Split returns a list
sections = np.vsplit(arr, 2)  # Returns: [array1, array2]

# To get a single array back, concatenate
result = np.concatenate(sections, axis=0)
```

This is a common source of errors when expecting an array but getting a list.

### np.dsplit (Depth Split)

Splits array along depth (axis=2) - requires 3D array.

```python
arr = np.arange(24).reshape(2, 3, 4)
# Shape: (2, 3, 4)

# Split along depth (axis=2)
result = np.dsplit(arr, 2)
# Shape: [(2, 3, 2), (2, 3, 2)]
```

**Use case:** Splitting RGB channels
```python
image = np.random.rand(100, 100, 3)  # RGB image
r, g, b = np.dsplit(image, 3)
# Each channel: (100, 100, 1)
```

### np.array_split (Uneven Splits)

Allows splitting when array size is not evenly divisible.

```python
arr = np.arange(10)

# np.split would raise error (10 not divisible by 3)
# np.array_split handles it
result = np.array_split(arr, 3)
# [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]

# Uneven split along axis
arr = np.arange(12).reshape(4, 3)
result = np.array_split(arr, 3, axis=0)
# [array([[0, 1, 2]]),
#  array([[3, 4, 5], [6, 7, 8]]),
#  array([[9, 10, 11]])]
```

### When to Use What

| Scenario | Function |
|----------|----------|
| Split rows | `np.vsplit` or `np.split(axis=0)` |
| Split columns | `np.hsplit` or `np.split(axis=1)` |
| Split 3D depth | `np.dsplit` or `np.split(axis=2)` |
| General case | `np.split` with axis parameter |
| Uneven divisions | `np.array_split` |
| Train/test split | `np.vsplit` or `np.array_split` |
| Feature/label separation | `np.hsplit` |

### Quick Reference

```python
# All equivalent for 2D arrays:
np.hsplit(arr, n)      == np.split(arr, n, axis=1)
np.vsplit(arr, n)      == np.split(arr, n, axis=0)

# For 3D arrays:
np.dsplit(arr, n)      == np.split(arr, n, axis=2)
```

---

## Understanding the Axis Parameter

### ⚠️ Critical Distinction: Two Meanings of "axis"

The `axis` parameter behaves differently depending on the operation type:

#### 1. **Reduction Operations** (axis = dimension to eliminate)
- `axis=0`: Collapse rows → reduce to 1 row
- `axis=1`: Collapse columns → reduce to 1 column
- **Functions**: `np.sum()`, `np.mean()`, `np.max()`, `np.min()`, `np.argmax()`, `np.std()`, etc.

#### 2. **Building Operations** (axis = direction to grow)
- `axis=0`: Operate along rows (add more rows)
- `axis=1`: Operate along columns (add more columns)
- **Functions**: `np.concatenate()`, `np.stack()`, `np.split()`, etc.

### Reduction Examples

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# axis=0: Collapse rows (sum down columns)
np.sum(arr, axis=0)  # [12, 15, 18] - one row, three columns

# axis=1: Collapse columns (sum across rows)
np.sum(arr, axis=1)  # [ 6, 15, 24] - three rows, one column

# axis=0: Find max in each column
np.max(arr, axis=0)  # [7, 8, 9] - one row

# axis=1: Find max in each row
np.max(arr, axis=1)  # [3, 6, 9] - one column
```

### Memory Trick

**For Reduction Operations:**
- `axis=0` = "Eliminate the 0th dimension" (rows)
- `axis=1` = "Eliminate the 1st dimension" (columns)

**For Building Operations:**
- `axis=0` = "Operate along the 0th dimension" (add rows)
- `axis=1` = "Operate along the 1st dimension" (add columns)

---

## Stacking Arrays

### Core Concept

Stacking joins multiple arrays together along a **new axis** to create a single array. It's the opposite of splitting operations.

### Main Stacking Functions

| Function | Description | New Axis Position |
|----------|-------------|-------------------|
| `np.stack` | Stack along new axis (default) | First dimension (axis=0) |
| `np.vstack` | Vertical stacking (row-wise) | Adds rows (axis=0) |
| `np.hstack` | Horizontal stacking (column-wise) | Adds columns (axis=1) |
| `np.dstack` | Depth stacking (3rd dimension) | Adds depth (axis=2) |

### np.stack (General Stacking)

Stacks arrays along a new axis. All input arrays must have the **same shape**.

```python
import numpy as np

# 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Stack vertically (default axis=0)
result = np.stack([arr1, arr2, arr3])
# Shape: (3, 3)
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

# Stack along different axis
result = np.stack([arr1, arr2, arr3], axis=1)
# Shape: (3, 3)
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]
```

### np.vstack (Vertical Stack)

Stacks arrays vertically (row-wise). Arrays must have the **same number of columns**.

```python
# 2D arrays with same columns
arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[4, 5, 6]])

result = np.vstack([arr1, arr2])
# [[1, 2, 3],
#  [4, 5, 6]]

# Can also stack 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.vstack([arr1, arr2])
# [[1, 2, 3],
#  [4, 5, 6]]
```

### np.hstack (Horizontal Stack)

Stacks arrays horizontally (column-wise). Arrays must have the **same number of rows**.

```python
# 2D arrays with same rows
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([[4], [5], [6]])

result = np.hstack([arr1, arr2])
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# Can also stack 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.hstack([arr1, arr2])
# [1, 2, 3, 4, 5, 6]
```

### np.dstack (Depth Stack)

Stacks arrays along the third dimension (depth). Creates 3D arrays.

```python
# 2D arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

result = np.dstack([arr1, arr2])
# Shape: (2, 2, 2)
# [[[1, 5],
#   [2, 6]],
#  [[3, 7],
#   [4, 8]]]
```

### Key Use Cases

#### Creating Batches for Machine Learning
```python
# Individual samples
sample1 = np.array([1, 2, 3, 4])
sample2 = np.array([5, 6, 7, 8])
sample3 = np.array([9, 10, 11, 12])

# Create batch
batch = np.stack([sample1, sample2, sample3])
# Shape: (3, 4) - 3 samples, 4 features each
```

#### Combining Image Channels
```python
# RGB channels
red_channel = np.array([[255, 0], [128, 64]])
green_channel = np.array([[0, 255], [64, 128]])
blue_channel = np.array([[0, 0], [32, 96]])

# Combine into RGB image
rgb_image = np.dstack([red_channel, green_channel, blue_channel])
# Shape: (2, 2, 3)
```

---

## Array Concatenation

### Core Concept

Concatenation joins arrays together along an **existing axis**. Unlike stacking, it doesn't create new dimensions - it extends existing ones.

### np.concatenate Syntax

```python
np.concatenate(arrays, axis=0, out=None)
```

- `arrays`: Sequence of arrays to concatenate
- `axis`: Axis along which to join (default: 0)
- `out`: Optional output array

### Understanding the Axis Parameter

For concatenation, `axis` determines **which dimension** gets expanded:

#### axis=0 (Default) - Vertical Concatenation
Joins arrays **vertically** (adds rows)

```python
import numpy as np

arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[4, 5, 6]])

result = np.concatenate([arr1, arr2], axis=0)
# [[1, 2, 3],
#  [4, 5, 6]]
# Shape: (2, 3) - rows increased from 1→2
```

#### axis=1 - Horizontal Concatenation
Joins arrays **horizontally** (adds columns)

```python
arr1 = np.array([[1], [2]])
arr2 = np.array([[3], [4]])

result = np.concatenate([arr1, arr2], axis=1)
# [[1, 3],
#  [2, 4]]
# Shape: (2, 2) - columns increased from 1→2
```

### Shape Compatibility Rules

For concatenation along `axis=k`:
- All arrays must have the **same shape** in all dimensions **except** axis `k`
- Along axis `k`, the dimensions can be different (they get added together)

```python
# axis=0: Same columns required
arr1 = np.array([[1, 2, 3]])    # Shape: (1, 3)
arr2 = np.array([[4, 5, 6]])    # Shape: (1, 3) ✓ Same columns
# Result: (2, 3)

# axis=1: Same rows required
arr1 = np.array([[1, 2]])       # Shape: (1, 2)
arr2 = np.array([[3, 4, 5]])    # Shape: (1, 3) ✓ Same rows (1)
# Result: (1, 5)

# Error: Different shapes in non-concatenated dimensions
arr1 = np.array([[1, 2, 3]])    # Shape: (1, 3)
arr2 = np.array([[4, 5]])       # Shape: (1, 2) ✗ Different columns
# np.concatenate([arr1, arr2], axis=0) -> Error!
```

### Key Use Cases

#### Combining Datasets
```python
# Dataset 1 and 2 with same features
data1 = np.array([[1, 2, 3], [4, 5, 6]])    # 2 samples, 3 features
data2 = np.array([[7, 8, 9], [10, 11, 12]]) # 2 samples, 3 features

# Add more samples (axis=0)
combined_data = np.concatenate([data1, data2], axis=0)
# Shape: (4, 3) - 4 samples total
```

#### Adding Features
```python
# Existing features and new features
existing = np.array([[1, 2], [3, 4]])       # 2 samples, 2 features
new_features = np.array([[5], [6]])         # 2 samples, 1 feature

# Add features (axis=1)
enhanced_data = np.concatenate([existing, new_features], axis=1)
# Shape: (2, 3) - 3 features total
```

### Specialized Concatenation Functions

These are convenience wrappers around `np.concatenate`:

```python
# Equivalent operations:
np.vstack([arr1, arr2])    == np.concatenate([arr1, arr2], axis=0)
np.hstack([arr1, arr2])    == np.concatenate([arr1, arr2], axis=1)
np.dstack([arr1, arr2])    == np.concatenate([arr1, arr2], axis=2)
```

### Handling 1D Arrays in Concatenation

**Critical Issue:** 1D arrays cannot be directly concatenated with 2D arrays due to dimension mismatch.

```python
import numpy as np

A = np.array([1, 2, 3])        # Shape: (3,) - 1D
B = np.array([[4], [5]])       # Shape: (2, 1) - 2D
C = np.array([[1, 2, 3],       # Shape: (2, 3) - 2D
              [4, 5, 6]])

# ❌ Error: 1D with 2D
np.concatenate([A, B], axis=0)  # ValueError: all arrays must have same number of dimensions
np.concatenate([C, A], axis=0) # ValueError: all arrays must have same number of dimensions
```

**Solution:** Reshape 1D arrays to 2D before concatenation.

```python
# Reshape A to 2D column vector
A_2d = A.reshape(-1, 1)  # Shape: (3, 1)

# Now compatible with B (both 2D)
result = np.concatenate([A_2d, B], axis=0)  # Shape: (5, 1)
```

### Dimension Matching Rules for Concatenation

When concatenating along `axis=k`:

1. **All arrays must have the same number of dimensions**
2. **All dimensions except axis=k must match exactly**

**Example Analysis:**

Given:
- A: shape (m,) - 1D array
- B: shape (n, 1) - 2D column vector
- C: shape (n, m) - 2D matrix

**Option 1: `np.concatenate((C, B), axis=0)`** ❌
- C: (n, m), B: (n, 1)
- axis=0 requires same columns
- C has m columns, B has 1 column → mismatch

**Option 2: `np.concatenate((C, A), axis=0)`** ❌
- C: (n, m) is 2D, A: (m,) is 1D
- Different number of dimensions → error

**Option 3: `np.concatenate((C.T, A.reshape(m,1)), axis=1)`** ✓
- C.T: (m, n), A.reshape(m,1): (m, 1)
- axis=1 requires same rows
- Both have m rows → compatible
- Result: (m, n+1)

**Option 4: `np.concatenate((A, B), axis=0)`** ❌
- A: (m,) is 1D, B: (n, 1) is 2D
- Different number of dimensions → error

### Quick Checklist for Concatenation

Before calling `np.concatenate`:

1. **Check dimensions:** All arrays must have same `ndim`
2. **Check non-concatenation axes:** All dimensions except the concatenation axis must match
3. **Reshape if needed:** Use `.reshape()` to convert 1D to 2D
4. **Use transpose if needed:** Use `.T` to swap dimensions for compatibility

```python
# Common pattern: make 1D compatible with 2D
A_reshaped = A.reshape(1, -1)  # Row vector
A_reshaped = A.reshape(-1, 1)  # Column vector
```

### Stacking vs Concatenation

| Operation | New Axis | Shape Change | Use Case |
|-----------|----------|--------------|----------|
| `np.concatenate` | No | Extends existing axis | Adding more data along same dimension |
| `np.stack` | Yes | Creates new axis | Creating batches/tensors |

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate: extends existing dimension
concat_result = np.concatenate([arr1, arr2])
# Shape: (6,) - [1, 2, 3, 4, 5, 6]

# Stack: creates new dimension
stack_result = np.stack([arr1, arr2])
# Shape: (2, 3) - [[1, 2, 3], [4, 5, 6]]
```

---

## Advanced Indexing

### What It Is

NumPy allows using **arrays as indices**, not just single integers. This is called "fancy indexing" or "advanced indexing." It enables selecting multiple elements or sub-arrays at once.

### Case 1: Using 1D Array to Index 2D Array

When you use a 1D array to index a 2D array, NumPy selects **multiple rows**.

```python
import numpy as np

# Create identity matrix
identity = np.eye(5)
# [[1., 0., 0., 0., 0.],
#  [0., 1., 0., 0., 0.],
#  [0., 0., 1., 0., 0.],
#  [0., 0., 0., 1., 0.],
#  [0., 0., 0., 0., 1.]]

# Array of row indices
arr = np.array([1, 3, 4, 1, 2])

# Select rows using array as index
result = identity[arr]
# [[0., 1., 0., 0., 0.],  # row at index 1
#  [0., 0., 0., 1., 0.],  # row at index 3
#  [0., 0., 0., 0., 1.],  # row at index 4
#  [0., 1., 0., 0., 0.],  # row at index 1
#  [0., 0., 1., 0., 0.]]  # row at index 2
```

**Think of it as:** "Give me rows 1, 3, 4, 1, 2 from the identity matrix"

**Equivalent to:**
```python
np.array([
    identity[1],
    identity[3],
    identity[4],
    identity[1],
    identity[2]
])
```

### Case 2: Using Two Arrays for Indexing

You can use **two arrays** for indexing - one for rows, one for columns. This pairs up elements from both arrays.

```python
import numpy as np

# Create a zero matrix
one_hot = np.zeros((5, 5), dtype=int)

# Row indices
rows = np.arange(5)  # [0, 1, 2, 3, 4]

# Column indices
cols = np.array([1, 3, 4, 1, 2])

# Set 1 at paired positions
one_hot[rows, cols] = 1
# [[0, 1, 0, 0, 0],  # position (0, 1) = 1
#  [0, 0, 0, 1, 0],  # position (1, 3) = 1
#  [0, 0, 0, 0, 1],  # position (2, 4) = 1
#  [0, 1, 0, 0, 0],  # position (3, 1) = 1
#  [0, 0, 1, 0, 0]]  # position (4, 2) = 1
```

**Think of it as:** "Pair up the elements: (0,1), (1,3), (2,4), (3,1), (4,2) and set each to 1"

**Equivalent to:**
```python
one_hot[0, 1] = 1
one_hot[1, 3] = 1
one_hot[2, 4] = 1
one_hot[3, 1] = 1
one_hot[4, 2] = 1
```

### Simple Example with 1D Array

```python
arr = np.array([10, 20, 30, 40])
indices = np.array([0, 2, 3])

result = arr[indices]  # [10, 30, 40]
```

This selects elements at positions 0, 2, and 3 from the array.

### Practical Application: One-Hot Encoding

Advanced indexing is commonly used for one-hot encoding:

```python
def one_hot_encode(arr):
    num_classes = np.max(arr) + 1
    identity = np.eye(num_classes, dtype=int)
    return identity[arr]

arr = np.array([1, 3, 4, 1, 2])
result = one_hot_encode(arr)
# [[0, 1, 0, 0, 0],
#  [0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 1],
#  [0, 1, 0, 0, 0],
#  [0, 0, 1, 0, 0]]
```

### Boolean Indexing and Flattening

When you use boolean indexing on a 2D array, NumPy automatically flattens the result to 1D. It extracts only the elements where the condition is True and returns them as a 1D array.

```python
import numpy as np

matrix = np.array([[11, 22, 33],
                   [44, 55, 66],
                   [77, 88, 99]])

# Step 1: Create boolean mask (same shape as matrix)
mask = matrix % 2 == 0
# Result:
# [[False  True False]
#  [ True False  True]
#  [False  True False]]

# Step 2: Apply mask - NumPy extracts True positions and flattens
even_numbers = matrix[mask]
# Result: [22, 44, 66, 88] - 1D array
```

**Key points:**

- **Boolean indexing always returns a 1D array** regardless of the original shape
- **NumPy scans the array in row-major order** (left to right, top to bottom)
- **It only picks elements where the mask is True**
- **The result is automatically flattened**

**If you wanted to preserve shape:** You'd need to use `np.where` or fill with a placeholder value:

```python
# This preserves shape but uses 0 for odd numbers
result = np.where(matrix % 2 == 0, matrix, 0)
# [[ 0 22  0]
#  [44  0 66]
#  [ 0 88  0]]
```

### Key Points

- **Single array index**: Selects multiple elements or rows
- **Two array indices**: Pairs elements to set/access specific positions
- **Boolean arrays**: Can also be used for masking (selecting elements where condition is True)
- **Boolean indexing flattens**: Always returns 1D array regardless of original shape
- **Returns copy**: Advanced indexing always returns a new array (not a view)

---

## Summary

This reference guide covers:
- **Matrix Multiplication**: Dimension compatibility, computation rules, and properties
- **Broadcasting**: Rules for array shape compatibility with examples
- **np.vectorize**: When to use, performance considerations, and alternatives
- **Array Splitting**: Functions for splitting arrays along different axes
- **Understanding the Axis Parameter**: Critical distinction between reduction and building operations
- **Stacking Arrays**: Joining arrays along new axes with practical use cases
- **Array Concatenation**: Joining arrays along existing axes with shape compatibility rules
- **Advanced Indexing**: Using arrays as indices for fancy indexing

Use this guide as a quick reference for NumPy operations and their rules.
