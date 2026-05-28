# NumPy Cheatsheet for ML / DL Engineers
### Complete reference for experienced AI/ML engineers and senior data scientists

```python
import numpy as np
```

---

## Table of Contents

1. [Array Creation](#1-array-creation)
2. [Array Inspection & Memory](#2-array-inspection--memory)
3. [Indexing & Slicing](#3-indexing--slicing)
4. [Advanced Indexing](#4-advanced-indexing)
5. [Reshaping & Stacking](#5-reshaping--stacking)
6. [Math & Arithmetic](#6-math--arithmetic)
7. [Linear Algebra](#7-linear-algebra)
8. [Statistics & Aggregation](#8-statistics--aggregation)
9. [Random Module](#9-random-module)
10. [Boolean Operations & Masking](#10-boolean-operations--masking)
11. [Broadcasting](#11-broadcasting)
12. [Set Operations](#12-set-operations)
13. [Sorting & Searching](#13-sorting--searching)
14. [einsum & Tensor Operations](#14-einsum--tensor-operations)
15. [Padding, Tiling & Sliding Windows](#15-padding-tiling--sliding-windows)
16. [FFT & Signal Processing](#16-fft--signal-processing)
17. [File I/O](#17-file-io)
18. [Performance & Memory Optimization](#18-performance--memory-optimization)
19. [Critical ML Patterns](#19-critical-ml-patterns)
20. [Quick Lookup: ML Algorithm → NumPy Functions](#20-quick-lookup-ml-algorithm--numpy-functions)

---

## 1. Array Creation

```python
# From Python literals
np.array([1, 2, 3])                          # 1D array
np.array([[1, 2], [3, 4]])                   # 2D array (matrix)
np.array([1, 2, 3], dtype=np.float32)        # specify dtype (important for GPU)

# Filled arrays
np.zeros((3, 4))                             # all zeros, shape (3,4)
np.ones((2, 3))                              # all ones
np.full((3, 3), 7)                           # filled with 7
np.full((3, 3), np.inf)                      # initialize attention mask with inf
np.eye(4)                                    # 4×4 identity matrix
np.eye(4, k=1)                               # diagonal shifted by k
np.empty((2, 2))                             # uninitialized (fast, overwrite immediately)

# Ranges
np.arange(0, 10, 2)                          # [0, 2, 4, 6, 8]  like range()
np.arange(0, 1, 0.1, dtype=np.float32)       # float range
np.linspace(0, 1, 5)                         # [0, 0.25, 0.5, 0.75, 1.0] evenly spaced
np.logspace(0, 3, 4)                         # [1, 10, 100, 1000] log-spaced (LR search)
np.geomspace(1, 1000, 4)                     # geometric spacing

# From existing array
np.zeros_like(arr)                           # zeros with same shape/dtype as arr
np.ones_like(arr)
np.full_like(arr, fill_value=0.5)
np.empty_like(arr)
np.copy(arr)                                 # deep copy (avoids aliasing bugs)

# Triangular & diagonal
np.triu(A)                                   # upper triangular (attention masks!)
np.tril(A)                                   # lower triangular
np.triu(np.ones((n, n)), k=1)               # causal mask for transformers
np.diag(arr)                                 # 1D → diagonal matrix; 2D → extract diagonal
np.diagflat([1, 2, 3])                       # build diagonal matrix from values
np.trace(A)                                  # sum of diagonal elements

# Mesh grids (decision boundary, feature space plots)
xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                     np.linspace(-3, 3, 100))
# Flatten for model input: np.c_[xx.ravel(), yy.ravel()]
```

| Function             | ML Use Case                                           |
|----------------------|-------------------------------------------------------|
| `np.zeros / np.ones` | Initialize weight matrices, bias vectors              |
| `np.eye`             | Identity matrix in PCA, matrix inversion              |
| `np.zeros_like`      | Initialize gradients with same shape as weights       |
| `np.linspace`        | Plotting decision boundaries, learning rate schedules |
| `np.logspace`        | Log-scale hyperparameter grid search                  |
| `np.arange`          | Batch indices, epoch loops                            |
| `np.triu(ones, k=1)` | Causal / attention mask in Transformer decoder        |
| `np.meshgrid`        | Evaluate classifier over 2D feature grid              |
| `np.full(inf)`       | Initialize -inf mask before softmax attention         |

---

## 2. Array Inspection & Memory

```python
arr.shape          # (rows, cols, ...) — most important attribute in ML
arr.ndim           # number of dimensions
arr.size           # total number of elements
arr.dtype          # data type (float32, int64, ...)
arr.itemsize       # bytes per element
arr.nbytes         # total memory in bytes (arr.size * arr.itemsize)
arr.T              # transpose (swap last two axes for 2D)
arr.strides        # bytes to step in each dimension (stride tricks)
arr.flags          # C_CONTIGUOUS, F_CONTIGUOUS, WRITEABLE, etc.
len(arr)           # length of first dimension

# Numeric limits — crucial for numerical stability
np.finfo(np.float32).eps      # machine epsilon (~1.2e-7)
np.finfo(np.float32).max      # max representable float32
np.finfo(np.float16).eps      # ~0.001 — relevant in mixed-precision training
np.iinfo(np.int32).max        # max int32 value
np.iinfo(np.int64).max        # max int64 value

# Type promotion
np.result_type(np.float32, np.int32)   # → float64 (be careful!)
np.can_cast(np.float32, np.float16)    # check if safe cast
np.common_type(arr1, arr2)             # common scalar type

# Check contiguity (important for performance)
arr.flags['C_CONTIGUOUS']     # row-major (default)
arr.flags['F_CONTIGUOUS']     # column-major (Fortran-order)
np.ascontiguousarray(arr)     # force C-contiguous copy if needed
```

| Attribute      | ML Use Case                                            |
| -------------- | ------------------------------------------------------ |
| `.shape`       | Debug matrix dimension mismatches (most common ML bug) |
| `.dtype`       | Ensure float32 for GPU; float16 for mixed precision    |
| `.nbytes`      | Estimate memory before loading large datasets          |
| `.strides`     | Sliding window tricks, memory layout debugging         |
| `finfo.eps`    | Add epsilon to denominators to avoid divide-by-zero    |
| `C_CONTIGUOUS` | Required for efficient BLAS/cuBLAS operations          |

---

## 3. Indexing & Slicing

```python
# 1D
arr[0]             # first element
arr[-1]            # last element
arr[1:4]           # elements at index 1, 2, 3
arr[::2]           # every 2nd element
arr[::-1]          # reversed

# 2D — always [row, col]
arr[0, :]          # entire first row
arr[:, 1]          # entire second column
arr[0:2, 1:3]      # sub-matrix rows 0-1, cols 1-2
arr[[0, 2], :]     # rows 0 and 2 (fancy indexing — returns copy)
arr[:, [0, 3]]     # columns 0 and 3

# 3D+ (batch, channels, spatial) — common in DL
arr[0]             # first sample in batch
arr[:, 0, :]       # first channel, all spatial positions
arr[..., -1]       # last element along last axis (Ellipsis notation)
arr[0:32, ...]     # first 32 items across all remaining dims

# Boolean indexing
arr[arr > 0]                  # all positive values (returns 1D copy)
arr[arr == 0]                 # find zeros
mask = (arr > 0.3) & (arr < 0.7)
arr[mask]                     # values in range

# np.ix_ — open mesh for outer-product-style indexing
rows = np.array([0, 2])
cols = np.array([1, 3])
arr[np.ix_(rows, cols)]       # 2×2 sub-matrix at those row/col combos
```

| Pattern                | ML Use Case                                   |
|------------------------|-----------------------------------------------|
| `arr[:, 0]`            | Extract a single feature column               |
| `arr[0:batch_size]`    | Get one mini-batch                            |
| `arr[arr > threshold]` | Thresholding predictions                      |
| `arr[[idx1, idx2]]`    | Sampling specific rows for mini-batches       |
| `arr[..., -1]`         | Extract last element in arbitrary-rank tensor |
| `np.ix_(rows, cols)`   | Sub-matrix selection for covariance slicing   |

---

## 4. Advanced Indexing

```python
# Fancy indexing (always returns a COPY, not a view)
idx = np.array([3, 1, 0, 2])
arr[idx]                            # reorder rows by index

# Integer array indexing for gathering (like torch.gather concept)
rows = np.arange(batch_size)
pred_class = np.argmax(logits, axis=1)
selected = logits[rows, pred_class]  # gather predicted-class logits

# unravel_index / ravel_multi_index — convert between flat and ND indices
flat_idx = np.argmax(heatmap)                         # peak in flattened array
row, col = np.unravel_index(flat_idx, heatmap.shape)  # → 2D coordinates (CAM, heatmaps)

flat_idx = np.ravel_multi_index((row, col), arr.shape)  # 2D → flat index

# np.take — gather along axis (like advanced indexing, but axis-aware)
np.take(arr, indices=[0, 2], axis=0)   # rows 0 and 2 (same as arr[[0,2]])
np.take(arr, indices=[0, 2], axis=1)   # cols 0 and 2

# np.put — scatter values into flat positions (in-place)
np.put(arr, [0, 2], [99, 88])          # set arr.flat[0]=99, arr.flat[2]=88

# np.choose — select from array of choices per index
choices = [arr0, arr1, arr2]
idx = np.array([0, 1, 2, 0])
np.choose(idx, choices)               # pick from arr0, arr1, arr2 per element
```

| Pattern                     | ML Use Case                                            |
|-----------------------------|--------------------------------------------------------|
| `arr[rows, pred_class]`     | Gather class-specific logits for NLL loss              |
| `np.unravel_index`          | Find peak coordinates in heatmaps (CAM, Grad-CAM)      |
| `np.take(axis=1)`           | Column-wise feature selection by index array           |
| `np.ravel_multi_index`      | Convert ND position to flat buffer offset              |

---

## 5. Reshaping & Stacking

```python
# Reshape
arr.reshape(3, 4)              # reshape to 3 rows, 4 cols (total must match)
arr.reshape(-1, 1)             # column vector  (infer rows automatically)
arr.reshape(1, -1)             # row vector     (infer cols automatically)
arr.reshape(B, C, H*W)         # flatten spatial dims: (B,C,H,W) → (B,C,H*W)
arr.flatten()                  # collapse to 1D (returns copy)
arr.ravel()                    # collapse to 1D (returns view if possible, faster)

# Add/remove dimensions
arr[np.newaxis, :]             # add axis at position 0
np.expand_dims(arr, axis=0)    # same as above
np.expand_dims(arr, axis=(0, 1))  # add multiple axes at once
arr.squeeze()                  # remove all size-1 dimensions
arr.squeeze(axis=0)            # remove only axis 0 (safe for batch dim)

# Transpose & axis manipulation
arr.T                          # reverse axes (2D: rows↔cols)
np.transpose(arr, (0, 2, 1))   # permute specific axes: (B,H,W) → (B,W,H)
np.moveaxis(arr, source=1, destination=-1)  # move channel-first → channel-last
np.swapaxes(arr, 0, 1)         # swap two axes

# Stacking
np.concatenate([a, b], axis=0) # stack rows (vertically)
np.concatenate([a, b], axis=1) # stack cols (horizontally)
np.vstack([a, b])              # vertical stack   (row-wise), same as concat axis=0
np.hstack([a, b])              # horizontal stack (col-wise), same as concat axis=1
np.stack([a, b], axis=0)       # new axis stacking → adds a new dimension
np.dstack([a, b])              # depth stack (3rd axis)

# Splitting
np.split(arr, 3, axis=0)          # split into 3 equal parts along axis 0
np.array_split(arr, 3)            # split into 3 parts (allows unequal)
np.hsplit(arr, 2)                 # horizontal split
np.vsplit(arr, 2)                 # vertical split
```

| Function                    | ML Use Case                                             |
|-----------------------------|---------------------------------------------------------|
| `.reshape(-1, 1)`           | Convert 1D predictions to column vector for sklearn     |
| `.reshape(batch, -1)`       | Flatten image (28×28) → (784,) for Dense layer          |
| `np.expand_dims`            | Add batch dimension: (H,W,C) → (1,H,W,C)               |
| `.squeeze(axis=0)`          | Remove batch dim from single prediction safely          |
| `np.transpose(axes)`        | NCHW ↔ NHWC channel format conversion                   |
| `np.moveaxis`               | PyTorch (NCHW) ↔ TF/Keras (NHWC) format swap           |
| `np.concatenate`            | Combine train/val splits, merge feature sets            |
| `np.stack`                  | Create batch from list of samples                       |
| `np.split / array_split`    | Split dataset into k folds for cross-validation         |

---

## 6. Math & Arithmetic

```python
# Element-wise operations
np.exp(arr)                    # eˣ  → sigmoid, softmax
np.exp2(arr)                   # 2ˣ
np.log(arr)                    # natural log → cross-entropy loss
np.log2(arr)                   # log base 2  → information gain (Decision Tree)
np.log10(arr)                  # log base 10
np.log1p(arr)                  # log(1+x) → numerically stable for small x
np.expm1(arr)                  # exp(x)-1 → inverse of log1p
np.sqrt(arr)                   # square root → RMS, normalization
np.cbrt(arr)                   # cube root
np.abs(arr)                    # absolute value → MAE, L1 loss
np.square(arr)                 # x²  → MSE loss
np.power(arr, 3)               # xⁿ  → any power
np.reciprocal(arr)             # 1/x

# Clipping — crucial in DL
np.clip(arr, 0, 1)             # clamp values to [0, 1]
np.clip(arr, 1e-7, 1-1e-7)    # prevent log(0) in cross-entropy
np.clip(arr, -1, 1)            # gradient clipping range

# Rounding
np.round(arr, 4)               # round to 4 decimal places
np.floor(arr)                  # round down
np.ceil(arr)                   # round up
np.trunc(arr)                  # truncate toward zero
np.fix(arr)                    # same as trunc

# Signs & activations
np.sign(arr)                   # -1, 0, or +1
np.maximum(0, arr)             # ReLU
np.maximum(arr, 0.01 * arr)    # Leaky ReLU (α=0.01)
np.minimum(arr, 6)             # ReLU6 upper cap
np.where(arr > 0, arr, 0.01*arr)  # Leaky ReLU via where
np.where(cond, x, y)           # element-wise ternary

# Trigonometric (used in positional encoding)
np.sin(arr)
np.cos(arr)
np.sinh(arr)
np.cosh(arr)
np.tanh(arr)                   # tanh activation (RNN, LSTM gates)

# Numerical differences
np.diff(arr)                   # first-order differences (arr[i+1] - arr[i])
np.diff(arr, n=2)              # second-order differences
np.gradient(arr)               # numerical gradient (central differences)
np.gradient(loss_curve, lr)    # estimate local gradient of a curve
np.cumsum(arr)                 # cumulative sum
np.cumprod(arr)                # cumulative product
np.nancumsum(arr)              # cumsum ignoring NaN

# Outer, inner, cross products
np.outer(a, b)                 # outer product → (n,) × (m,) = (n,m)
np.inner(a, b)                 # inner product (= dot for 1D)
np.cross(a, b)                 # cross product (3D vectors)
np.kron(A, B)                  # Kronecker product
np.tensordot(A, B, axes=1)     # tensor dot (generalized matmul)

# Interpolation
np.interp(x, xp, fp)           # 1D linear interpolation
```

| Function              | ML Use Case                                        |
|-----------------------|----------------------------------------------------|
| `np.exp`              | Softmax, sigmoid activation, GMM density           |
| `np.log`              | Cross-entropy loss, log-likelihood                 |
| `np.log1p`            | Numerically stable log for count/price features    |
| `np.log2`             | Entropy in Decision Trees (ID3, C4.5)              |
| `np.clip`             | Prevent log(0); gradient clipping in RNNs          |
| `np.maximum(0, z)`    | ReLU activation                                    |
| `np.tanh`             | Activation in RNN/LSTM gates                       |
| `np.where`            | Leaky ReLU, conditional predictions, masking       |
| `np.outer`            | Hebbian weight updates, self-attention outer prod  |
| `np.diff / np.gradient` | Smoothness regularization, curve analysis        |
| `np.sin / np.cos`     | Sinusoidal positional encodings in Transformers    |
| `np.interp`           | LR schedule interpolation, calibration curves      |

---

## 7. Linear Algebra

```python
# Dot product & matrix multiply
np.dot(a, b)                   # dot product (1D) or matrix multiply (2D)
a @ b                          # matrix multiply — preferred syntax
np.matmul(a, b)                # same as @; does NOT broadcast scalars

# Batch matrix multiply
np.matmul(A, B)                # works on (B, m, k) @ (B, k, n) → (B, m, n)
np.einsum('bik,bkj->bij', A, B) # equivalent, more explicit

# Norms
np.linalg.norm(v)              # L2 norm (Euclidean length)
np.linalg.norm(v, ord=1)       # L1 norm
np.linalg.norm(v, ord=np.inf)  # L∞ norm (max absolute value)
np.linalg.norm(v, ord=-np.inf) # min absolute value
np.linalg.norm(M, axis=1)      # L2 norm of each ROW
np.linalg.norm(M, axis=0)      # L2 norm of each COLUMN
np.linalg.norm(M, ord='fro')   # Frobenius norm (√Σaᵢⱼ²) — default for matrices

# Matrix operations
np.linalg.inv(A)               # matrix inverse A⁻¹ (avoid when possible)
np.linalg.pinv(A)              # pseudo-inverse (for non-square or singular)
np.linalg.det(A)               # determinant (can overflow for large matrices)
np.linalg.slogdet(A)           # sign + log|det| — numerically stable!
np.linalg.matrix_rank(A)       # rank of matrix
np.linalg.cond(A)              # condition number (higher = more ill-conditioned)
np.trace(A)                    # sum of diagonal elements

# Decompositions
np.linalg.eig(A)               # eigenvalues & eigenvectors (may return complex)
np.linalg.eigh(A)              # for Hermitian/symmetric matrices — faster & stable
np.linalg.svd(A)               # SVD → U, S, Vt (full matrices)
np.linalg.svd(A, full_matrices=False)  # economy/thin SVD (cheaper)
np.linalg.qr(A)                # QR decomposition
np.linalg.cholesky(A)          # Cholesky: A = L @ L.T  (A must be pos. definite)
np.linalg.matrix_power(A, n)   # A^n — matrix power

# Solving systems
np.linalg.solve(A, b)          # solve Ax = b — faster than inv(A) @ b
np.linalg.lstsq(A, b, rcond=None)  # least-squares solution (overdetermined)

# Low-rank approximation via truncated SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = 50  # keep top-k singular values
A_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
```

| Function                   | ML Use Case                                              |
|----------------------------|----------------------------------------------------------|
| `a @ b`                    | Forward pass (X @ W), attention scores (Q @ K.T)        |
| `np.matmul` (batched)      | Batch attention: (B,H,L,d) × (B,H,d,L)                  |
| `np.linalg.norm`           | L2 normalization, cosine similarity denominator          |
| `np.linalg.norm('fro')`    | Weight regularization, weight matrix scale monitoring    |
| `np.linalg.inv`            | Normal equation: θ = (XᵀX)⁻¹Xᵀy                         |
| `np.linalg.eigh`           | PCA — fast path for symmetric covariance matrix          |
| `np.linalg.svd`            | SVD-based PCA, matrix factorization for RecSys           |
| `np.linalg.svd` truncated  | Low-rank approximation, latent semantic analysis         |
| `np.linalg.cholesky`       | Sampling from multivariate Gaussians, GP inference       |
| `np.linalg.solve`          | Linear regression normal equation (numerically stable)   |
| `np.linalg.slogdet`        | Log-determinant in Gaussian log-likelihood               |
| `np.linalg.cond`           | Diagnose ill-conditioned systems (exploding gradients)   |
| `np.linalg.lstsq`          | Overdetermined systems, OLS, ridge regression base       |
| `np.trace`                 | Nuclear norm regularization, KL divergence               |

---

## 8. Statistics & Aggregation

```python
# Basic stats
np.mean(arr)                   # arithmetic mean
np.mean(arr, axis=0)           # column-wise mean
np.mean(arr, axis=1)           # row-wise mean
np.median(arr)                 # median
np.std(arr)                    # standard deviation (population, ddof=0)
np.std(arr, ddof=1)            # sample standard deviation (unbiased, ddof=1)
np.var(arr)                    # variance
np.var(arr, ddof=1)            # sample variance
np.sum(arr)                    # sum of all elements
np.sum(arr, axis=0)            # column-wise sum
np.prod(arr)                   # product of all elements
np.cumsum(arr)                 # cumulative sum
np.cumsum(arr, axis=0)         # cumulative sum along axis

# Min / Max
np.min(arr)                    # minimum value
np.max(arr)                    # maximum value
np.argmin(arr)                 # INDEX of minimum value (flat if no axis)
np.argmax(arr)                 # INDEX of maximum value  ← very common in ML
np.argmax(arr, axis=1)         # predicted class per row (after softmax)
np.nanargmin(arr)              # argmin ignoring NaN
np.nanargmax(arr)              # argmax ignoring NaN

# Sorting
np.sort(arr)                   # sorted values (ascending by default)
np.sort(arr)[::-1]             # descending (copy, then reverse)
np.argsort(arr)                # indices that would sort arr
np.argsort(arr)[::-1]          # indices sorted descending (top-K)
np.argsort(arr, axis=1)        # sort each row independently
np.lexsort((secondary, primary))  # multi-key sort (last key = primary)
np.partition(arr, k)           # partial sort — O(n) vs O(n log n)
np.argpartition(arr, -k)[-k:]  # indices of top-k values (faster than argsort)

# Percentile / Quantile
np.percentile(arr, 25)                # 25th percentile (Q1)
np.percentile(arr, [25, 50, 75])      # Q1, median, Q3
np.quantile(arr, 0.95)                # 95th quantile (same as percentile/100)
np.percentile(arr, 95, axis=0)        # column-wise 95th percentile
np.nanpercentile(arr, 50)             # median ignoring NaN

# Counts & histograms
np.bincount(int_arr)                  # count occurrences of each integer (fast!)
np.bincount(labels, weights=scores)   # weighted bincount (soft label sums)
np.unique(arr)                        # unique values
np.unique(arr, return_counts=True)    # unique values + their counts
np.unique(arr, return_inverse=True)   # + inverse mapping (label encode!)
np.unique(arr, return_index=True)     # + first occurrence indices
np.histogram(arr, bins=10)            # bin edges + counts
np.histogram2d(x, y, bins=10)        # 2D histogram (joint distribution)
np.digitize(arr, bins)               # assign each element to a bin index

# NaN-safe versions
np.nanmean(arr)
np.nanstd(arr)
np.nansum(arr)
np.nanmax(arr)
np.nanmin(arr)
np.nanvar(arr)

# Covariance & Correlation
np.cov(X.T)                    # covariance matrix  ← PCA
np.corrcoef(X.T)               # Pearson correlation matrix

# Weighted statistics
np.average(arr, weights=w)     # weighted mean
```

| Function                    | ML Use Case                                           |
|-----------------------------|-------------------------------------------------------|
| `np.mean(axis=0)`           | Feature means for standardization (compute on train)  |
| `np.std(axis=0, ddof=1)`    | Unbiased feature std for standardization              |
| `np.argmax(axis=1)`         | Convert softmax output to class label                 |
| `np.argpartition(-k)[-k:]`  | Top-K selection — O(n) instead of O(n log n)          |
| `np.argsort()[::-1]`        | Top-K recommendations, feature importance ranking     |
| `np.bincount`               | Fast label frequency count (classification)           |
| `np.unique(return_inverse)` | Label encoding in one call                            |
| `np.unique(return_counts)`  | Class distribution analysis                           |
| `np.histogram`              | Activation/weight distribution visualization          |
| `np.digitize`               | Bin continuous predictions for calibration plots      |
| `np.percentile(95)`         | Outlier detection thresholding                        |
| `np.cov`                    | Covariance matrix in PCA, GMM, LDA                    |
| `np.average(weights)`       | Weighted evaluation metrics                           |
| `np.nanmean`                | Stats on data with missing values                     |

---

## 9. Random Module

```python
# Recommended: seeded Generator (reproducible, faster than legacy API)
rng = np.random.default_rng(seed=42)

# Uniform distributions
rng.random((3, 4))                        # uniform [0, 1)
rng.uniform(low=-1, high=1, size=(3, 3))  # uniform between low and high
rng.integers(0, 10, size=5)              # random integers [0, 10)

# Gaussian / Normal distributions
rng.standard_normal((3, 3))              # standard normal μ=0, σ=1
rng.normal(loc=0, scale=1, size=(3, 3)) # Gaussian with μ, σ
rng.multivariate_normal(mean, cov, size=100)  # multivariate Gaussian (GMM, VAE)

# Other continuous distributions
rng.exponential(scale=1.0, size=100)     # exponential (inter-arrival times)
rng.beta(a=0.4, b=0.4, size=100)         # Beta (MixUp augmentation α~Beta)
rng.gamma(shape=2.0, scale=1.0, size=100)
rng.uniform(0, 1, size=(n,)) < dropout_rate  # dropout mask
rng.laplace(loc=0, scale=1.0, size=100)  # Laplace (L1-noise)

# Discrete distributions
rng.binomial(n=1, p=0.5, size=100)       # Bernoulli flips
rng.multinomial(n=1, pvals=probs)        # Gumbel softmax trick base
rng.poisson(lam=3.0, size=100)           # Poisson (count data)
rng.dirichlet(alpha=[0.5]*K, size=100)   # Dirichlet (LDA topic priors, MixUp)

# Sampling
rng.choice(arr, size=5)                  # with replacement
rng.choice(arr, size=5, replace=False)   # without replacement
rng.choice(n, size=batch_size)           # sample indices (mini-batch)
rng.permutation(n)                       # random permutation of 0..n-1
rng.permutation(arr)                     # shuffle copy of array
rng.shuffle(arr)                         # in-place shuffle (modifies arr)

# Legacy API (still widely used in older codebases)
np.random.seed(42)
np.random.randn(3, 4)          # standard normal
np.random.rand(3, 4)           # uniform [0,1)
np.random.randint(0, 10, (3,)) # random integers
np.random.permutation(n)       # random permutation

# Reproducibility best practice
rng = np.random.default_rng(seed=42)
# Pass rng as argument to functions, do NOT use global state in libraries
```

| Function                      | ML Use Case                                          |
|-------------------------------|------------------------------------------------------|
| `rng.standard_normal`         | Weight initialization (Xavier, He init)              |
| `rng.multivariate_normal`     | Sample from Gaussian prior (VAE, GMM)                |
| `rng.beta(α, α)`              | MixUp data augmentation coefficient λ~Beta(α,α)      |
| `rng.dirichlet`               | LDA topic distribution priors, label smoothing       |
| `rng.random() < p`            | Dropout mask (Bernoulli sampling)                    |
| `rng.choice(n, batch_size)`   | Mini-batch index sampling                            |
| `rng.permutation`             | Shuffle data before splitting train/val              |
| `rng.shuffle`                 | In-place dataset shuffle each epoch                  |
| `default_rng(seed)`           | Reproducible experiments; preferred over legacy API  |

---

## 10. Boolean Operations & Masking

```python
# Comparisons → return boolean arrays
arr > 0                        # element-wise greater than
arr == 1                       # element-wise equality
arr != 0                       # not equal

# Combine conditions (use & | ~, NOT Python 'and'/'or')
(arr > 0) & (arr < 10)         # AND
(arr < 0) | (arr > 10)         # OR
~(arr > 0)                     # NOT

# Check conditions
np.any(arr > 0)                # True if ANY element satisfies
np.all(arr > 0)                # True if ALL elements satisfy
np.any(arr > 0, axis=0)        # column-wise any
np.all(arr > 0, axis=1)        # row-wise all

# Count Trues
np.sum(arr > 0)                # count of True (True=1, False=0)
np.mean(arr > 0)               # fraction of True values

# Find indices
np.where(condition)                  # tuple of index arrays where True
np.where(arr > 0, arr, 0)            # conditional replace (ternary)
np.where(arr > 0, arr, 0.01*arr)     # Leaky ReLU
np.nonzero(arr)                      # indices of non-zero elements (same as where(arr))
np.flatnonzero(arr > 0)              # flat indices of True elements
np.argwhere(arr > 0)                 # Nx1 array of index tuples

# NaN & Inf handling
np.isnan(arr)                  # boolean mask of NaN positions
np.isinf(arr)                  # mask of +/-inf positions
np.isfinite(arr)               # True where not NaN and not Inf
np.isneginf(arr)               # negative infinity
np.isposinf(arr)               # positive infinity
np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)  # replace special values
np.nanmean(arr)                # mean ignoring NaN
np.nansum(arr)                 # sum ignoring NaN

# Masking in attention mechanisms
causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
scores[causal_mask] = -np.inf  # mask future positions before softmax
```

| Function                  | ML Use Case                                       |
|---------------------------|---------------------------------------------------|
| `arr > threshold`         | Convert probability to binary prediction          |
| `np.where(cond, a, b)`    | ReLU, Leaky ReLU, conditional masking             |
| `np.any / np.all`         | Early stopping condition checks                   |
| `np.mean(arr > 0)`        | Compute accuracy in one line: `(pred==y).mean()`  |
| `np.isnan`                | Detect NaN in gradients (exploding gradient check)|
| `np.nan_to_num`           | Safe gradient clipping / special value handling   |
| `np.flatnonzero`          | Find indices of non-zero sparse features          |
| `causal_mask` pattern     | Autoregressive attention in Transformers          |

---

## 11. Broadcasting

```python
# Broadcasting: operate on arrays of different shapes
# Rule: dimensions are compatible if equal OR one of them is 1.

# Subtract column mean from each column
X - X.mean(axis=0)             # (100,5) - (5,) → works; (5,) broadcast to (100,5)

# Normalize each row (keepdims is crucial)
X / X.sum(axis=1, keepdims=True)   # keepdims: (100,1) not (100,) → broadcasts correctly
X / X.norm(axis=1, keepdims=True)  # L2-normalize each sample

# Standard patterns
arr - arr.mean()               # zero-center all data
arr / arr.std()                # standardize
(arr - arr.min()) / (arr.max() - arr.min())  # min-max normalize to [0,1]

# Numerically stable softmax
z = z - z.max(axis=1, keepdims=True)   # subtract max (broadcast over cols)
exp_z = np.exp(z)
probs = exp_z / exp_z.sum(axis=1, keepdims=True)

# Pairwise distances (N×M without loops)
# X: (N, D), Y: (M, D)
diffs = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (N, M, D)
dists = np.linalg.norm(diffs, axis=-1)              # (N, M)

# Cosine similarity matrix
X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)   # (N, D)
Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)   # (M, D)
cos_sim = X_norm @ Y_norm.T                             # (N, M)

# Bias addition in neural net (feature dim at axis=-1)
Z = X @ W + b          # b: (out_features,) broadcasts over batch
```

| Pattern                                  | ML Use Case                                     |
|------------------------------------------|-------------------------------------------------|
| `X - mean`, `X / std`                    | Z-score standardization                         |
| `keepdims=True`                          | Softmax, batch normalization, row-wise ops      |
| `X - X.max(axis=1, keepdims=True)`       | Numerically stable softmax (subtract max)       |
| `X[:, np.newaxis, :] - Y[np.newaxis, :]` | Pairwise distance matrix (K-Means, KNN)         |
| `X_norm @ Y_norm.T`                      | Cosine similarity matrix (retrieval, ANN)       |

---

## 12. Set Operations

```python
# 1D set operations
np.unique(arr)                           # unique elements (sorted)
np.intersect1d(a, b)                     # common elements
np.union1d(a, b)                         # all unique elements from both
np.setdiff1d(a, b)                       # elements in a but not in b
np.setxor1d(a, b)                        # elements in a or b but not both
np.in1d(arr, test_values)                # membership test → bool array
np.isin(arr, test_values)                # same but works on N-D arrays
np.isin(arr, test_values, invert=True)   # NOT in test_values

# Example: find OOV tokens (not in vocabulary)
oov_mask = ~np.isin(token_ids, vocab_ids)
```

| Function           | ML Use Case                                         |
|--------------------|-----------------------------------------------------|
| `np.unique`        | Get class labels, vocabulary tokens                 |
| `np.intersect1d`   | Find common features between two feature sets       |
| `np.isin`          | Mask out-of-vocabulary tokens, filter invalid IDs   |
| `np.setdiff1d`     | Find missing classes, split train/test by ID        |

---

## 13. Sorting & Searching

```python
# Sorting
np.sort(arr)                   # returns sorted copy
np.sort(arr, axis=0)           # sort each column
np.sort(arr, axis=1)           # sort each row
arr.sort()                     # in-place sort (modifies arr)
np.argsort(arr)                # indices that sort arr (stable by default)
np.argsort(arr, kind='stable') # guaranteed stable sort (preserve order of ties)

# Partial sort — O(n) — much faster for top-K
np.partition(arr, kth=3)            # arr[3] is the 4th smallest; rest unordered
np.argpartition(arr, kth=-k)[-k:]   # INDICES of k largest (unordered)
np.argpartition(arr, kth=k)[:k]     # indices of k smallest

# Multi-key sort
np.lexsort((col2, col1))       # sort by col1 first, then col2 (last = primary)

# Searching
np.searchsorted(sorted_arr, v)         # binary search: index to insert v (left)
np.searchsorted(sorted_arr, v, side='right')  # right insertion point
# Use case: map continuous scores to discrete bins efficiently

# Example: fast top-K retrieval
scores = model_scores  # (N,)
top_k_idx = np.argpartition(scores, -K)[-K:]            # O(N)
top_k_idx_sorted = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]  # sort just top-K
```

| Function                       | ML Use Case                                        |
|--------------------------------|----------------------------------------------------|
| `np.argsort(kind='stable')`    | Rank-based metrics (MAP, NDCG), stable ranking     |
| `np.argpartition(-k)[-k:]`     | Top-K recommendation/retrieval — O(n), not O(nlogn)|
| `np.lexsort`                   | Sort by multiple criteria (e.g., class then score) |
| `np.searchsorted`              | Efficient bin assignment, calibration, quantization|

---

## 14. einsum & Tensor Operations

```python
# np.einsum — Einstein summation, extremely powerful
# Syntax: 'input_subscripts->output_subscripts'

# Matrix multiply: (m,k) × (k,n) → (m,n)
np.einsum('ik,kj->ij', A, B)          # same as A @ B

# Batch matrix multiply: (B,m,k) × (B,k,n) → (B,m,n)
np.einsum('bik,bkj->bij', A, B)

# Dot product of two vectors: (n,) · (n,) → scalar
np.einsum('i,i->', a, b)              # same as np.dot(a, b)

# Outer product: (m,) × (n,) → (m,n)
np.einsum('i,j->ij', a, b)           # same as np.outer(a, b)

# Trace: sum diagonal elements
np.einsum('ii->', A)                  # same as np.trace(A)

# Element-wise multiply then sum (useful for attention)
np.einsum('ij,ij->', A, B)           # sum of element-wise product

# Multi-head attention scores: (B,H,L,d) × (B,H,d,L) → (B,H,L,L)
np.einsum('bhid,bhjd->bhij', Q, K)   # Q·Kᵀ without explicit transpose

# Bilinear form: xᵀAy
np.einsum('i,ij,j->', x, A, y)

# Weighted sum over sequence: (B,L) × (B,L,d) → (B,d)
np.einsum('bl,bld->bd', weights, values)   # attention-weighted sum

# np.tensordot
np.tensordot(A, B, axes=1)           # contract last axis of A with first of B
np.tensordot(A, B, axes=([1,2],[0,1]))  # contract specific axes

# Element-wise ops with np.multiply (explicit)
np.multiply(A, B)                    # A * B element-wise
np.divide(A, B)                      # A / B element-wise
```

| Pattern                          | ML Use Case                                          |
|----------------------------------|------------------------------------------------------|
| `einsum('bik,bkj->bij')`         | Batched matmul for attention, RNN                    |
| `einsum('bhid,bhjd->bhij')`      | Multi-head self-attention score computation          |
| `einsum('bl,bld->bd')`           | Attention-weighted context vector                    |
| `einsum('i,j->ij')`              | Outer product — key/query rank-1 updates             |
| `einsum('ij,ij->')`              | Frobenius inner product of two matrices              |
| `np.tensordot`                   | Generalized tensor contractions in custom layers     |

---

## 15. Padding, Tiling & Sliding Windows

```python
# np.pad — add padding around array
np.pad(arr, pad_width=1, mode='constant', constant_values=0)  # zero-pad all sides
np.pad(arr, ((1,1),(2,2)), mode='constant')   # asymmetric padding (top,bot),(left,right)
np.pad(arr, 1, mode='reflect')                # reflect padding (used in convolutions)
np.pad(arr, 1, mode='edge')                   # replicate edge values
np.pad(arr, 1, mode='wrap')                   # circular/toroidal padding

# Padding for conv "same" mode: input (H,W), kernel k, dilation d
pad = (k - 1) * d // 2
padded = np.pad(arr, ((0,0),(pad,pad),(pad,pad),(0,0)))  # NHWC

# np.tile — repeat array N times
np.tile(arr, 3)                # repeat 3 times along axis 0
np.tile(arr, (2, 3))           # repeat 2× along axis 0, 3× along axis 1
np.tile(arr, (B, 1, 1))        # expand to batch: (H,W) → (B,H,W)

# np.repeat — repeat each element N times
np.repeat(arr, 3)              # repeat each element 3 times (flattened)
np.repeat(arr, 3, axis=0)      # repeat each row 3 times
np.repeat(arr, [2,3,1], axis=0)  # different repeat count per element

# Sliding windows with stride tricks (zero-copy view)
from numpy.lib.stride_tricks import sliding_window_view  # NumPy 1.20+
windows = sliding_window_view(arr, window_shape=(window_size,))  # 1D
# Returns (N - W + 1, W) — each row is a window

# 2D sliding windows for patch extraction
patches = sliding_window_view(image, window_shape=(patch_h, patch_w))
# (H-ph+1, W-pw+1, ph, pw)

# Manual stride tricks (for older NumPy or custom strides)
from numpy.lib.stride_tricks import as_strided
shape  = (n_windows, window_size)
strides = (arr.strides[0], arr.strides[0])
windows = as_strided(arr, shape=shape, strides=strides)  # CAUTION: no bounds check
```

| Function                       | ML Use Case                                            |
|--------------------------------|--------------------------------------------------------|
| `np.pad(constant, 0)`          | Zero-padding for "same" convolutions                   |
| `np.pad(reflect)`              | Reflection padding for conv to preserve edge stats     |
| `np.tile(B, 1, 1)`             | Expand a single sample to a batch                      |
| `np.repeat(axis=0)`            | Upsample feature maps, oversample minority class       |
| `sliding_window_view`          | Efficient rolling window features for time series      |
| `sliding_window_view` (2D)     | Image patch extraction for ViT, patch-based methods    |

---

## 16. FFT & Signal Processing

```python
# 1D Fast Fourier Transform (audio, time-series ML)
fft_vals = np.fft.fft(signal)           # complex spectrum
freqs    = np.fft.fftfreq(n, d=1/sr)   # frequency bins (sr = sample rate)
magnitudes = np.abs(fft_vals)           # power spectrum
phase      = np.angle(fft_vals)         # phase spectrum

# Real FFT (signal is real → ~2× faster, half the bins)
rfft_vals  = np.fft.rfft(signal)                   # (n//2 + 1,) complex
rfreqs     = np.fft.rfftfreq(n, d=1/sr)            # positive frequencies only
power      = np.abs(rfft_vals)**2                   # power spectral density

# Inverse FFT
reconstructed = np.fft.irfft(rfft_vals, n=n)       # back to time domain

# 2D FFT (image processing, texture analysis)
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)         # center zero-frequency component

# Short-time Fourier (manual, for spectrograms in audio ML)
# Use scipy.signal.stft for production; this shows the concept:
frame_len, hop = 512, 128
n_frames = (len(signal) - frame_len) // hop + 1
frames = np.array([signal[i*hop : i*hop+frame_len]
                   for i in range(n_frames)])
window = np.hanning(frame_len)
stft = np.fft.rfft(frames * window, axis=1)   # (n_frames, freq_bins)

np.fft.fftshift(arr)               # shift zero-frequency component to center
np.fft.ifftshift(arr)              # inverse fftshift
np.fft.fftfreq(n, d=1.0)           # frequency bin centers
```

| Function             | ML Use Case                                              |
|----------------------|----------------------------------------------------------|
| `np.fft.rfft`        | Audio feature extraction (power spectrum)                |
| `np.abs(fft)**2`     | Mel spectrogram base, speaker identification             |
| `np.fft.fft2`        | Frequency-domain convolution, texture features           |
| `np.fft.fftshift`    | Visualize frequency spectrum in image analysis           |
| manual STFT frames   | Custom spectrogram for audio ML models                   |

---

## 17. File I/O

```python
# NumPy binary format (.npy / .npz) — fastest, preserves dtype/shape
np.save('array.npy', arr)               # single array
arr = np.load('array.npy')              # load single array

np.savez('arrays.npz', X=X, y=y)        # multiple arrays, uncompressed
np.savez_compressed('arrays.npz', X=X, y=y)  # compressed (smaller file)
data = np.load('arrays.npz')
X, y = data['X'], data['y']

# Memory-mapped arrays — access large arrays without loading into RAM
arr = np.load('large.npy', mmap_mode='r')   # read-only mmap
arr = np.memmap('large.dat', dtype='float32',
                mode='r+', shape=(N, D))     # read-write mmap

# Text / CSV loading (slower than binary; prefer pandas for CSVs)
np.loadtxt('data.csv', delimiter=',', skiprows=1)   # skip header
np.genfromtxt('data.csv', delimiter=',',
              names=True, dtype=None)               # handle missing, auto dtype
np.savetxt('output.csv', arr, delimiter=',', fmt='%.4f', header='a,b,c')

# From bytes / buffers
np.frombuffer(buffer, dtype=np.float32)   # parse raw bytes (protobuf, sockets)
np.frompyfunc(func, nin=1, nout=1)        # vectorize Python function to ufunc
```

| Function                    | ML Use Case                                             |
|-----------------------------|---------------------------------------------------------|
| `np.save / np.load`         | Cache preprocessed arrays (embeddings, features)        |
| `np.savez_compressed`       | Store train/test splits efficiently                     |
| `np.memmap`                 | Train on datasets larger than RAM                       |
| `mmap_mode='r'`             | Zero-copy access to large embedding matrices            |
| `np.frombuffer`             | Parse binary sensor data, custom data formats           |

---

## 18. Performance & Memory Optimization

```python
# dtype downcasting — reduce memory usage
arr.astype(np.float32)         # float64 → float32 (2× memory savings)
arr.astype(np.float16)         # float32 → float16 (used in mixed precision)
arr.astype(np.int8)            # quantization to int8 (model compression)

# Check if view or copy
np.shares_memory(a, b)         # True if they share memory (view vs copy)

# Vectorization over Python loops — always prefer
# Bad:
result = [np.exp(x) for x in arr]          # slow Python loop
# Good:
result = np.exp(arr)                        # vectorized ufunc

# np.vectorize — wrap Python functions (convenience, not performance!)
# Note: np.vectorize is syntactic sugar; still a Python loop internally
vfunc = np.vectorize(custom_python_fn)
vfunc(arr)

# np.frompyfunc — true ufunc from Python func, slightly faster
ufunc = np.frompyfunc(lambda x: x**2 + 1, 1, 1)

# In-place ops — avoid creating intermediate arrays
np.add(a, b, out=a)            # a += b without allocating new array
np.multiply(a, b, out=a)       # a *= b
np.exp(arr, out=arr)           # in-place exp

# Contiguity — ensure for BLAS operations
np.ascontiguousarray(arr)      # C-order (row-major) — for most ops
np.asfortranarray(arr)         # F-order (column-major) — for LAPACK

# Avoid unnecessary copies
arr.view(np.float32)           # reinterpret dtype without copy (use carefully)
arr.T                          # transpose is a VIEW (no copy)
arr.reshape(-1)                # reshape is often a VIEW

# Numeric precision checks
np.allclose(a, b, rtol=1e-5, atol=1e-8)   # compare floats with tolerance
np.isclose(a, b)                           # element-wise allclose

# Structured arrays — store mixed-type tabular data in NumPy
dtype = np.dtype([('name', 'U20'), ('age', np.int32), ('score', np.float32)])
structured = np.array([('Alice', 30, 0.95), ('Bob', 25, 0.88)], dtype=dtype)
structured['name']    # access field like dict
structured['score']   # → array of scores

# Memory layout inspection
arr.strides                    # bytes per step in each dim
arr.flags                      # C_CONTIGUOUS, F_CONTIGUOUS, WRITEABLE, OWNDATA
```

| Technique                     | ML Use Case                                              |
|-------------------------------|----------------------------------------------------------|
| `.astype(float32)`            | GPU training — PyTorch/TF default dtype                  |
| `.astype(float16)`            | Mixed-precision training (AMP)                           |
| `.astype(int8)`               | Post-training quantization (model compression)           |
| `np.allclose`                 | Unit test gradients, verify numerical implementations    |
| `np.add(out=a)`               | Gradient accumulation without extra allocations          |
| `np.ascontiguousarray`        | Fix stride layout before passing to C extensions         |
| `np.memmap`                   | Dataset larger than RAM; zero-copy inference             |
| `np.shares_memory`            | Debug unexpected in-place modifications                  |

---

## 19. Critical ML Patterns

### Train/Test Split (manual, stratified-aware)

```python
# Simple shuffle split
idx = np.random.default_rng(42).permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = idx[:split], idx[split:]
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Stratified split (preserve class balance)
classes = np.unique(y)
train_idx, val_idx = [], []
rng = np.random.default_rng(42)
for cls in classes:
    cls_idx = np.where(y == cls)[0]
    rng.shuffle(cls_idx)
    split = int(0.8 * len(cls_idx))
    train_idx.extend(cls_idx[:split])
    val_idx.extend(cls_idx[split:])
train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)
```

### Standardization (fit on train, transform both)

```python
mean = X_train.mean(axis=0)           # compute ONLY on train set
std  = X_train.std(axis=0, ddof=1)    # unbiased std
std[std == 0] = 1                      # avoid divide-by-zero for constant features

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std      # use TRAIN stats — no data leakage
X_test  = (X_test  - mean) / std
```

### Mini-Batch Generator

```python
def get_batches(X, y, batch_size, shuffle=True, rng=None):
    rng = rng or np.random.default_rng()
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), batch_size):
        batch_idx = idx[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]
```

### Numerically Stable Softmax

```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)    # subtract max for stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

### Log-Softmax (more stable for NLL loss)

```python
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
# More stable version using logsumexp:
def log_softmax_v2(z):
    z = z - z.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
    return z - log_sum_exp
```

### One-Hot Encode Labels

```python
def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1
    return oh
```

### Cross-Entropy Loss (numerically stable)

```python
def cross_entropy(y_true_oh, y_pred_probs):
    y_pred_probs = np.clip(y_pred_probs, 1e-7, 1 - 1e-7)   # prevent log(0)
    return -np.mean(np.sum(y_true_oh * np.log(y_pred_probs), axis=1))

# From logits (preferred — more numerically stable)
def cross_entropy_from_logits(y_true_idx, logits):
    log_probs = log_softmax(logits)                           # stable log-softmax
    return -np.mean(log_probs[np.arange(len(y_true_idx)), y_true_idx])
```

### Cosine Similarity Matrix (vectorized)

```python
def cosine_similarity_matrix(A, B):
    # A: (N, D), B: (M, D) → returns (N, M)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T
```

### MixUp Data Augmentation

```python
def mixup(X, y_oh, alpha=0.4, rng=None):
    rng = rng or np.random.default_rng()
    lam = rng.beta(alpha, alpha, size=len(X))[:, None]
    idx = rng.permutation(len(X))
    X_mixed = lam * X + (1 - lam) * X[idx]
    y_mixed = lam * y_oh + (1 - lam) * y_oh[idx]
    return X_mixed, y_mixed
```

### Pairwise Euclidean Distances (no loop)

```python
def pairwise_distances(A, B):
    # A: (N, D), B: (M, D) → (N, M)
    # ||a - b||² = ||a||² + ||b||² - 2a·b
    A_sq = np.sum(A**2, axis=1, keepdims=True)    # (N, 1)
    B_sq = np.sum(B**2, axis=1, keepdims=True).T  # (1, M)
    return np.sqrt(np.maximum(A_sq + B_sq - 2 * A @ B.T, 0))
```

### K-Fold Cross-Validation Indices

```python
def kfold_indices(n, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx
```

### Gradient Numerical Check

```python
def numerical_gradient(f, x, eps=1e-5):
    """Verify analytical gradients with finite differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        orig = x[ix]
        x[ix] = orig + eps;  fplus  = f(x)
        x[ix] = orig - eps;  fminus = f(x)
        grad[ix] = (fplus - fminus) / (2 * eps)
        x[ix] = orig
        it.iternext()
    return grad
```

---

## 20. Quick Lookup: ML Algorithm → NumPy Functions

| Algorithm             | Key NumPy Functions                                              |
|-----------------------|------------------------------------------------------------------|
| Linear Regression     | `@`, `linalg.solve`, `linalg.lstsq`, `linalg.inv`, `mean`, `std`|
| Logistic Regression   | `exp`, `log`, `clip`, `@`, `where`, `log_softmax`               |
| Softmax Regression    | `exp`, `log`, `clip`, `einsum`, `argmax`                        |
| K-Means               | `linalg.norm`, `argmin`, `mean`, `expand_dims`, pairwise dists  |
| KNN                   | `argsort`, pairwise dists, `argpartition` (top-K)               |
| PCA                   | `linalg.eigh`, `linalg.svd`, `cov`, `argsort`, `cumsum`         |
| LDA                   | `linalg.eigh`, `cov`, `mean`, `linalg.solve`                    |
| Decision Tree         | `log2`, `unique`, `sum`, `where`, `bincount`                    |
| Naive Bayes           | `exp`, `log`, `sum`, `mean`, `var`                              |
| SVM                   | `dot`, `linalg.norm`, `clip`, `sign`, `outer`                   |
| K-Fold CV             | `array_split`, `permutation`, `concatenate`                     |
| Neural Network        | `@`, `exp`, `maximum`, `clip`, `log`, `tanh`, `einsum`          |
| Transformer Attention | `einsum`, `triu`, `softmax`, `sqrt`, `matmul` (batched)         |
| GMM / EM              | `linalg.det`, `linalg.inv`, `linalg.cholesky`, `exp`, `log`     |
| SVD / RecSys          | `linalg.svd`, `@`, `linalg.norm`, truncated SVD                 |
| TF-IDF                | `log`, `linalg.norm`, `@`, `zeros`, `unique`                    |
| Cosine Similarity     | `dot`, `linalg.norm`, `@`, einsum                               |
| MixUp Augmentation    | `beta`, `permutation`, broadcasting                             |
| Gradient Check        | `linalg.norm`, `allclose`, numerical difference                 |
| LSTM / RNN            | `tanh`, `sigmoid`, `@`, `clip`, `zeros`                        |
| Positional Encoding   | `sin`, `cos`, `arange`, `linspace`, `expand_dims`               |

---

*NumPy cheatsheet compiled for ML/DL engineering. Covers classical ML, deep learning, transformers, and production optimization.*
*Dependencies: `numpy>=1.20` for `sliding_window_view`; `numpy>=1.17` for `default_rng`.*
