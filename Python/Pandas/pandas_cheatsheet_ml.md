# Pandas Cheatsheet for ML / DL Engineers
### Complete reference for experienced AI/ML engineers and senior data scientists

```python
import pandas as pd
import numpy as np
```

---

## Table of Contents

1. [Creating DataFrames](#1-creating-dataframes)
2. [Inspection & Info](#2-inspection--info)
3. [Selecting & Slicing](#3-selecting--slicing)
4. [Filtering Rows](#4-filtering-rows)
5. [Handling Missing Values](#5-handling-missing-values)
6. [Feature Engineering](#6-feature-engineering)
7. [String Accessor (.str)](#7-string-accessor-str)
8. [DateTime Accessor (.dt)](#8-datetime-accessor-dt)
9. [Categorical Accessor (.cat)](#9-categorical-accessor-cat)
10. [Grouping & Aggregation](#10-grouping--aggregation)
11. [Window Functions (rolling, ewm, expanding)](#11-window-functions-rolling-ewm-expanding)
12. [Reshaping: pivot, melt, stack, unstack](#12-reshaping-pivot-melt-stack-unstack)
13. [Merging & Joining](#13-merging--joining)
14. [MultiIndex](#14-multiindex)
15. [Sorting & Ranking](#15-sorting--ranking)
16. [Encoding & Type Conversion](#16-encoding--type-conversion)
17. [Applying Functions](#17-applying-functions)
18. [Method Chaining & Pipelines](#18-method-chaining--pipelines)
19. [Memory Optimization](#19-memory-optimization)
20. [Time Series](#20-time-series)
21. [Exporting Data](#21-exporting-data)
22. [Critical ML Patterns](#22-critical-ml-patterns)

---

## 1. Creating DataFrames

```python
# From dict
df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

# From list of dicts (common when building from API results)
df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

# From numpy array
df = pd.DataFrame(arr, columns=["a", "b", "c"])

# From CSV (most common in ML)
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", index_col=0)               # first col as index
df = pd.read_csv("data.csv", usecols=["a", "b"])        # load specific cols only
df = pd.read_csv("data.csv", nrows=1000)                # load first 1000 rows
df = pd.read_csv("data.csv", dtype={"age": np.int32})   # specify dtypes on load
df = pd.read_csv("data.csv", parse_dates=["date"])       # auto-parse date cols
df = pd.read_csv("data.csv", na_values=["NA", "?", ""])  # custom NA markers
df = pd.read_csv("data.csv", low_memory=False)           # avoid mixed dtype warnings
df = pd.read_csv("data.csv", encoding="utf-8")

# Chunked reading — process large files that don't fit in RAM
chunks = []
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    chunk = chunk[chunk["label"] == 1]     # filter before accumulating
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# From Excel / Parquet / JSON
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_parquet("data.parquet")           # fastest columnar format
df = pd.read_json("data.json")
df = pd.read_json("data.jsonl", lines=True)    # JSON lines (common in NLP datasets)

# Flatten nested JSON
from pandas import json_normalize
df = json_normalize(records, record_path="items",
                    meta=["user_id", "timestamp"])

# From SQL
# import sqlalchemy; df = pd.read_sql("SELECT * FROM table", con=engine)

# Series (single column)
s = pd.Series([1, 2, 3], name="feature", index=["a", "b", "c"])
```

| Function                     | ML Use Case                                                |
|------------------------------|------------------------------------------------------------|
| `read_csv(dtype=...)`        | Avoid auto-inference overhead on large files               |
| `read_csv(usecols=...)`      | Load only required features (memory efficiency)            |
| `read_csv(chunksize=...)`    | Preprocess datasets larger than RAM                        |
| `read_parquet`               | Preferred format for feature stores; fast & compressed     |
| `read_json(lines=True)`      | Load JSONL datasets common in NLP (HuggingFace datasets)   |
| `json_normalize`             | Flatten nested API / log data for feature extraction       |

---

## 2. Inspection & Info

```python
df.head(10)                          # first 10 rows
df.tail(5)                           # last 5 rows
df.sample(5, random_state=42)        # 5 random rows (reproducible)
df.shape                             # (rows, cols)
df.dtypes                            # data type of each column
df.info()                            # shape + dtypes + non-null counts + memory
df.info(memory_usage="deep")         # accurate memory with object dtype
df.describe()                        # count, mean, std, min, quartiles, max (numeric)
df.describe(include="all")           # include categorical columns too
df.describe(percentiles=[.1,.25,.5,.75,.9,.99])  # custom quantiles
df.columns.tolist()                  # list of column names
df.index                             # row index
df.nunique()                         # unique values per column
df.nunique(dropna=False)             # include NaN as a unique value
df["col"].value_counts()             # frequency of each unique value
df["col"].value_counts(normalize=True)  # relative frequencies (proportions)
df["col"].value_counts(dropna=False) # include NaN count
df["col"].unique()                   # array of unique values
df.isnull().sum()                    # count of missing values per column
df.isnull().mean() * 100             # percentage missing per column
df.notnull().sum()                   # count of non-missing values
df.duplicated().sum()                # count duplicate rows
df.duplicated(subset=["col1","col2"]).sum()  # duplicates in specific cols
df.memory_usage(deep=True)           # memory footprint per column
df.memory_usage(deep=True).sum() / 1e6  # total MB

# Pairwise correlation
df.corr()                            # Pearson correlation matrix (numeric cols)
df.corr(method="spearman")          # Spearman (monotonic, robust to outliers)
df.corr(method="kendall")           # Kendall tau
df["target"].corr(df["feature"])    # correlation between two columns
df.cov()                             # covariance matrix

# Quick EDA patterns
df.isnull().sum()[df.isnull().sum() > 0]  # only columns with missing values
high_cardinality = df.nunique()[df.nunique() > 50]  # high-cardinality cols
```

| Function                       | ML Use Case                                          |
|--------------------------------|------------------------------------------------------|
| `df.describe(percentiles=...)`  | Spot outliers with tails; plan feature clipping     |
| `df.isnull().sum()`            | Identify columns needing imputation                  |
| `df.dtypes`                    | Find object columns needing encoding                 |
| `df.nunique()`                 | Identify high-cardinality categorical features       |
| `df.value_counts(normalize=True)` | Class distribution — spot imbalance              |
| `df.duplicated()`              | Find duplicate training samples                      |
| `df.corr(method='spearman')`   | Robust feature-target correlation analysis           |
| `info(memory_usage='deep')`    | Diagnose memory issues before training               |

---

## 3. Selecting & Slicing

```python
# Single column → Series
df["col"]
df.col                              # dot notation (fails if col has spaces/special chars)

# Multiple columns → DataFrame
df[["col1", "col2"]]

# By position (.iloc) — integer-location based
df.iloc[0]                          # first row
df.iloc[0:5]                        # first 5 rows
df.iloc[:, 0]                       # first column
df.iloc[:, 0:3]                     # first 3 columns
df.iloc[[0, 2, 4], [1, 3]]          # specific rows and cols by position
df.iloc[:, :-1]                     # all cols except last (X matrix)
df.iloc[:, -1]                      # last col (target variable y)

# By label (.loc) — label-based
df.loc[0, "col"]                    # specific cell
df.loc[:, "col1":"col3"]            # column range (inclusive in .loc!)
df.loc[df["col"] > 5, "col2"]       # filtered rows, specific column
df.loc[row_mask, feature_cols]      # boolean row mask + column list

# Filter columns by pattern
df.filter(like="age")               # cols containing "age"
df.filter(regex="^feat_")           # cols starting with "feat_"
df.filter(regex=".*_encoded$")      # cols ending with "_encoded"

# Select by dtype
df.select_dtypes(include="number")            # numeric columns only
df.select_dtypes(include="object")            # string/categorical columns
df.select_dtypes(include=["float64","float32"])  # specific numeric types
df.select_dtypes(exclude="object")            # non-string columns
df.select_dtypes(include="datetime")          # datetime columns
df.select_dtypes(include="category")          # categorical columns

# Cross-section with .xs (useful with MultiIndex)
df.xs(key="2024", level="year")
```

| Pattern                       | ML Use Case                                      |
|-------------------------------|--------------------------------------------------|
| `df[feature_cols]`            | Select X (features) as DataFrame                 |
| `df["target"]`                | Select y (labels) as Series                      |
| `df.select_dtypes("number")`  | Get numeric features for scaling                 |
| `df.select_dtypes("object")`  | Get categorical features for encoding            |
| `df.iloc[:, :-1]`             | All cols except last (X matrix)                  |
| `df.filter(regex=...)`        | Select feature families by naming convention     |

---

## 4. Filtering Rows

```python
# Single condition
df[df["age"] > 30]
df[df["city"] == "Mumbai"]
df[df["salary"].isnull()]
df[df["salary"].notnull()]

# Multiple conditions (use & | ~, NOT Python 'and'/'or')
df[(df["age"] > 25) & (df["city"] == "Delhi")]
df[(df["age"] < 20) | (df["age"] > 60)]
df[~(df["city"] == "Mumbai")]              # NOT Mumbai

# .isin() / .notin()
df[df["city"].isin(["Delhi", "Mumbai"])]
df[~df["city"].isin(["Delhi"])]            # exclude

# .between()
df[df["age"].between(25, 35)]              # 25 ≤ age ≤ 35 (inclusive)
df[df["age"].between(25, 35, inclusive="left")]  # left-closed interval

# .query() — readable, works with variable scoping
df.query("age > 25 and city == 'Delhi'")
df.query("salary > @threshold")            # use Python variable with @
df.query("age.between(25, 35)")            # method calls in query

# String filtering
df[df["name"].str.contains("kumar", case=False, na=False)]
df[df["name"].str.startswith("A")]
df[df["text"].str.len() > 50]              # filter short texts

# Clip values (for outlier handling)
df["salary"].clip(lower=0, upper=500_000)  # cap to valid range
df.clip(lower=df.quantile(0.01),
        upper=df.quantile(0.99), axis=1)   # percentile-based clipping

# df.where and df.mask
df.where(df > 0)                           # keep values where True, else NaN
df.mask(df > 1000, other=1000)             # replace where True with `other`
```

| Pattern                       | ML Use Case                                        |
|-------------------------------|----------------------------------------------------|
| `df[df["label"]==1]`          | Isolate one class for oversampling / analysis      |
| `df.query("split=='train'")`  | Separate pre-split datasets cleanly                |
| `.isin(valid_vals)`           | Filter invalid category values                     |
| `.between(lo, hi)`            | Remove outliers by range                           |
| `df.clip(quantile)`           | Winsorize features to remove outlier influence     |
| `df.where / df.mask`          | Conditional NaN insertion for data augmentation    |

---

## 5. Handling Missing Values

```python
# Detect
df.isnull()                                # boolean DataFrame
df.isnull().sum()                          # count per column
df.isnull().any(axis=1)                    # rows with any missing value
df.isnull().all(axis=1)                    # rows with all missing

# Drop
df.dropna()                                # drop rows with ANY missing
df.dropna(subset=["col1", "col2"])         # drop if missing in specific cols
df.dropna(thresh=3)                        # keep rows with ≥ 3 non-null values
df.dropna(axis=1)                          # drop columns with any missing
df.dropna(axis=1, thresh=int(0.5*len(df))) # drop cols > 50% missing

# Fill — simple imputation
df.fillna(0)                               # fill all NaN with 0
df["col"].fillna(df["col"].mean())         # mean imputation (numeric)
df["col"].fillna(df["col"].median())       # median imputation (robust)
df["col"].fillna(df["col"].mode()[0])      # mode imputation (categorical)
df.fillna(df.mean(numeric_only=True))      # fill all numeric cols with column mean

# Fill — time series
df.fillna(method="ffill")                  # forward fill (propagate last valid)
df.fillna(method="bfill")                  # backward fill
df.ffill(limit=3)                          # forward fill max 3 consecutive NaNs
df.bfill(limit=3)

# Interpolate
df["col"].interpolate(method="linear")     # linear interpolation
df["col"].interpolate(method="time")       # time-weighted (DatetimeIndex)
df["col"].interpolate(method="polynomial", order=2)  # polynomial interpolation
df["col"].interpolate(method="spline", order=3)      # spline interpolation
df["col"].interpolate(limit_direction="both")        # fill in both directions

# Group-wise imputation (fill with group mean — avoids global leakage)
df["age"].fillna(df.groupby("city")["age"].transform("mean"))

# Replace specific values
df.replace(-999, np.nan)                   # treat sentinel values as NaN
df.replace({"col": {-1: np.nan, 0: np.nan}})  # column-specific replacement
df.replace(["N/A", "na", "none"], np.nan)  # replace multiple sentinels

# pd.NA — nullable types (pandas >= 1.0)
# pd.NA is the new NA scalar; np.nan is float-only, pd.NA works with Int/boolean/string
df["col"] = df["col"].astype("Int64")      # nullable integer (supports pd.NA)
df["col"].isna()                           # works with both np.nan and pd.NA
```

| Function                           | ML Use Case                                          |
|------------------------------------|------------------------------------------------------|
| `fillna(mean)`                     | Mean imputation for numeric features                 |
| `fillna(median)`                   | Median imputation (robust to skew/outliers)          |
| `fillna(mode)`                     | Categorical column imputation                        |
| `groupby().transform("mean")`      | Group-conditional imputation (no data leakage)       |
| `ffill / bfill`                    | Time series missing value propagation                |
| `interpolate(method="time")`       | Temporal interpolation for sensor/financial data     |
| `dropna(thresh=...)`               | Drop mostly-empty rows/cols                          |
| `replace(-999, np.nan)`            | Handle sentinel-encoded missing values               |

---

## 6. Feature Engineering

```python
# Create new column
df["new_col"]  = df["a"] + df["b"]
df["ratio"]    = df["salary"] / df["age"].clip(lower=1)  # safe division
df["log_sal"]  = np.log1p(df["salary"])       # log(1+x) handles 0 safely
df["sqrt_age"] = np.sqrt(df["age"])
df["sq_area"]  = df["area"] ** 2

# Binning (continuous → categorical)
pd.cut(df["age"], bins=3)                     # equal-width bins
pd.cut(df["age"], bins=[0, 25, 45, 100],
       labels=["young", "mid", "senior"],
       right=True)                            # custom bins with labels
pd.qcut(df["salary"], q=4)                    # equal-frequency (quantile) bins
pd.qcut(df["salary"], q=4, labels=False)      # return integer bin codes (0-3)
pd.qcut(df["salary"], q=4, retbins=True)      # also return bin edges

# Manual Z-score normalization (compute on train, transform both)
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()

# Min-Max normalization
mn, mx = df["age"].min(), df["age"].max()
df["age_mm"] = (df["age"] - mn) / (mx - mn)

# Rank features
df["salary_rank"] = df["salary"].rank(ascending=False)
df["pct_rank"]    = df["salary"].rank(pct=True)         # percentile rank [0,1]
df["salary_rank"] = df["salary"].rank(method="dense")   # no gaps in rank

# Boolean / flag features
df["is_senior"]   = (df["age"] > 55).astype(int)
df["has_missing"] = df["salary"].isnull().astype(int)   # missingness indicator

# Interaction features
df["age_x_income"] = df["age"] * df["income"]
df["income_per_age"] = df["income"] / df["age"].clip(1)

# Shift / lag features (time series — avoid data leakage!)
df["lag_1"]  = df["sales"].shift(1)               # previous time step
df["lag_7"]  = df["sales"].shift(7)               # 7 steps back (weekly lag)
df["diff_1"] = df["sales"].diff(1)                # first difference
df["diff_7"] = df["sales"].diff(7)                # seasonal difference
df["pct_change"] = df["sales"].pct_change()       # % change from previous step
df["pct_change_7"] = df["sales"].pct_change(7)    # % change vs 7 steps ago

# Cumulative features
df["cum_sum"]  = df["sales"].cumsum()
df["cum_max"]  = df["sales"].cummax()
df["cum_min"]  = df["sales"].cummin()

# assign() — for method chaining (non-mutating)
df = (df
      .assign(log_salary=np.log1p(df["salary"]))
      .assign(age_group=pd.cut(df["age"], bins=3))
      .assign(is_senior=lambda x: (x["age"] > 55).astype(int)))
```

| Function                 | ML Use Case                                              |
|--------------------------|----------------------------------------------------------|
| `np.log1p`               | Fix right-skewed distributions (salary, price, count)    |
| `pd.cut / pd.qcut`       | Discretize continuous features for trees / embeddings    |
| `pd.qcut(retbins=True)`  | Save bin edges to apply same binning to test set         |
| `Z-score`                | Required for SVM, KNN, PCA, neural networks              |
| `Min-Max`                | Required for neural networks with bounded activations    |
| `.rank(pct=True)`        | Rank-based feature (robust to outliers)                  |
| `.shift(n) / .diff(n)`   | Lag / difference features for time series models         |
| `isnull().astype(int)`   | Missingness indicator — tells model WHERE data is missing|

---

## 7. String Accessor (.str)

```python
# Case
df["col"].str.lower()
df["col"].str.upper()
df["col"].str.title()
df["col"].str.capitalize()

# Whitespace
df["col"].str.strip()
df["col"].str.lstrip()
df["col"].str.rstrip()

# Pattern matching
df["col"].str.contains("pattern", case=False, na=False)   # boolean mask
df["col"].str.startswith("prefix")
df["col"].str.endswith("suffix")
df["col"].str.match(r"^\d{4}")     # regex match from start
df["col"].str.fullmatch(r"\d+")    # full string regex match

# Extraction
df["col"].str.extract(r"(\d+)")              # first group → new column
df["col"].str.extractall(r"(\d+)")           # all matches → MultiIndex result
df["col"].str.findall(r"\d+")                # list of all matches
df["col"].str.split(" ")                     # split string into list
df["col"].str.split(" ", expand=True)        # split into separate columns
df["col"].str.split(" ").str[0]              # first word
df["col"].str.split(" ").str[-1]             # last word
df["col"].str.get(0)                         # get element from list/dict col

# Replacement
df["col"].str.replace("old", "new", regex=False)
df["col"].str.replace(r"\s+", " ", regex=True)   # collapse whitespace
df["col"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)  # remove special chars

# Info
df["col"].str.len()                  # string length (as a feature)
df["col"].str.count("a")             # count occurrences
df["col"].str.isdigit()              # boolean: all digits?
df["col"].str.isalpha()              # boolean: all alphabetic?
df["col"].str.isnumeric()            # boolean: numeric characters?

# Padding & formatting
df["col"].str.zfill(5)               # zero-pad to width 5
df["col"].str.pad(10, fillchar=" ")  # pad to width

# Concatenation
df["first"] + " " + df["last"]      # string concat (vectorized)
df[["first","last"]].agg(" ".join, axis=1)   # join multiple cols
```

| Function                        | ML Use Case                                          |
|---------------------------------|------------------------------------------------------|
| `.str.lower() / strip()`        | Normalize text before NLP pipeline                   |
| `.str.contains(regex, na=False)`| Filter/flag relevant text records                    |
| `.str.extract(r"...")`          | Extract structured info from raw text                |
| `.str.split(expand=True)`       | Parse compound features (e.g., "city_state")         |
| `.str.replace(regex=True)`      | Text cleaning for NLP (remove HTML, punctuation)     |
| `.str.len()`                    | Text length as a feature for document classification |

---

## 8. DateTime Accessor (.dt)

```python
# Parse dates
df["date"] = pd.to_datetime(df["date_str"])
df["date"] = pd.to_datetime(df["date_str"], format="%Y-%m-%d")
df["date"] = pd.to_datetime(df["date_str"], errors="coerce")  # invalid → NaT

# Date component extraction
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["day"]        = df["date"].dt.day
df["hour"]       = df["date"].dt.hour
df["minute"]     = df["date"].dt.minute
df["second"]     = df["date"].dt.second
df["dayofweek"]  = df["date"].dt.dayofweek    # 0=Monday, 6=Sunday
df["dayofyear"]  = df["date"].dt.dayofyear
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]    = df["date"].dt.quarter
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

# Date properties
df["date"].dt.is_month_start
df["date"].dt.is_month_end
df["date"].dt.is_quarter_start
df["date"].dt.is_leap_year
df["date"].dt.days_in_month

# Time differences / deltas
df["days_since"] = (pd.Timestamp.now() - df["date"]).dt.days
df["months_since"] = (pd.Timestamp.now() - df["date"]).dt.days / 30

# Timezone
df["date"].dt.tz_localize("UTC")
df["date"].dt.tz_convert("US/Eastern")

# Floor / ceil / round
df["date"].dt.floor("H")    # floor to hour
df["date"].dt.ceil("D")     # ceil to day
df["date"].dt.round("15min") # round to 15 min

# Date arithmetic
df["date"] + pd.DateOffset(months=1)
df["date"] + pd.Timedelta(days=7)

# Sinusoidal encoding for cyclic features (months, hours, weekday)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
```

| Function                  | ML Use Case                                              |
|---------------------------|----------------------------------------------------------|
| `dt.dayofweek`            | Day-of-week feature for retail/demand forecasting        |
| `dt.hour / dt.month`      | Temporal features for time-series models                 |
| `dt.is_weekend`           | Weekend flag for sales/usage models                      |
| `dt.quarter`              | Seasonality feature for financial forecasting            |
| `(now - date).dt.days`    | "Recency" feature in RFM analysis, churn models          |
| `sin / cos encoding`      | Cyclic encoding — model knows Dec and Jan are adjacent   |

---

## 9. Categorical Accessor (.cat)

```python
# Create categorical column
df["city"] = df["city"].astype("category")
df["priority"] = pd.Categorical(df["priority"],
                                categories=["low","medium","high"],
                                ordered=True)

# Accessing categories
df["city"].cat.categories        # Index of defined categories
df["city"].cat.codes             # integer codes (label encoding)
df["city"].cat.ordered           # whether it's an ordered category

# Modifying categories
df["city"].cat.add_categories(["NewCity"])
df["city"].cat.remove_categories(["OldCity"])
df["city"].cat.rename_categories({"Mumbai": "Bombay"})
df["city"].cat.set_categories(["Delhi","Mumbai"], ordered=False)
df["city"].cat.remove_unused_categories()   # clean up stale categories

# Ordered comparison (useful for ordinal features)
df[df["priority"] > "low"]       # only works when ordered=True

# Reorder levels
df["city"].cat.reorder_categories(["Mumbai","Delhi","Hyderabad"])
```

| Function                 | ML Use Case                                              |
|--------------------------|----------------------------------------------------------|
| `.astype("category")`    | Reduce memory for high-cardinality columns (2-5× savings)|
| `.cat.codes`             | Label encode for tree models (XGBoost, LightGBM, CatBoost)|
| `pd.Categorical(ordered)`| Represent ordinal features (Low < Medium < High)         |
| `.cat.remove_unused_categories()` | Clean up after filtering (avoid ghost categories) |

---

## 10. Grouping & Aggregation

```python
# Basic groupby
df.groupby("city")["salary"].mean()
df.groupby("city")["salary"].agg(["mean", "std", "count", "median"])
df.groupby("city")["salary"].describe()    # full descriptive stats per group

# Multiple group keys
df.groupby(["city", "gender"])["salary"].mean()

# Custom aggregation (dict form)
df.groupby("city").agg({
    "salary": ["mean", "max", "std"],
    "age":    "median",
    "id":     "count"
})

# Named aggregations (pandas >= 0.25 — clean column names)
df.groupby("city").agg(
    avg_salary = pd.NamedAgg(column="salary", aggfunc="mean"),
    max_salary = pd.NamedAgg(column="salary", aggfunc="max"),
    n_employees = pd.NamedAgg(column="id", aggfunc="count"),
)

# Custom aggregation function
df.groupby("city")["salary"].agg(lambda x: x.quantile(0.9))  # 90th percentile

# transform — keep original shape (for feature engineering, NO data leakage risk!)
df["city_avg_salary"]  = df.groupby("city")["salary"].transform("mean")
df["city_std_salary"]  = df.groupby("city")["salary"].transform("std")
df["salary_vs_city"]   = df["salary"] - df["city_avg_salary"]
df["salary_pct_rank"]  = df.groupby("city")["salary"].transform(lambda x: x.rank(pct=True))

# filter — drop entire groups failing a condition
df.groupby("city").filter(lambda x: len(x) >= 10)   # only cities with ≥10 rows

# apply — most flexible, slowest
df.groupby("city").apply(lambda g: g.nlargest(3, "salary"))  # top-3 per city

# Size and counts
df.groupby("city").size()                    # count rows per group
df.groupby("city")["col"].nunique()          # unique values per group
df.groupby("city")["col"].value_counts()     # frequency per group

# Cumulative within groups
df["cum_sales_in_city"] = df.groupby("city")["sales"].cumsum()

# Cross-tabulation
pd.crosstab(df["gender"], df["city"])                     # frequency table
pd.crosstab(df["gender"], df["city"], normalize="index")  # row-wise proportions
pd.crosstab(df["gender"], df["city"], values=df["salary"], aggfunc="mean")  # pivot-like

# Pivot table
df.pivot_table(values="salary", index="city",
               columns="gender", aggfunc="mean",
               fill_value=0, margins=True)
```

| Function                        | ML Use Case                                          |
|---------------------------------|------------------------------------------------------|
| `groupby().mean()`              | Target mean encoding (fit on train only!)            |
| `groupby().transform("mean")`   | Group-level features without row loss                |
| `groupby().transform("rank")`   | Group-relative rank features                         |
| `groupby().filter(...)`         | Remove small groups before training                  |
| `groupby().agg(NamedAgg)`       | Clean multi-agg feature table with proper col names  |
| `pd.crosstab(normalize=...)`    | Feature co-occurrence analysis                       |
| `pivot_table(margins=True)`     | Summary statistics with totals                       |

---

## 11. Window Functions (rolling, ewm, expanding)

```python
# Rolling — fixed-size sliding window
df["roll_mean_7"]  = df["sales"].rolling(window=7).mean()
df["roll_std_7"]   = df["sales"].rolling(window=7).std()
df["roll_max_7"]   = df["sales"].rolling(window=7).max()
df["roll_min_7"]   = df["sales"].rolling(window=7).min()
df["roll_sum_7"]   = df["sales"].rolling(window=7).sum()
df["roll_median_7"]= df["sales"].rolling(window=7).median()

# Rolling with min_periods (handle edges)
df["roll_mean"]    = df["sales"].rolling(7, min_periods=1).mean()  # no NaN at start

# Rolling with custom function
df["roll_q90"]     = df["sales"].rolling(7).apply(lambda x: np.percentile(x, 90))
df["roll_range"]   = df["sales"].rolling(7).apply(lambda x: x.max() - x.min())

# Center window (symmetric — careful: causes lookahead in time series)
df["roll_mean_c"]  = df["sales"].rolling(7, center=True).mean()

# Rolling grouped by entity (multi-series)
df["roll_mean_grp"] = df.groupby("store_id")["sales"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# Exponentially Weighted Moving Average (EWM) — upweights recent obs
df["ewm_mean"]     = df["sales"].ewm(span=7).mean()     # span-based decay
df["ewm_mean"]     = df["sales"].ewm(com=3).mean()      # center-of-mass
df["ewm_mean"]     = df["sales"].ewm(halflife=3).mean() # half-life decay
df["ewm_std"]      = df["sales"].ewm(span=7).std()      # EWM std (volatility)
df["ewm_var"]      = df["sales"].ewm(span=7).var()

# Expanding — cumulative window (grows from start)
df["cum_mean"]     = df["sales"].expanding().mean()
df["cum_max"]      = df["sales"].expanding().max()
df["cum_std"]      = df["sales"].expanding().std()
df["cum_q90"]      = df["sales"].expanding().quantile(0.9)

# Rolling corr / cov between two series
df["roll_corr"]    = df["a"].rolling(30).corr(df["b"])   # 30-step rolling correlation
df["roll_cov"]     = df["a"].rolling(30).cov(df["b"])
```

| Function                       | ML Use Case                                              |
|--------------------------------|----------------------------------------------------------|
| `rolling(7).mean()`            | Moving average feature for time-series models            |
| `rolling(7).std()`             | Volatility / uncertainty feature                         |
| `rolling(min_periods=1)`       | Avoid NaN at window edges during training                |
| `ewm(span=...)`                | Exponential smoothing — captures trend with recency bias |
| `ewm(halflife=...)`            | Decay-based volatility for financial time-series         |
| `expanding().mean()`           | Cumulative statistic feature (no future leakage)         |
| `groupby(...).transform(rolling)` | Per-entity rolling features for panel data            |
| `rolling().corr(other)`        | Dynamic correlation features between paired series       |

---

## 12. Reshaping: pivot, melt, stack, unstack

```python
# pivot — reshape long → wide (unique index+column combinations required)
df.pivot(index="date", columns="city", values="sales")

# pivot_table — like pivot but handles duplicates via aggregation
df.pivot_table(index="date", columns="city",
               values="sales", aggfunc="sum", fill_value=0)

# melt — wide → long (inverse of pivot; very common in ML preprocessing)
df.melt(id_vars=["id","date"],
        value_vars=["store_A","store_B","store_C"],
        var_name="store",
        value_name="sales")

# stack — column levels → row index (wide → long for MultiIndex)
df.stack()              # last column level → innermost row index
df.stack(level=0)       # stack specific level

# unstack — row index → column levels (long → wide for MultiIndex)
df.unstack()            # innermost row level → columns
df.unstack(level="city")

# wide_to_long — structured melt for repeated measures
pd.wide_to_long(df, stubnames=["score","grade"],
                i="student_id", j="subject", sep="_")

# explode — one row per list element (common in NLP, multi-label)
df["tokens"] = df["text"].str.split()
df.explode("tokens")           # one row per token
df.explode(["col1", "col2"])   # multi-column explode (pandas >= 1.3)

# crosstab (see groupby section)
pd.crosstab(df["A"], df["B"])
```

| Function              | ML Use Case                                               |
|-----------------------|-----------------------------------------------------------|
| `pivot_table`         | Build user-item matrix for collaborative filtering        |
| `melt`                | Reshape multi-column survey/sensor data to long format    |
| `stack / unstack`     | Reshape hierarchical DataFrames for ML input              |
| `explode`             | One-row-per-token for NLP, one-row-per-label for multi-label|
| `wide_to_long`        | Convert repeated-measures data to tidy format             |

---

## 13. Merging & Joining

```python
# merge — SQL-style JOIN (most flexible)
pd.merge(df1, df2, on="id", how="inner")          # inner join
pd.merge(df1, df2, on="id", how="left")           # left join (keep all df1 rows)
pd.merge(df1, df2, on="id", how="right")          # right join
pd.merge(df1, df2, on="id", how="outer")          # outer join (keep all rows)

# Different column names
pd.merge(df1, df2, left_on="user_id", right_on="id")

# Multiple keys
pd.merge(df1, df2, on=["user_id", "date"])

# Validate merge (catch unexpected many-to-many)
pd.merge(df1, df2, on="id", how="left",
         validate="m:1")   # many df1 rows to one df2 row

# Suffixes for overlapping column names
pd.merge(df1, df2, on="id", suffixes=("_train", "_test"))

# merge_asof — time-series / ordered merge (nearest key)
pd.merge_asof(trades, quotes, on="timestamp",
              by="ticker", direction="backward")  # last quote before trade

# merge_ordered — ordered merge with optional fill
pd.merge_ordered(df1, df2, on="date", fill_method="ffill")

# concat — stack multiple DataFrames
pd.concat([df1, df2], axis=0, ignore_index=True)    # add more rows
pd.concat([df1, df2], axis=1)                        # add more columns
pd.concat([df1, df2], axis=0, keys=["train","val"])  # add hierarchical index
pd.concat(dfs, ignore_index=True, sort=False)        # list of many DataFrames

# join — index-based merge
df1.join(df2, on="id", how="left")
df1.join(df2, lsuffix="_l", rsuffix="_r")

# combine_first — fill NaN in df1 with values from df2
df1.combine_first(df2)         # useful for imputation from secondary source
```

| Function                    | ML Use Case                                           |
|-----------------------------|-------------------------------------------------------|
| `merge(how='left')`         | Join features table to labels table                   |
| `merge(validate='m:1')`     | Validate join integrity in data pipelines             |
| `pd.merge_asof`             | Join time-series data at nearest timestamp            |
| `pd.concat(axis=0)`         | Combine train + test before preprocessing             |
| `pd.concat(axis=1)`         | Add engineered features to original DataFrame         |
| `pd.concat(keys=[...])`     | Track data source in hierarchical index               |
| `combine_first`             | Fill missing features from fallback/secondary source  |

---

## 14. MultiIndex

```python
# Create MultiIndex
idx = pd.MultiIndex.from_tuples([("a",1),("a",2),("b",1)], names=["letter","num"])
df = pd.DataFrame({"val": [10, 20, 30]}, index=idx)

# From groupby (produces MultiIndex)
agg = df.groupby(["city","gender"])["salary"].mean()  # MultiIndex Series

# Accessing MultiIndex
df.loc["a"]                    # all rows with outer level = "a"
df.loc[("a", 1)]               # specific (letter=a, num=1) row
df.loc["a", "val"]             # outer level + column

# pd.IndexSlice — cleaner slicing
idx_sl = pd.IndexSlice
df.loc[idx_sl["a":"b", 1:2], :]  # slice both levels

# Cross-section
df.xs("a", level="letter")        # rows where letter == "a"
df.xs(1, level="num")             # rows where num == 1

# Flatten MultiIndex columns (after groupby agg)
agg.columns = ["_".join(col) for col in agg.columns]  # "salary_mean", "salary_max"

# Reset and set index
df.reset_index()                   # MultiIndex → regular columns
df.set_index(["city", "date"])     # create MultiIndex from columns
df.swaplevel(0, 1)                 # swap outer and inner levels
df.sort_index(level=0)            # sort by outer level
df.droplevel("num")               # remove a level
```

| Function                   | ML Use Case                                          |
|----------------------------|------------------------------------------------------|
| `groupby` → MultiIndex     | Hierarchical aggregation results                     |
| `pd.IndexSlice`            | Slice panel data (entity × time) efficiently         |
| `df.xs(key, level=...)`    | Select a specific entity from panel data             |
| `reset_index()`            | Flatten after groupby for sklearn/PyTorch input      |
| `swaplevel / sort_index`   | Reorder dimensions for time-series panel processing  |
| flatten columns pattern    | Clean column names after multi-level agg             |

---

## 15. Sorting & Ranking

```python
df.sort_values("salary", ascending=False)
df.sort_values(["city", "salary"], ascending=[True, False])  # multi-column sort
df.sort_values("salary").reset_index(drop=True)             # sort + clean index
df.nlargest(10, "salary")                                   # top 10
df.nsmallest(5, "salary")                                   # bottom 5
df.nlargest(10, ["salary", "age"])                          # break ties with age

# Sort index
df.sort_index()                     # sort by row index
df.sort_index(axis=1)               # sort columns alphabetically

# Ranking
df["salary_rank"] = df["salary"].rank(ascending=False, method="min")
# method: 'average', 'min', 'max', 'first', 'dense'
df["salary_dense_rank"] = df["salary"].rank(method="dense")   # no rank gaps
df["pct_rank"] = df["salary"].rank(pct=True)                  # [0,1] percentile

# Within-group rank (for group-relative features)
df["rank_in_city"] = df.groupby("city")["salary"].rank(ascending=False)
```

| Function                   | ML Use Case                                          |
|----------------------------|------------------------------------------------------|
| `nlargest`                 | Top-K recommendations, feature importance display    |
| `sort_values`              | Sort predictions for precision-recall curve          |
| `rank(pct=True)`           | Rank normalization feature (robust to outliers)      |
| `groupby().rank()`         | Within-group relative ranking feature                |
| `reset_index(drop=True)`   | Clean index after filtering/sorting                  |

---

## 16. Encoding & Type Conversion

```python
# One-Hot Encoding
pd.get_dummies(df["city"])                               # binary cols per category
pd.get_dummies(df, columns=["city","gender"])             # encode multiple cols
pd.get_dummies(df, columns=["city"], drop_first=True)    # avoid dummy trap (linear models)
pd.get_dummies(df, columns=["city"], dtype=np.float32)   # float for neural nets
pd.get_dummies(df["city"], sparse=True)                  # sparse matrix (high-cardinality)

# Label Encoding (ordinal / tree models)
df["city_code"] = df["city"].astype("category").cat.codes

# Manual ordinal encoding
mapping = {"low": 0, "medium": 1, "high": 2}
df["priority_enc"] = df["priority"].map(mapping)

# Frequency encoding (target encoding alternative — no leakage)
freq_map = df["city"].value_counts().to_dict()
df["city_freq"] = df["city"].map(freq_map)

# Target mean encoding (fit ONLY on train — prevent leakage)
target_mean = train_df.groupby("city")["target"].mean()
df["city_target_enc"] = df["city"].map(target_mean)

# Hashing (for high-cardinality, reproducible without fit)
# (use sklearn HashingEncoder or feature_hasher for production)

# Binary encoding (for categories with many levels)
# from category_encoders import BinaryEncoder — external library

# Type conversion
df["age"] = df["age"].astype(int)
df["salary"] = df["salary"].astype(float)
df["city"] = df["city"].astype("category")             # memory savings
df["date"] = pd.to_datetime(df["date"])
df["flag"] = df["flag"].astype(bool)
df["count"] = df["count"].astype("Int64")              # nullable integer

# Auto convert to best types (pandas >= 1.0)
df = df.convert_dtypes()                               # infer nullable types

# Convert to numpy (for sklearn / pytorch)
X = df[feature_cols].to_numpy()                        # → numpy array
X = df[feature_cols].values                            # same (older style)
y = df["target"].to_numpy()

# Convert numpy back to DataFrame
df = pd.DataFrame(arr, columns=feature_cols)

# Convert to sparse (for high-dimensional one-hot features)
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(df[oh_cols].values)
```

| Function                      | ML Use Case                                           |
|-------------------------------|-------------------------------------------------------|
| `pd.get_dummies(drop_first)`  | Avoid multicollinearity in linear models              |
| `pd.get_dummies(dtype=float32)` | Float output for neural networks                    |
| `cat.codes`                   | Label encoding for XGBoost, LightGBM, CatBoost       |
| `.map(freq_map)`              | Frequency encoding — no target leakage               |
| `groupby().mean()` target enc | Target encoding — fit on train fold only!            |
| `.astype("category")`         | Reduce memory usage (2-10× on string cols)           |
| `convert_dtypes()`            | Auto-infer best nullable types after load            |
| `.to_numpy()`                 | Convert to numpy array before sklearn/PyTorch        |

---

## 17. Applying Functions

```python
# apply to a Series (column)
df["col"].apply(lambda x: x ** 2)
df["col"].apply(np.log1p)
df["col"].apply(lambda x: "positive" if x > 0 else "negative")

# apply to each column (axis=0) or each row (axis=1)
df.apply(np.mean, axis=0)                               # column-wise mean
df.apply(lambda row: row["a"] + row["b"], axis=1)       # row-wise operation
df.apply(lambda col: (col - col.mean()) / col.std())    # column-wise normalize

# map — element-wise for Series (pandas >= 2.1 preferred over applymap for DataFrames)
df["col"].map({"a": 1, "b": 2})           # dictionary mapping (replaces unmapped with NaN)
df["col"].map(lambda x: x.upper())        # function mapping

# applymap / map (DataFrame) — element-wise on whole DataFrame
df.map(lambda x: round(x, 2))            # pandas >= 2.1 (replaces applymap)
df.applymap(lambda x: round(x, 2))       # pandas < 2.1 (still works, deprecated)

# transform — like apply but must return same-shape Series/DataFrame
df["col"].transform(lambda x: (x - x.mean()) / x.std())  # normalize within group

# pipe — for method chaining (apply functions that take DataFrame as input)
def clip_features(df, lower=0.01, upper=0.99):
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].clip(
        lower=df[num_cols].quantile(lower),
        upper=df[num_cols].quantile(upper), axis=1)
    return df

df = df.pipe(clip_features, lower=0.01, upper=0.99)

# Vectorized string ops (much faster than apply for text)
df["name"].str.lower()               # DO THIS
df["name"].apply(str.lower)          # AVOID — slower Python loop

# Performance: apply vs vectorized
# apply: ~10-100× slower; use for complex per-row logic not expressible with vectorized ops
# vectorized (arithmetic / str accessor / np.where): always prefer

# np.where / np.select for multi-condition mapping (much faster than apply)
conditions = [df["score"] >= 90, df["score"] >= 70, df["score"] >= 50]
choices    = ["A", "B", "C"]
df["grade"] = np.select(conditions, choices, default="F")
```

| Function                 | ML Use Case                                          |
|--------------------------|------------------------------------------------------|
| `.apply(func, axis=1)`   | Complex row-wise feature engineering                 |
| `.map(dict)`             | Ordinal encoding, label re-mapping                   |
| `.transform(func)`       | Group-conditional feature normalization              |
| `df.pipe(func)`          | Composable preprocessing functions (sklearn-like)    |
| `np.select(conditions)`  | Fast multi-branch conditional feature creation       |

---

## 18. Method Chaining & Pipelines

```python
# Method chaining with assign, pipe, query
df_processed = (
    df
    .query("age > 0 and salary > 0")                       # filter bad rows
    .assign(log_salary=np.log1p(df["salary"]))              # new feature
    .assign(age_group=pd.cut(df["age"], bins=3, labels=False))
    .assign(is_senior=(df["age"] > 55).astype(int))
    .pipe(lambda d: d.fillna(d.median(numeric_only=True))) # impute
    .drop(columns=["redundant_col"])
    .reset_index(drop=True)
)

# Using assign with lambda (avoids stale df reference)
df_processed = (
    df
    .assign(log_salary=lambda x: np.log1p(x["salary"]))    # lambda avoids ref issues
    .assign(income_rank=lambda x: x["salary"].rank(pct=True))
    .rename(columns={"salary": "income"})
)

# pipe with external preprocessing functions
def remove_outliers(df, col, n_std=3):
    mean, std = df[col].mean(), df[col].std()
    return df[np.abs(df[col] - mean) <= n_std * std]

def encode_categoricals(df, cols):
    return pd.get_dummies(df, columns=cols, drop_first=True)

df_clean = (
    df
    .pipe(remove_outliers, col="salary", n_std=3)
    .pipe(encode_categoricals, cols=["city","gender"])
)

# eval() — fast expression evaluation (uses numexpr if installed)
df.eval("income_ratio = salary / age")               # in-place new column
result = df.eval("salary > 50000 and age < 40")      # boolean expression
df = df.query("salary > @threshold and city in @valid_cities")
```

| Function      | ML Use Case                                                  |
|---------------|--------------------------------------------------------------|
| `.assign(lambda)` | Safe method chaining (lambda avoids stale references)   |
| `.pipe(func)` | Composable, testable preprocessing steps                     |
| `.query(@var)`| Parameterized row filtering in pipelines                     |
| `df.eval()`   | Fast vectorized expression evaluation                        |

---

## 19. Memory Optimization

```python
# Check memory usage
df.memory_usage(deep=True).sum() / 1e6     # total MB (deep=True for object cols)

# Downcast numeric dtypes
def reduce_mem_usage(df):
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")   # float64 → float32
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer") # int64 → int8/16/32
    return df

# Convert string → category (major savings for low-cardinality strings)
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() / len(df) < 0.5:   # less than 50% unique → use category
        df[col] = df[col].astype("category")

# Use nullable integer (avoids float64 for int columns with NaN)
df["count"] = df["count"].astype("Int32")   # pd.Int32Dtype — supports NA without float

# Sparse columns (for high-cardinality one-hot encoded cols)
df["col_sparse"] = df["col"].astype(pd.SparseDtype("float32", fill_value=0))

# Load only needed columns
df = pd.read_csv("file.csv", usecols=["col1","col2","target"])

# Use Parquet for storage (columnar, compressed, fast)
df.to_parquet("data.parquet", compression="snappy")   # save
df = pd.read_parquet("data.parquet")                  # load

# Chunked reading (see Section 1)
for chunk in pd.read_csv("large.csv", chunksize=50_000):
    process(chunk)

# del + gc for large intermediate DataFrames
import gc
del df_large
gc.collect()   # force garbage collection
```

| Technique                          | Memory Impact                                          |
|------------------------------------|--------------------------------------------------------|
| `float64 → float32`                | 2× reduction for all float columns                     |
| `int64 → int8/16/32`               | 2-8× reduction depending on value range                |
| `object → category`                | 10-100× reduction for low-cardinality string columns   |
| `pd.SparseDtype`                   | 10-100× for very sparse one-hot features               |
| `read_csv(usecols=...)`            | Load only required columns from disk                   |
| Parquet format                     | 5-10× smaller than CSV; faster read/write             |

---

## 20. Time Series

```python
# Set DatetimeIndex
df = df.set_index("date")
df.index = pd.to_datetime(df.index)

# Create date range
date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
date_range = pd.date_range(start="2023-01-01", periods=365, freq="D")
pd.bdate_range(start="2023-01-01", end="2023-12-31")   # business days only

# Frequency strings: D=day, W=week, M=month, Q=quarter, Y=year,
#                    H=hour, T/min=minute, S=second, B=business day

# Resampling — change frequency (like groupby for time)
df.resample("W").mean()                    # weekly mean (downsample)
df.resample("M").sum()                     # monthly sum
df.resample("Q").agg({"sales":"sum","price":"mean"})
df.resample("D").ffill()                   # upsample to daily, forward-fill
df.resample("H").interpolate("linear")    # upsample to hourly, interpolate

# TimeGrouper for groupby + resample
df.groupby([pd.Grouper(freq="M"), "city"])["sales"].sum()

# Time-based indexing
df["2023"]                                 # all of 2023
df["2023-06"]                              # all of June 2023
df["2023-01-01":"2023-03-31"]              # date range slicing
df.loc["2023-01-01":"2023-03-31"]          # same with .loc

# Shifting and differencing (same as Section 6)
df["lag_1"] = df["value"].shift(1)
df["diff_1"] = df["value"].diff(1)

# Period type (for fiscal periods)
df.index = df.index.to_period("M")        # convert to monthly periods
df.index = df.index.to_timestamp()        # back to timestamps

# Timedelta
df["days_open"] = (df["close_date"] - df["open_date"]).dt.days
pd.Timedelta("1 days 2 hours")
df["date"] + pd.offsets.BDay(5)           # add 5 business days

# Seasonal decomposition
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(df["sales"], model="additive", period=12)

# Stationarity check example
df["sales_log_diff"] = np.log1p(df["sales"]).diff(1)   # log + first diff
```

| Function                       | ML Use Case                                           |
|--------------------------------|-------------------------------------------------------|
| `resample("M").mean()`         | Downsample to monthly for aggregated time series      |
| `resample("D").ffill()`        | Upsample + fill gaps in irregular time series         |
| `pd.date_range(freq="B")`      | Business day calendar for financial models            |
| `df["2023-01":"2023-06"]`      | Slice time series for train/val split by date         |
| `Grouper(freq="M")`            | Group by time period + other dimensions simultaneously|
| `.diff(1)` + `np.log1p`        | First-difference of log for stationarity              |

---

## 21. Exporting Data

```python
df.to_csv("output.csv", index=False)                  # save without row numbers
df.to_csv("output.csv", index=False, float_format="%.4f")  # format floats
df.to_excel("output.xlsx", index=False, sheet_name="Features")
df.to_parquet("output.parquet", compression="snappy")  # preferred for ML pipelines
df.to_parquet("output.parquet", engine="pyarrow")
df.to_json("output.json", orient="records", lines=True)  # JSONL format
df.to_feather("output.feather")                        # fast in-memory format
df.to_pickle("output.pkl")                             # Python pickle (not portable)
df.to_numpy()                                          # convert to numpy array
df.to_dict("records")                                  # list of row dicts
df.to_dict("list")                                     # {col: [values...]}

# Write multiple sheets to Excel
with pd.ExcelWriter("output.xlsx") as writer:
    df_train.to_excel(writer, sheet_name="Train", index=False)
    df_test.to_excel(writer, sheet_name="Test", index=False)
    df_meta.to_excel(writer, sheet_name="Metadata", index=False)

# Save only specific dtypes (e.g., exclude object for model input)
df.select_dtypes("number").to_parquet("numeric_features.parquet")
```

| Format        | Use Case                                              | Notes                      |
|---------------|-------------------------------------------------------|----------------------------|
| `.parquet`    | Feature store, large datasets, ML pipelines           | Compressed, typed, fast    |
| `.csv`        | Sharing, EDA, human-readable output                   | Slow, no dtype preservation|
| `.feather`    | Fast inter-process sharing (pandas ↔ R/Spark)         | Not for long-term storage  |
| `.pickle`     | Cache intermediate objects (DataFrames, dicts)        | Not portable across versions|
| `.json(lines)`| NLP datasets, API outputs, JSONL format               | Human-readable, flexible   |

---

## 22. Critical ML Patterns

### Correct Train/Test Split & Preprocessing (No Data Leakage)

```python
from sklearn.model_selection import train_test_split

# Split BEFORE any statistics computation
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df["target"])

feature_cols = [c for c in df.columns if c != "target"]
num_cols = train_df[feature_cols].select_dtypes("number").columns.tolist()
cat_cols = train_df[feature_cols].select_dtypes("object").columns.tolist()

# Compute imputation stats on TRAIN only
num_fill = train_df[num_cols].median()       # fit on train
cat_fill = train_df[cat_cols].mode().iloc[0] # fit on train

train_df[num_cols] = train_df[num_cols].fillna(num_fill)
test_df[num_cols]  = test_df[num_cols].fillna(num_fill)   # apply train stats to test
train_df[cat_cols] = train_df[cat_cols].fillna(cat_fill)
test_df[cat_cols]  = test_df[cat_cols].fillna(cat_fill)

# Compute scaling stats on TRAIN only
mean = train_df[num_cols].mean()
std  = train_df[num_cols].std().replace(0, 1)  # avoid divide-by-zero

train_df[num_cols] = (train_df[num_cols] - mean) / std
test_df[num_cols]  = (test_df[num_cols]  - mean) / std    # use TRAIN stats!
```

### Target Mean Encoding (Leak-Safe with K-Fold)

```python
from sklearn.model_selection import KFold

def target_encode_kfold(train_df, test_df, col, target, n_splits=5):
    train_df = train_df.copy()
    test_df  = test_df.copy()
    global_mean = train_df[target].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df[f"{col}_enc"] = global_mean

    for tr_idx, val_idx in kf.split(train_df):
        fold_map = (train_df.iloc[tr_idx]
                    .groupby(col)[target].mean())
        train_df.loc[train_df.index[val_idx], f"{col}_enc"] = (
            train_df.iloc[val_idx][col].map(fold_map).fillna(global_mean))

    test_map = train_df.groupby(col)[f"{col}_enc"].mean()
    test_df[f"{col}_enc"] = test_df[col].map(test_map).fillna(global_mean)
    return train_df, test_df
```

### Class Imbalance Handling

```python
# Check balance
print(df["target"].value_counts(normalize=True))

# Oversampling minority with pandas
minority_df = df[df["target"] == 1]
majority_df = df[df["target"] == 0]

# Random oversample minority to match majority
oversampled = minority_df.sample(n=len(majority_df),
                                  replace=True,
                                  random_state=42)
balanced_df = pd.concat([majority_df, oversampled], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute class weights for loss function
n_samples = len(df)
n_classes = df["target"].nunique()
class_counts = df["target"].value_counts().sort_index()
class_weights = n_samples / (n_classes * class_counts)
weight_dict = class_weights.to_dict()
```

### Feature Selection by Correlation

```python
def remove_highly_correlated(df, threshold=0.95):
    num_df = df.select_dtypes("number")
    corr_matrix = num_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropping {len(to_drop)} correlated features: {to_drop}")
    return df.drop(columns=to_drop)
```

### Build Feature Matrix from DataFrame

```python
# Select and validate features before model training
feature_cols = (
    df.select_dtypes("number").columns
    .difference(["target", "id", "date"])
    .tolist()
)

# Assert no missing values in features
assert df[feature_cols].isnull().sum().sum() == 0, "Missing values in features!"
# Assert dtype
assert df[feature_cols].dtypes.apply(lambda d: d in [np.float32, np.float64, np.int32, np.int64]).all()

X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.int64)
```

### Time Series Train/Val Split (No Shuffling — Respect Temporal Order)

```python
df = df.sort_values("date").reset_index(drop=True)
split_date = "2023-10-01"

train_df = df[df["date"] < split_date].copy()
val_df   = df[df["date"] >= split_date].copy()

# Lag features — compute BEFORE split to avoid index gaps
df["lag_1"]     = df["sales"].shift(1)
df["roll_7"]    = df["sales"].rolling(7, min_periods=1).mean()

# Always drop first N rows (NaN from lag features) from train
train_df = train_df.dropna(subset=["lag_1"])
```

### Efficient EDA Report

```python
def quick_eda(df, target=None):
    print(f"Shape: {df.shape}")
    print(f"\nMemory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"\nMissing values:\n{df.isnull().mean().mul(100).round(1)[lambda x: x>0].sort_values(ascending=False)}")
    print(f"\nHigh-cardinality cols (>50 unique):\n{df.nunique()[df.nunique()>50]}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    if target:
        print(f"\nTarget distribution:\n{df[target].value_counts(normalize=True).round(3)}")
    return df.describe(include="all")
```

---

*Pandas cheatsheet compiled for ML/DL engineering. Covers EDA, feature engineering, time series, preprocessing pipelines, and production best practices.*
*Dependencies: `pandas>=1.3` for `explode(multi-column)`, `pd.NA`; `pandas>=0.25` for `NamedAgg`; `pandas>=2.0` for `.map()` on DataFrame.*
