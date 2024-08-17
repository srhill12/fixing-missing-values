#```markdown
# Crowdfunding Dataset: Handling Missing Data

This project involves working with a crowdfunding dataset that contains missing values. The focus is on preprocessing the data to handle missing values in a meaningful way, specifically in the `backers_count` column.

## Dataset Overview

The dataset consists of the following features:

- `goal`: The funding goal of the project.
- `pledged`: The amount pledged by backers.
- `backers_count`: The number of backers supporting the project (contains missing values).
- `days_active`: The number of days the campaign has been active.
- `outcome`: The target variable indicating whether the project was successful (`1`) or not (`0`).

### Initial Data Inspection

The dataset is loaded and an initial inspection reveals missing values in the `backers_count` column:

```python
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/crowdfunding-missing-data.csv")
df.head()
```

### Splitting the Data

The dataset is split into features (`X`) and the target variable (`y`), and further into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='outcome')
y = df['outcome'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
```

## Handling Missing Data

### Missing Data Analysis

The percentage of missing values in each column of the training set is calculated:

```python
X_train.isna().sum()/len(df)
```

It is found that approximately 9.39% of the rows have missing values in the `backers_count` column. 

### Descriptive Statistics for Rows with Missing Data

Descriptive statistics are generated for the rows with missing `backers_count` values:

```python
X_train.loc[X_train['backers_count'].isna()].describe()
```

### Exploratory Data Analysis

Histograms are plotted to explore the distribution of `days_active` for the entire dataset and specifically for the rows where `backers_count` is missing:

```python
X_train['days_active'].plot(kind='hist', alpha=0.2)
X_train.loc[df['backers_count'].isna(), 'days_active'].plot(kind='hist')
```

The analysis suggests that the `backers_count` is often missing during the first week of a campaign.

## Filling Missing Values

### Strategy for Handling Missing `backers_count`

Since `backers_count` is frequently missing during the first week, a decision is made to fill the missing values using half of the mean `backers_count` from the second week of campaigns (days 6 to 13):

```python
mean_of_week_2_backers_counts = X_train.loc[(X_train['days_active'] >= 6) & (X_train['days_active'] <= 13), 'backers_count'].mean()
```

A function is created to fill the missing values in `backers_count`:

```python
def X_preprocess(X_data):
    X_data['backers_count'] = X_data['backers_count'].fillna(int(round(mean_of_week_2_backers_counts/2)))
    return X_data
```

### Preprocessing Training and Testing Data

The function is applied to both the training and testing sets to fill in the missing values:

```python
X_train_clean = X_preprocess(X_train)
X_test_clean = X_preprocess(X_test)
```

### Verification

The presence of missing values is checked again to ensure that all missing values have been addressed:

```python
print(X_train_clean.isna().sum()/len(X_train_clean))
print(X_test_clean.isna().sum()/len(X_test_clean))
```

Both the training and testing sets show no missing values after preprocessing.

## Conclusion

The preprocessing steps successfully handled the missing data in the `backers_count` column, allowing for more accurate model training and evaluation. The chosen method of filling missing values was based on an understanding of the data distribution, specifically targeting the early stages of a crowdfunding campaign where data was most often missing.

This approach ensures that the model can make use of all available data, reducing bias and improving generalization in the predictions.
```
