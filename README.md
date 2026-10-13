# 🏨 Hotel Booking Cancellation Prediction

### An End-to-End Machine Learning Project Using Python, Scikit-learn, and Streamlit

<p align="center">
  <strong>Data Analysis | Feature Engineering | Machine Learning | Model Evaluation | Deployment</strong>
</p>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Project Objectives](#-project-objectives)
4. [Business Use Cases](#-business-use-cases)
5. [Technology Stack](#-technology-stack)
6. [Dataset Description](#-dataset-description)
7. [Project Workflow](#-project-workflow)
8. [Step 1: Data Understanding](#-step-1-data-understanding)
9. [Step 2: Data Cleaning](#-step-2-data-cleaning)
10. [Step 3: Exploratory Data Analysis](#-step-3-exploratory-data-analysis)
11. [Step 4: Target Variable Analysis](#-step-4-target-variable-analysis)
12. [Step 5: Data Leakage Prevention](#-step-5-data-leakage-prevention)
13. [Step 6: Feature Engineering](#-step-6-feature-engineering)
14. [Step 7: Handling Missing Values and Rare Categories](#-step-7-handling-missing-values-and-rare-categories)
15. [Step 8: Feature Selection](#-step-8-feature-selection)
16. [Step 9: Separating Features and Target](#-step-9-separating-features-and-target)
17. [Step 10: Train-Test Split](#-step-10-train-test-split)
18. [Step 11: Identifying Numerical and Categorical Features](#-step-11-identifying-numerical-and-categorical-features)
19. [Step 12: Building the Preprocessing Pipeline](#-step-12-building-the-preprocessing-pipeline)
20. [Step 13: Model Training](#-step-13-model-training)
21. [Step 14: Model Evaluation](#-step-14-model-evaluation)
22. [Step 15: Saving the Trained Pipeline](#-step-15-saving-the-trained-pipeline)
23. [Step 16: Streamlit Application](#-step-16-streamlit-application)
24. [How the Prediction System Works](#-how-the-prediction-system-works)
25. [Project Structure](#-project-structure)
26. [Installation and Local Execution](#-installation-and-local-execution)
27. [Deployment](#-deployment)
28. [Important Machine Learning Concepts Applied](#-important-machine-learning-concepts-applied)
29. [Challenges and Solutions](#-challenges-and-solutions)
30. [Limitations and Future Improvements](#-limitations-and-future-improvements)
31. [Key Learnings](#-key-learnings)
32. [Author](#-author)

---

## 📖 Project Overview

Hotel Booking Cancellation Prediction is an end-to-end supervised machine learning project designed to predict whether a hotel reservation is likely to be canceled.

The project uses the Hotel Booking Demand dataset from Kaggle. It applies a complete machine learning workflow, beginning with data inspection and exploratory data analysis and continuing through data cleaning, feature engineering, preprocessing, model training, evaluation, and application development.

A Random Forest classifier is used to learn patterns associated with booking cancellations. The trained model is integrated into a Streamlit web application, where users can enter reservation details and receive a cancellation prediction.

Rather than treating model training as an isolated task, this project emphasizes the complete prediction workflow, including consistent preprocessing, prevention of data leakage, reusable model pipelines, and integration with a user-facing application.

### Project at a Glance

| Component                   | Implementation                           |
| --------------------------- | ---------------------------------------- |
| Project type                | Supervised Machine Learning              |
| Problem type                | Binary Classification                    |
| Domain                      | Hospitality and Hotel Revenue Management |
| Dataset                     | Hotel Booking Demand                     |
| Data source                 | Kaggle                                   |
| Target variable             | `is_canceled`                            |
| Primary algorithm           | Random Forest Classifier                 |
| Numerical preprocessing     | StandardScaler                           |
| Categorical preprocessing   | OneHotEncoder                            |
| Preprocessing orchestration | ColumnTransformer                        |
| Model workflow              | Scikit-learn Pipeline                    |
| Model persistence           | Joblib                                   |
| Application framework       | Streamlit                                |
| Programming language        | Python                                   |
| Deployment platform         | Streamlit Community Cloud                |

---

## 🎯 Business Problem

Hotel reservations are frequently canceled after they have been made. Unexpected cancellations can make it difficult for hotels to forecast occupancy, plan staffing, allocate resources, and manage room availability.

For example, consider a hotel that receives 100 reservations for a future date. If a significant number of these reservations are canceled shortly before arrival, the hotel may experience lower occupancy than expected.

On the other hand, if the hotel anticipates cancellations too aggressively, it may make poor decisions about room allocation or overselling.

A predictive system can help hotel management identify reservations that exhibit characteristics associated with cancellation.

### Problem Statement

**Can historical hotel reservation data be used to predict whether a future booking will be canceled?**

This project formulates the problem as binary classification.

The model receives information about a reservation and predicts one of two classes:

* `0` — The reservation is not canceled.
* `1` — The reservation is canceled.

The prediction can potentially support operational planning, although it should not be treated as a guarantee that an individual guest will or will not cancel.

---

## 🎯 Project Objectives

The primary objectives are:

1. Understand the structure and characteristics of the Hotel Booking Demand dataset.
2. Identify data quality issues, missing values, duplicate records, and potential inconsistencies.
3. Perform exploratory data analysis to investigate booking patterns and cancellation behavior.
4. Prevent target leakage by removing information unavailable at the intended prediction time.
5. Engineer additional features that describe reservations more meaningfully.
6. Prepare numerical and categorical variables for machine learning.
7. Split the dataset into training and testing subsets using stratification.
8. Build a reusable preprocessing and classification pipeline.
9. Train a Random Forest classifier.
10. Evaluate the model using appropriate classification metrics.
11. Save the complete fitted pipeline for later use.
12. Integrate the model into a Streamlit application.
13. Deploy the application so that users can interact with the prediction system.

---

## 💼 Business Use Cases

A cancellation prediction model could help hotel operations teams with:

* **Occupancy forecasting:** Estimate the likelihood of cancellations across upcoming reservations.
* **Revenue management:** Incorporate cancellation risk into demand planning.
* **Operational planning:** Improve staffing and resource planning by considering expected reservation changes.
* **Reservation monitoring:** Identify bookings that may warrant closer operational attention.
* **Business analysis:** Investigate how lead time, deposit policies, customer type, and other booking characteristics relate to cancellation behavior.

These are potential applications rather than demonstrated business outcomes. Actual operational use would require validation on appropriately timed, real-world reservation data.

---

## 🛠️ Technology Stack

### 1. Python

Python is used for data analysis, feature engineering, machine learning, and application development.

### 2. Pandas

Used for:

* Loading and inspecting the dataset.
* Handling missing values.
* Detecting and removing duplicate rows.
* Creating engineered features.
* Filtering and transforming data.
* Preparing features and the target variable.

### 3. NumPy

Used for numerical operations and supporting data manipulation.

### 4. Matplotlib and Seaborn

Used during exploratory data analysis to visualize distributions, relationships, and cancellation patterns.

### 5. Scikit-learn

Used for:

* `train_test_split`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `RandomForestClassifier`
* `Pipeline`
* Classification evaluation metrics

### 6. Joblib

Used to save and reload the trained machine learning pipeline.

### 7. Streamlit

Used to develop the interactive application through which users can submit reservation information and obtain predictions.

### 8. Git and GitHub

Used for version control and hosting the project repository.

### 9. Streamlit Community Cloud

Used as the deployment platform for the web application.

---

## 📊 Dataset Description

The project uses the Hotel Booking Demand dataset, which contains historical reservation records for a city hotel and a resort hotel.

The original dataset contains approximately 119,390 rows and 32 columns.

Each row represents a hotel booking, while the columns describe characteristics such as:

* Hotel type.
* Lead time.
* Arrival date.
* Length of stay.
* Number of guests.
* Meal plan.
* Country of origin.
* Market segment.
* Distribution channel.
* Deposit policy.
* Previous booking history.
* Special requests.
* Cancellation status.

### Target Variable

The target variable is:

`is_canceled`

| Value | Meaning                  |
| ----- | ------------------------ |
| `0`   | Booking was not canceled |
| `1`   | Booking was canceled     |

Because the target has two possible values, this is a binary classification problem.

### Important Dataset Consideration

Not every column in a historical dataset is necessarily appropriate for predicting a future event.

Some columns may contain information that becomes available only after the booking has been made or after the cancellation outcome is known.

Consequently, selecting features requires understanding both their statistical relationship with the target and the point in time when those features would actually be available.

---

# 🔄 Project Workflow

The project follows this sequence:

```text
                HOTEL BOOKING DATASET
                          |
                          v
                  DATA UNDERSTANDING
                          |
                          v
                    DATA CLEANING
                          |
                          v
              EXPLORATORY DATA ANALYSIS
                          |
                          v
             TARGET AND LEAKAGE ANALYSIS
                          |
                          v
                  FEATURE ENGINEERING
                          |
                          v
              FEATURE SELECTION / SPLIT
                          |
                          v
                 TRAIN-TEST SPLIT
                          |
                          v
              IDENTIFY FEATURE TYPES
                          |
                          v
                PREPROCESSING PIPELINE
                 /                \
                v                  v
         NUMERICAL DATA       CATEGORICAL DATA
                |                  |
                v                  v
          STANDARD SCALER       ONE-HOT ENCODER
                 \                /
                  \              /
                   v            v
                   RANDOM FOREST
                          |
                          v
                    MODEL EVALUATION
                          |
                          v
                 SAVE FITTED PIPELINE
                          |
                          v
                   STREAMLIT APP
                          |
                          v
              USER ENTERS BOOKING DETAILS
                          |
                          v
                  PREPROCESS + PREDICT
                          |
                          v
                CANCELLATION PREDICTION
```

Each stage serves a distinct purpose. The following sections explain the implementation and the reasoning behind the decisions.

---

# 🔍 Step 1: Data Understanding

The first step is to understand the dataset before applying transformations or training a model.

Typical inspection operations include:

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### What each operation does

**`df.head()`**

Displays the first five records. This helps inspect column names, sample values, and the general structure of the data.

**`df.shape`**

Returns the number of rows and columns.

For the original dataset:

```text
Rows:       119390
Columns:        32
```

**`df.info()`**

Displays column names, data types, and non-null counts.

This helps identify columns that contain missing values or use an unexpected data type.

**`df.describe()`**

Generates descriptive statistics for numerical columns, including:

* Mean.
* Standard deviation.
* Minimum.
* Maximum.
* Quartiles.

These statistics help identify unusual distributions and potential outliers.

**`df.isnull().sum()`**

Counts missing values in each column.

**`df.duplicated().sum()`**

Counts exact duplicate rows.

### Why this step matters

Data inspection establishes the foundation for subsequent decisions.

For example, missing values in `company` have a different interpretation from missing values in `children`. Similarly, a date-related column and a numerical measurement may require different preprocessing strategies.

Data understanding helps determine what needs cleaning, what should be investigated during EDA, and what can be used as a model feature.

---

# 🧹 Step 2: Data Cleaning

Data cleaning improves the reliability of the dataset before model development.

## 2.1 Removing Duplicate Records

The original dataset contains 31,994 exact duplicate rows.

The operation used is:

```python
df = df.drop_duplicates()
```

After removing duplicates, the dataset contains:

```text
Original rows:       119390
Duplicate rows:       31994
Remaining rows:       87396
```

### Why remove duplicates?

Duplicate records can cause repeated observations to receive additional influence during model training.

For example, if an identical canceled booking appears several times, the model may give that observation more influence than intended.

Removing exact duplicates reduces this problem.

### Important consideration

An exact duplicate is not automatically proof of a data collection error in every dataset. In some applications, identical records can represent legitimate repeated events.

In this project, duplicate rows were removed as part of the data-cleaning workflow.

The decision should still be documented because it affects the number and composition of observations used for training.

---

## 2.2 Investigating Missing Values

The dataset contains missing values in several columns.

Important examples from the original dataset include:

| Column     | Approximate missing values | Approximate percentage |
| ---------- | -------------------------: | ---------------------: |
| `company`  |                     82,137 |                  68.8% |
| `agent`    |                     12,193 |                  10.2% |
| `country`  |                        452 |                   0.4% |
| `children` |                          4 |        Less than 0.01% |

These figures describe the original dataset, before duplicate removal.

### Why missing-value analysis matters

Missing values can affect both statistical analysis and model training.

Different columns may require different treatment:

* A mostly missing column may contribute little reliable information.
* A categorical column may need an explicit unknown category.
* A numerical column may need an imputation strategy based on its meaning.
* A missingness indicator may capture potentially useful information about the observation.

Missingness should therefore be investigated column by column rather than handled using a single rule for the entire dataset.

---

## 2.3 Handling `company` and `agent`

The `company` and `agent` columns contain identifiers associated with companies and travel agents.

The `company` column is particularly sparse, with approximately 94% of its values missing after duplicate removal.

Rather than directly using the raw identifiers as model inputs, this project retains missingness indicators:

```text
company_missing
agent_missing
```

These features represent whether the corresponding original values were missing.

### Why use missingness indicators?

The absence of a company or travel agent identifier may be associated with the booking channel or reservation process.

For example, a reservation without a company identifier may have been made through a different channel from one associated with a company.

However, missingness is not necessarily evidence of cancellation risk. Its usefulness must be established through evaluation.

The project therefore retains missingness information without feeding the raw identifiers directly into the model.

---

## 2.4 Handling Missing Values in `children`

Only a small number of records have missing values in `children`.

These values are filled with zero:

```python
df["children"] = df["children"].fillna(0)
```

### Why zero?

The feature represents the number of children associated with a booking. Zero is a meaningful value when a reservation has no children.

However, this operation makes an assumption: missing values are treated as if no children were present.

That assumption should be verified against the dataset's data dictionary and business context when developing a production system.

---

## 2.5 Handling Missing Values in `country`

The missing values in `country` are replaced with an explicit category:

```python
df["country"] = df["country"].fillna("Unknown")
```

### Why use `Unknown`?

A missing country value does not necessarily indicate that the guest belongs to a particular country.

Using an explicit category preserves the distinction between:

* A known country.
* A country that was not recorded.

This is preferable to arbitrarily assigning a country to an unknown observation.

---

# 📈 Step 3: Exploratory Data Analysis

Exploratory Data Analysis, or EDA, is used to understand the data and investigate relationships between variables before finalizing the modeling approach.

The purpose is not merely to create visualizations. It is to use statistical summaries and plots to support decisions about cleaning, feature engineering, feature selection, and evaluation.

## 3.1 Univariate Analysis

Univariate analysis investigates one variable at a time.

Examples include:

* Distribution of `lead_time`.
* Frequency of each `hotel` type.
* Distribution of `adr`.
* Frequency of `deposit_type`.
* Distribution of `market_segment`.

Common visualizations include:

* Histograms.
* Count plots.
* Box plots.
* Bar charts.

### Example

A histogram of `lead_time` can help determine whether bookings are concentrated around short lead times or whether many reservations are made months in advance.

A box plot of `adr` can help identify unusually high or low daily rates.

These patterns may influence the choice of transformations and the interpretation of model behavior.

---

## 3.2 Bivariate Analysis

Bivariate analysis investigates the relationship between two variables.

In this project, the target variable is `is_canceled`.

Useful comparisons include:

| Feature                     | Investigation                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `hotel`                     | Compare cancellation patterns between city and resort hotels                        |
| `lead_time`                 | Investigate whether cancellation frequency varies with booking lead time            |
| `deposit_type`              | Examine cancellation patterns across deposit policies                               |
| `market_segment`            | Compare cancellation patterns across market segments                                |
| `customer_type`             | Investigate differences between customer categories                                 |
| `total_of_special_requests` | Examine the relationship between special requests and cancellation status           |
| `previous_cancellations`    | Investigate whether previous cancellations are associated with future cancellations |

### Example: Lead Time

`lead_time` measures the number of days between the booking date and the arrival date.

A longer lead time means the reservation was made further in advance.

It is reasonable to investigate whether bookings with different lead times exhibit different cancellation rates.

However, an observed relationship does not prove that lead time causes cancellation. Other variables may also explain the pattern.

---

## 3.3 Correlation Analysis

Correlation analysis can help identify relationships between numerical variables.

For example:

* `stays_in_weekend_nights`
* `stays_in_week_nights`
* `adults`
* `children`
* `babies`
* `adr`
* `lead_time`

A correlation matrix can help reveal highly related numerical features.

However, correlation analysis has limitations:

1. Correlation does not establish causation.
2. A weak linear correlation does not imply that a feature has no predictive value.
3. Numerical correlation does not fully capture relationships involving categorical variables.
4. Nonlinear models such as Random Forest can learn relationships that a simple correlation matrix may not reveal.

EDA should guide investigation rather than act as the sole feature-selection mechanism.

---

## 3.4 EDA Versus Feature Engineering

These two activities serve different purposes.

### Exploratory Data Analysis

EDA asks:

* What patterns exist in the data?
* Are there unusual observations?
* Which variables appear related to cancellation?
* Are missing values concentrated in particular columns?
* Are there class imbalances?

### Feature Engineering

Feature engineering asks:

* Can existing columns be combined into a more meaningful feature?
* Can raw dates be converted into useful components?
* Can missingness be represented explicitly?
* Can several related variables describe a higher-level concept?

For example, EDA may reveal that the length of stay varies considerably across reservations.

Feature engineering can then create:

```python
df["total_stay_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)
```

EDA helps identify a potentially useful pattern. Feature engineering creates a representation that the model can use.

---

# 🎯 Step 4: Target Variable Analysis

Before training a classifier, it is important to understand the distribution of the target variable.

The target is:

```python
is_canceled
```

Its values are:

```text
0 = Not canceled
1 = Canceled
```

After duplicate removal, the target distribution is approximately:

| Class     | Meaning      |    Records | Percentage |
| --------- | ------------ | ---------: | ---------: |
| `0`       | Not canceled |     63,371 |     72.51% |
| `1`       | Canceled     |     24,025 |     27.49% |
| **Total** |              | **87,396** |   **100%** |

The cancellation rate is therefore approximately 27.49% in the cleaned dataset.

## Why is target distribution important?

The target is imbalanced: non-canceled bookings are more common than canceled bookings.

A model that predicts every booking as not canceled would achieve approximately 72.51% accuracy.

However, that model would fail to identify any cancellations.

This demonstrates why accuracy alone is insufficient.

The project therefore considers additional metrics, including precision, recall, F1-score, and ROC-AUC.

### Important distinction

Class imbalance does not automatically mean that the dataset must be oversampled.

The appropriate strategy depends on the model's performance, the business cost of different mistakes, and the evaluation protocol.

A useful baseline should be established before deciding whether class weighting, resampling, or threshold adjustment is necessary.

---

# 🚨 Step 5: Data Leakage Prevention

Data leakage occurs when information unavailable at the intended prediction time enters the training process.

It can produce deceptively strong evaluation results because the model indirectly receives information about the answer it is supposed to predict.

Preventing leakage is particularly important in reservation prediction.

## 5.1 Removing `reservation_status`

The column `reservation_status` describes the eventual reservation outcome.

Its values can directly reveal whether a booking was canceled.

Using it to predict cancellation would allow the model to exploit information that is too closely tied to the target.

Therefore, this column is excluded from the model features.

## 5.2 Removing `reservation_status_date`

The column `reservation_status_date` records a date associated with the reservation's status.

Depending on the reservation outcome, this date may contain information that becomes available only after booking or cancellation.

It is therefore excluded from the feature set used by this model.

## 5.3 Excluding Other Post-Booking Fields

The project also excludes the following fields:

```text
assigned_room_type
booking_changes
days_in_waiting_list
company
agent
```

The raw `company` and `agent` identifiers are excluded while their missingness indicators are retained.

Some of these excluded fields may be useful for a different prediction task, depending on when their values become available.

For example, a prediction made after a booking has undergone changes would have a different information boundary from a prediction made immediately after reservation.

The correct feature set therefore depends on the intended prediction time.

### Why this decision matters

A model should be evaluated using information that would genuinely be available when the prediction is made.

Removing outcome-related and inappropriate post-booking information makes the task more realistic and reduces the risk of an artificially inflated evaluation score.

**An important future improvement is to define the exact prediction time explicitly and audit every feature against that time.**

---

# 🧩 Step 6: Feature Engineering

Feature engineering transforms existing information into representations that may be more useful for machine learning.

In this project, several new features are created from the original reservation data.

## 6.1 `total_stay_nights`

### Original features

```text
stays_in_weekend_nights
stays_in_week_nights
```

These columns separately represent weekend and weekday nights.

### Transformation

```python
df["total_stay_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)
```

### Meaning

`total_stay_nights` represents the total length of the stay in nights.

### Why create it?

The original columns describe different parts of the stay.

Their sum provides a direct measure of total duration, which may be easier to interpret when analyzing reservation behavior.

The original columns are retained, allowing the model to use both the combined measure and the individual components.

---

## 6.2 `total_guests`

### Original features

```text
adults
children
babies
```

### Transformation

```python
df["total_guests"] = (
    df["adults"]
    + df["children"]
    + df["babies"]
)
```

### Meaning

This feature represents the total number of people associated with a reservation.

### Why create it?

The three original columns distinguish between different guest categories.

Their sum provides an overall measure of reservation size.

For example:

```text
Adults:       2
Children:     1
Babies:       0

Total guests: 3
```

The model can use both the total guest count and the individual guest counts.

---

## 6.3 `zero_guests`

### Transformation

```python
df["zero_guests"] = (
    df["total_guests"] == 0
).astype(int)
```

### Meaning

This is a binary feature.

| Value | Meaning                        |
| ----- | ------------------------------ |
| `0`   | At least one guest is recorded |
| `1`   | No guests are recorded         |

### Why create it?

A reservation with zero recorded guests may represent an unusual or potentially problematic record.

A separate indicator allows the model to distinguish these records explicitly rather than relying only on the numerical guest counts.

This feature should be evaluated to determine whether it provides useful predictive information.

---

## 6.4 `total_guest_nights`

### Transformation

```python
df["total_guest_nights"] = (
    df["total_guests"]
    * df["total_stay_nights"]
)
```

### Meaning

This feature combines reservation size and length of stay.

For example:

```text
Total guests:      3
Total stay nights: 4

Total guest-nights: 12
```

### Why create it?

Two reservations can have the same number of nights but different numbers of guests.

Similarly, two reservations can contain the same number of guests but have different durations.

`total_guest_nights` combines these dimensions into one measure.

It may help describe the scale of the reservation, although its actual predictive contribution must be determined through evaluation.

---

## 6.5 `arrival_day_of_week`

The project uses an arrival weekday feature:

```text
arrival_day_of_week
```

This represents the day of the week associated with the arrival date.

Weekday information may help the model distinguish between reservations arriving on different days.

The feature is categorical in meaning, even if represented using integers.

The exact convention used to calculate weekdays must be consistent between training and prediction.

For example, Python's `date.weekday()` uses:

```text
Monday    = 0
Tuesday   = 1
Wednesday = 2
Thursday  = 3
Friday    = 4
Saturday  = 5
Sunday    = 6
```

The training and application code should use the same convention.

---

## 6.6 Arrival Date Features

The dataset includes:

```text
arrival_date_year
arrival_date_month
arrival_date_week_number
arrival_date_day_of_month
```

These columns describe the arrival date.

The model retains:

* `arrival_date_year`
* `arrival_date_month`
* `arrival_date_week_number`

The raw `arrival_date_day_of_month` column is excluded from the final feature set.

The weekday is represented separately by `arrival_day_of_week`.

### Why represent dates using components?

A raw date is not always the most convenient representation for a model.

Date components can capture potentially useful temporal patterns, such as:

* Differences across months.
* Seasonal demand patterns.
* Changes across years.
* Differences associated with arrival weekdays.

These features do not automatically establish seasonality or causal effects. Their value should be assessed through appropriate validation.

---

## 6.7 Summary of Engineered Features

| Feature               | Calculation or source                   | Purpose                                           |
| --------------------- | --------------------------------------- | ------------------------------------------------- |
| `total_stay_nights`   | Weekend nights + weekday nights         | Total reservation duration                        |
| `total_guests`        | Adults + children + babies              | Total number of guests                            |
| `zero_guests`         | Indicator that total guests equals zero | Identify unusual zero-guest records               |
| `total_guest_nights`  | Total guests × total stay nights        | Combine reservation size and duration             |
| `arrival_day_of_week` | Derived from arrival date               | Represent weekday of arrival                      |
| `company_missing`     | Missingness of company identifier       | Preserve information about missing company values |
| `agent_missing`       | Missingness of agent identifier         | Preserve information about missing agent values   |

The final feature matrix contains 30 features, including original features and engineered features.

Feature engineering does not guarantee better performance. The purpose is to provide potentially useful representations that can be evaluated by the model.

---

# 🧼 Step 7: Handling Missing Values and Rare Categories

## 7.1 Handling Rare Country Categories

The `country` feature contains many distinct categorical values.

Some countries appear frequently, while others appear only a small number of times.

In the project, categories with fewer than 20 observations are grouped into:

```text
Other
```

Missing values are represented as:

```text
Unknown
```

### Why group rare categories?

A large number of very rare categories can increase the dimensionality of one-hot encoding.

Grouping rare categories can:

* Reduce the number of distinct categories.
* Reduce the number of encoded features.
* Make the representation less sparse.
* Reduce dependence on categories with very few observations.

However, rare-category grouping can also discard meaningful distinctions.

The threshold should therefore be treated as a modeling decision rather than a universal rule.

### Important preprocessing consideration

If the rare-category threshold is calculated from the entire dataset before splitting, the test data influences preprocessing decisions.

A more rigorous production workflow learns rare-category grouping rules from the training data alone and applies those same rules to the test data and future predictions.

The project should be extended to enforce that separation if the category-grouping operation currently occurs before the train-test split.

---

## 7.2 Missing-Value Handling in the Prediction Application

The training data and application inputs must use compatible missing-value conventions.

For example:

* An unknown country should be represented consistently.
* Missing company and agent information should map to the corresponding missingness indicators.
* Children values should follow the same assumptions used during training.

A deployed model should not silently interpret a missing input differently from how the corresponding training feature was prepared.

---

# 🧠 Step 8: Feature Selection

Feature selection determines which columns are passed to the model.

The objective is to retain information appropriate for the prediction task while excluding columns that are unavailable at prediction time, redundant for the chosen workflow, or inappropriate for direct use.

The project excludes:

```text
reservation_status
reservation_status_date
assigned_room_type
booking_changes
days_in_waiting_list
company
agent
arrival_date_day_of_month
lead_time_group
```

### Why exclude `lead_time_group`?

`lead_time_group` is an EDA-related derived column.

It is not included in the final 30-feature model input.

If it were accidentally included in the feature matrix, the training and prediction schemas could differ. This could lead to feature-name errors or incorrect predictions.

The project explicitly excludes this column from the final model features.

### Why maintain a consistent feature schema?

The model is trained using a particular set of columns in a particular order.

At prediction time, the application must provide compatible input features.

If the training data contains 30 features but the application sends 29 or 31, the model may fail or receive an unintended representation.

The project therefore validates the feature schema before making predictions.

---

# ✂️ Step 9: Separating Features and Target

Supervised machine learning requires separating the input variables from the target variable.

The conceptual operation is:

```python
X = df[feature_columns]
y = df["is_canceled"]
```

Where:

* `X` contains the 30 selected input features.
* `y` contains the cancellation labels.

The model learns a relationship between the input features and the target:

$$
X \longrightarrow y
$$

During training, the model uses labeled observations to learn patterns associated with cancellation.

During prediction, it receives new input features without needing to know the target in advance.

### Why separate them?

The target must not accidentally become an input feature.

If `is_canceled` were included in `X`, the model would effectively receive the answer during training.

This would invalidate the prediction task.

---

# 🔀 Step 10: Train-Test Split

The dataset is divided into training and testing subsets.

The project uses:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

## Understanding the parameters

### `test_size=0.20`

Twenty percent of the observations are assigned to the test set.

The remaining 80% are assigned to the training set.

For the cleaned dataset of 87,396 records, this corresponds to approximately:

```text
Training observations: 69,916
Testing observations:  17,480
```

The exact allocation is determined by Scikit-learn.

### `random_state=42`

This fixes the random seed used by the split.

It makes the split reproducible when the same data and configuration are used.

The value `42` has no special predictive significance. It is simply a fixed seed.

### `stratify=y`

Stratification preserves approximately the same target-class proportions in both subsets.

Since the dataset contains more non-canceled bookings than canceled bookings, stratification helps prevent an accidental difference in class composition between training and testing data.

## Why split the data?

The model learns from the training set.

The test set is reserved for evaluating how the trained model performs on observations that were not used for fitting.

Without a separate test set, evaluation may overstate how well the model generalizes to unseen observations.

### Important consideration: time-based evaluation

The current workflow uses a stratified random split.

However, hotel reservations are time-dependent.

For a stricter simulation of future deployment, a time-based split may be more appropriate: train on earlier observations and evaluate on later ones.

The best strategy depends on the intended deployment scenario.

---

# ⚙️ Step 11: Identifying Numerical and Categorical Features

The model receives different types of information.

Numerical features represent quantities or counts, while categorical features represent groups or labels.

The project identifies these groups using Pandas:

```python
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()
```

### What does `select_dtypes()` do?

It selects columns based on their data types.

The first operation selects columns whose dtype is `object`.

The second selects columns that do not have that dtype.

The resulting lists are then used to direct different feature groups through different preprocessing operations.

### Numerical features

Examples include:

```text
lead_time
arrival_date_year
arrival_date_week_number
stays_in_weekend_nights
stays_in_week_nights
adults
children
babies
previous_cancellations
previous_bookings_not_canceled
adr
required_car_parking_spaces
total_of_special_requests
total_stay_nights
total_guests
total_guest_nights
```

Other numerical or indicator features are included in the complete feature set.

### Categorical features

Examples include:

```text
hotel
arrival_date_month
meal
country
market_segment
distribution_channel
reserved_room_type
deposit_type
customer_type
```

### Important implementation detail

Selecting columns with `include=["object"]` assumes that the categorical features are stored as object dtype.

If categorical features are represented using Pandas `category` or string dtypes, the selection logic should be adjusted.

For a more robust implementation, explicitly define or validate the numerical and categorical feature lists.

---

# 🔧 Step 12: Building the Preprocessing Pipeline

A machine learning algorithm cannot always work directly with raw tabular data.

Categorical values must be converted into numerical representations, and numerical variables may require scaling.

The project uses:

* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`

These are combined into a reusable preprocessing stage.

## 12.1 StandardScaler for Numerical Features

The StandardScaler transformation is:

$$
z = \frac{x-\mu}{\sigma}
$$

Where:

* \(x\) is the original value.
* \(\mu\) is the mean calculated from the training data.
* \(\sigma\) is the standard deviation calculated from the training data.
* \(z\) is the standardized value.

After standardization, a feature is typically centered around zero and scaled to unit variance, assuming a nonconstant feature.

### Example

Suppose a numerical feature contains values with a mean of 10 and a standard deviation of 2.

For an observation with value 14:

$$
z = \frac{14-10}{2}=2
$$

The standardized value is 2.

### Why use StandardScaler?

Different numerical features can have very different scales.

For example:

* `lead_time` may range from zero to several hundred days.
* `adults` may contain small integer values.
* `adr` may represent a monetary rate.

StandardScaler places these features on a comparable scale.

Random Forest does not generally require standardized numerical inputs because tree splits depend on feature ordering rather than Euclidean distances.

Nevertheless, StandardScaler is included in this project as part of the preprocessing workflow. Its presence should not be interpreted as a requirement for Random Forest.

---

## 12.2 OneHotEncoder for Categorical Features

Most traditional Scikit-learn classifiers require categorical values to be represented numerically.

OneHotEncoder converts a categorical feature into indicator columns.

### Example

Suppose the `hotel` feature contains:

```text
City Hotel
Resort Hotel
```

One-hot encoding represents the categories using separate binary columns.

Conceptually:

| hotel        | hotel_City | hotel_Resort |
| ------------ | ---------: | -----------: |
| City Hotel   |          1 |            0 |
| Resort Hotel |          0 |            1 |

This allows the model to use the category without assigning an arbitrary numerical ordering to the hotel types.

### Why not use Label Encoding for these features?

If City Hotel were encoded as 0 and Resort Hotel as 1, the numbers could be interpreted as having an ordering or numerical distance.

That interpretation is generally inappropriate for nominal categories.

One-hot encoding avoids this issue by representing category membership using indicator columns.

---

## 12.3 Handling Unseen Categories

The encoder uses:

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

### `handle_unknown="ignore"`

This allows the encoder to handle categories that were not observed during fitting.

For an unseen category, the encoder produces zeros for the encoded columns associated with that feature.

This prevents an error simply because a new category appears at prediction time.

However, it does not mean the model has learned the meaning of the new category. The prediction is made without a dedicated learned representation for that unseen value.

### `sparse_output=False`

This configures the encoder to return a dense array instead of a sparse matrix.

A dense representation can be convenient for some workflows, but it may consume more memory when categorical features produce many encoded columns.

For a larger dataset or a substantially higher number of categories, a sparse representation may be more memory-efficient.

---

## 12.4 Understanding ColumnTransformer

`ColumnTransformer` applies different transformations to different columns within the same dataset.

Conceptually:

```text
                    INPUT DATA
                        |
            +-----------+-----------+
            |                       |
            v                       v
     NUMERICAL COLUMNS       CATEGORICAL COLUMNS
            |                       |
            v                       v
      StandardScaler            OneHotEncoder
            |                       |
            +-----------+-----------+
                        |
                        v
             TRANSFORMED FEATURE MATRIX
                        |
                        v
                 RANDOM FOREST
```

A conceptual implementation is:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)
```

This illustrates the preprocessing design. The actual fitted transformer is stored as part of the trained pipeline.

### Why is ColumnTransformer useful?

Without it, numerical and categorical columns would need to be transformed separately and combined manually.

That approach can introduce errors, such as:

* Applying the wrong transformation to a column.
* Losing track of column ordering.
* Using inconsistent transformations between training and prediction.
* Applying preprocessing parameters calculated from the wrong data.

ColumnTransformer coordinates the transformations and returns a single feature matrix suitable for the classifier.

---

## 12.5 Understanding Pipeline

Scikit-learn's `Pipeline` connects preprocessing and model training into one reusable object.

Conceptually:

```text
RAW FEATURES
     |
     v
COLUMN TRANSFORMER
     |
     v
TRANSFORMED FEATURES
     |
     v
RANDOM FOREST CLASSIFIER
     |
     v
PREDICTION
```

The pipeline contains two principal stages:

1. The fitted preprocessing transformer.
2. The Random Forest classifier.

### Why use Pipeline?

It helps ensure that the same preprocessing operations are applied consistently during training and prediction.

When the pipeline is fitted on the training data:

* The scaler learns its numerical statistics.
* The encoder learns the categorical vocabulary.
* The classifier learns the relationship between the transformed features and the target.

When the fitted pipeline receives new data, it applies the learned preprocessing before generating predictions.

This is particularly important in deployment because the application must not independently recreate the preprocessing logic in a way that differs from training.

---

# 🌲 Step 13: Model Training

The primary algorithm used in the project is the Random Forest classifier.

## 13.1 What Is Random Forest?

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Each tree learns decision rules from the training data.

For example, a tree might learn that certain combinations of lead time, deposit type, market segment, and previous cancellations are associated with different cancellation outcomes.

The forest combines the predictions of many trees to produce the final classification.

For classification, the predicted class is generally determined from the aggregate class probabilities or votes of the trees, depending on the implementation.

## 13.2 Why Random Forest?

Random Forest is suitable for investigating relationships in structured tabular data because it can learn:

* Nonlinear relationships.
* Interactions between features.
* Different decision rules across regions of the feature space.
* Relationships that are not captured by a simple linear model.

It also generally requires less feature scaling than distance-based algorithms.

### Limitations

Random Forest is not automatically the best algorithm for every dataset.

Its performance depends on factors such as:

* Number of trees.
* Tree depth.
* Minimum samples required for splits and leaves.
* Feature selection.
* Class distribution.
* Data quality.
* Validation strategy.

A serious model-selection process should compare it with relevant alternatives and evaluate generalization.

---

## 13.3 Training Through the Pipeline

The conceptual training operation is:

```python
pipeline.fit(X_train, y_train)
```

The pipeline performs two operations in sequence.

**First: preprocessing**

The ColumnTransformer learns the numerical scaling parameters and categorical encoding vocabulary from `X_train`.

**Second: model fitting**

The Random Forest classifier learns from the transformed training features and their corresponding target labels.

The test data must not be used to fit the preprocessing transformations or classifier.

### Why this matters

If the scaler or encoder is fitted using the test data, information from the evaluation set can influence the learned preprocessing.

A Pipeline reduces this risk by fitting the transformations as part of the training operation.

---

# 📏 Step 14: Model Evaluation

After training, the model is evaluated using observations that were excluded from fitting.

For classification, multiple metrics should be considered because each describes a different aspect of performance.

## 14.1 Accuracy

Accuracy measures the proportion of predictions that are correct.

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

Where:

* \(TP\): True Positives.
* \(TN\): True Negatives.
* \(FP\): False Positives.
* \(FN\): False Negatives.

Accuracy is useful as an overall measure, but it does not reveal which class the model predicts well.

In an imbalanced dataset, a model can achieve relatively high accuracy while missing many canceled bookings.

---

## 14.2 Precision

Precision measures the proportion of predicted cancellations that were actually cancellations.

$$
Precision=\frac{TP}{TP+FP}
$$

A high precision means that fewer predicted cancellations are false alarms.

For example, a hotel team that manually reviews flagged reservations may care about precision because too many false alarms can waste operational resources.

---

## 14.3 Recall

Recall measures the proportion of actual cancellations that the model correctly identifies.

$$
Recall=\frac{TP}{TP+FN}
$$

A high recall means the model identifies a larger proportion of canceled bookings.

Recall can be particularly important when failing to identify a cancellation is considered costly.

However, increasing recall can also increase false positives.

---

## 14.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F1=2 \times
\frac{Precision \times Recall}
{Precision+Recall}
$$

It is useful when both precision and recall matter.

Unlike accuracy, it focuses on the positive class's precision and recall rather than counting all correct predictions equally.

---

## 14.5 ROC-AUC

ROC-AUC measures how well the model ranks positive examples above negative examples across classification thresholds.

It is based on the Receiver Operating Characteristic curve, which compares the true-positive rate with the false-positive rate at different thresholds.

A higher ROC-AUC generally indicates better ranking discrimination.

However, ROC-AUC does not directly specify the best operating threshold, nor does it guarantee that predicted probabilities are well calibrated.

---

## 14.6 Confusion Matrix

A confusion matrix compares predicted classes with actual classes.

|                      | Predicted: Not Canceled | Predicted: Canceled |
| -------------------- | ----------------------- | ------------------- |
| Actual: Not Canceled | True Negative           | False Positive      |
| Actual: Canceled     | False Negative          | True Positive       |

The four outcomes represent different types of prediction.

### False Positive

The model predicts cancellation, but the booking is not canceled.

### False Negative

The model predicts no cancellation, but the booking is canceled.

The operational costs of these errors may differ substantially.

For example, a false negative may cause the hotel to overestimate expected occupancy. A false positive may cause unnecessary intervention.

The preferred model should therefore reflect the business objective rather than rely solely on one metric.

---

## 14.7 Classification Threshold

A classifier that produces probabilities needs a threshold to convert those probabilities into class predictions.

The application uses the default binary classification threshold of 0.5.

Conceptually:

```python
prediction = (
    model.predict_proba(input_data)[:, 1] >= 0.5
).astype(int)
```

This illustrates threshold-based classification. The application may use the pipeline's `predict()` method for the final class instead.

### Why consider threshold tuning?

The threshold determines the trade-off between precision and recall.

A lower threshold generally flags more bookings as potential cancellations, often increasing recall while reducing precision.

A higher threshold generally flags fewer bookings, potentially increasing precision while reducing recall.

The best threshold depends on the relative costs of false positives and false negatives.

### Important evaluation rule

Threshold selection should be performed using training data through cross-validation or a separate validation set.

The final test set should remain untouched until the model and threshold decisions have been finalized.

A threshold selected after repeatedly examining test-set results produces an optimistic estimate of generalization performance.

---

# 💾 Step 15: Saving the Trained Pipeline

After training, the complete fitted pipeline is saved to disk.

The model artifact is:

```text
hotel_cancellation_pipeline.pkl
```

The project uses Joblib for model persistence.

Conceptually:

```python
import joblib

joblib.dump(
    pipeline,
    "hotel_cancellation_pipeline.pkl"
)
```

The exact variable name in the training notebook may differ.

## Why save the entire pipeline?

Saving only the Random Forest classifier would omit the learned preprocessing operations.

The deployed application would then need to recreate:

* Numerical scaling.
* Categorical encoding.
* Feature ordering.
* The classifier's input representation.

That introduces unnecessary complexity and creates opportunities for inconsistent transformations.

Saving the complete pipeline preserves the fitted preprocessing and classification stages in one artifact.

## Loading the Pipeline

The application loads the saved artifact:

```python
import joblib
from pathlib import Path

model_path = (
    Path(__file__).resolve().parent
    / "hotel_cancellation_pipeline.pkl"
)

model = joblib.load(model_path)
```

Using `Path(__file__).resolve().parent` makes the model path relative to the application's directory.

This is more reliable than depending on the terminal's current working directory.

### Model compatibility

Scikit-learn does not guarantee that serialized models will remain compatible across arbitrary library versions.

The Python and Scikit-learn versions used for training should therefore be documented and aligned with the deployment environment wherever possible.

The `.pkl` artifact should also be loaded only from a trusted source because pickle-based formats can execute code during deserialization.

---

# 🖥️ Step 16: Streamlit Application

The trained model is integrated into a Streamlit web application.

The application provides a user interface for entering reservation details and obtaining a prediction.

## 16.1 Application Initialization

The application imports its required libraries and loads the saved pipeline.

Typical dependencies include:

```python
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
```

The application also uses the relevant Python date utilities and logic required to prepare model inputs.

The saved pipeline is loaded so that the same preprocessing and classification operations used during training can be reused.

---

## 16.2 User Input

The interface collects reservation information corresponding to the model's expected input features.

Examples include:

* Hotel type.
* Lead time.
* Arrival date.
* Length of stay.
* Number of adults, children, and babies.
* Meal plan.
* Country.
* Market segment.
* Distribution channel.
* Repeated-guest status.
* Previous booking history.
* Reserved room type.
* Deposit type.
* Customer type.
* Average daily rate.
* Parking requirements.
* Special requests.

The application converts the user's selections into a structured record.

The exact input controls and defaults are defined in `app.py`.

---

## 16.3 Constructing the Input DataFrame

The application creates a Pandas DataFrame containing the user's reservation details.

Conceptually:

```python
input_df = pd.DataFrame([booking_data])
```

The resulting DataFrame represents one reservation.

A DataFrame is useful because Scikit-learn can use the column names to validate the incoming feature schema.

The application must supply the 30 expected features with compatible meanings and data types.

---

## 16.4 Feature Consistency Checks

Before prediction, the application checks whether the input features match the model's expected feature schema.

This is important because the model was trained using a fixed set of columns.

The application should not silently omit a feature or substitute a different column merely because its name or position appears similar.

Schema validation helps identify problems before they reach the prediction stage.

---

## 16.5 Prediction

The fitted pipeline receives the input DataFrame.

The conceptual operation is:

```python
prediction = model.predict(input_df)
```

Internally, the pipeline:

1. Selects the expected numerical and categorical columns.
2. Applies the fitted numerical scaler.
3. Applies the fitted categorical encoder.
4. Combines the transformed features.
5. Passes the transformed data to the Random Forest classifier.
6. Returns a predicted class.

The application then interprets the class.

```text
Prediction = 0
        |
        v
Booking predicted as not canceled


Prediction = 1
        |
        v
Booking predicted as canceled
```

The prediction is an estimate learned from historical data, not a guarantee of the reservation's future outcome.

---

## 16.6 Prediction Probability

The application can also obtain the estimated probability associated with the cancellation class:

```python
probability = model.predict_proba(input_df)[0, 1]
```

This extracts the model's estimated probability for class `1`, assuming the model's class ordering is the standard `[0, 1]`.

The output can be displayed as a percentage:

```python
probability_percentage = probability * 100
```

For example, a model output of `0.72` corresponds to an estimated cancellation probability of 72%.

**Important:** A predicted probability is not automatically a well-calibrated real-world probability. Calibration should be evaluated separately if the application will use probability values for operational decisions.

---

## 16.7 Displaying the Result

The application displays the predicted class and, where implemented, the associated probability.

The interface provides a more accessible way to interact with the model than manually preparing a DataFrame and calling Python functions.

It separates the user interaction layer from the trained machine learning pipeline.

---

# 🔁 How the Prediction System Works

The application can be understood as a sequence of operations.

```text
                 USER OPENS APP
                        |
                        v
             ENTER RESERVATION DETAILS
                        |
                        v
             APPLICATION BUILDS DATAFRAME
                        |
                        v
             VALIDATE EXPECTED FEATURES
                        |
                        v
               LOAD FITTED PIPELINE
                        |
                        v
              TRANSFORM INPUT FEATURES
                 /              \
                v                v
           NUMERICAL          CATEGORICAL
           SCALING            ENCODING
                \                /
                 \              /
                  v            v
                  RANDOM FOREST
                        |
                        v
                PREDICT CANCELLATION
                        |
              +---------+---------+
              |                   |
              v                   v
       PREDICTED CLASS      ESTIMATED PROBABILITY
              |                   |
              +---------+---------+
                        |
                        v
                 DISPLAY RESULT
```

### Training versus prediction

Training and prediction are separate processes.

**Training**

The notebook learns the preprocessing parameters and fits the classifier using historical labeled data.

**Prediction**

The deployed application uses those learned parameters and the fitted classifier to evaluate new reservation inputs.

The application does not retrain the model every time a user submits a booking.

This separation makes the application faster and helps ensure that its predictions are based on a consistent model artifact.

---

# 📁 Project Structure

The project repository contains the following core files:

```text
hotel-booking-cancellation/
│
├── app.py
│
├── requirements.txt
│
├── hotel_cancellation_pipeline.pkl
│
└── README.md
```

### `app.py`

Contains the Streamlit application, including user input controls, input preparation, schema checks, prediction logic, and result display.

### `requirements.txt`

Lists the Python dependencies required to run the application.

### `hotel_cancellation_pipeline.pkl`

Contains the saved, fitted preprocessing and classification pipeline.

### `README.md`

Documents the project's objective, methodology, implementation, setup, and limitations.

The exploratory notebook and original dataset may be maintained separately if they are not included in this deployment repository.

For a more comprehensive portfolio, the repository can later be expanded to include the notebook, model evaluation results, and a clearly documented data acquisition process.

---

# 🚀 Installation and Local Execution

Follow these steps to run the application locally.

## Step 1: Clone the Repository

```bash
git clone https://github.com/Siddharthcdt25/hotel-booking-cancellation.git
```

Navigate into the project directory:

```bash
cd hotel-booking-cancellation
```

## Step 2: Create a Virtual Environment

Using Python's built-in virtual environment module:

```bash
python -m venv .venv
```

A virtual environment isolates the project's dependencies from other Python projects.

### Activate on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by the machine's execution policy, use an appropriate environment-specific solution rather than changing system-wide security settings unnecessarily.

## Step 3: Install Dependencies

Run:

```bash
python -m pip install --upgrade pip
```

Then:

```bash
pip install -r requirements.txt
```

This installs the libraries listed in `requirements.txt` and their required dependencies.

## Step 4: Verify the Project Files

Ensure the project directory contains:

```text
app.py
requirements.txt
hotel_cancellation_pipeline.pkl
```

The application needs both the Python code and the trained model artifact.

## Step 5: Start the Application

Run:

```bash
streamlit run app.py
```

Streamlit starts a local web server and displays the application in your browser.

If the browser does not open automatically, use the local URL shown in the terminal.

---

# ☁️ Deployment

The application is intended to be deployed using Streamlit Community Cloud.

The deployment process uses the GitHub repository as the source of application code.

## Deployment Workflow

```text
LOCAL PROJECT
      |
      v
COMMIT CHANGES WITH GIT
      |
      v
PUSH TO GITHUB
      |
      v
CONNECT GITHUB REPOSITORY
TO STREAMLIT COMMUNITY CLOUD
      |
      v
SELECT MAIN BRANCH
      |
      v
SELECT app.py
      |
      v
INSTALL REQUIREMENTS
      |
      v
LOAD SAVED MODEL
      |
      v
RUN STREAMLIT APPLICATION
```

## Deployment Requirements

The repository should contain:

* `app.py`
* `requirements.txt`
* `hotel_cancellation_pipeline.pkl`

The deployed environment must have compatible Python and library versions.

### Dependency management

For reproducibility, the versions of critical dependencies should be aligned with those used to train the model.

This is particularly important for Scikit-learn because the serialized pipeline contains fitted estimators and transformers.

A successful package installation does not, by itself, guarantee that the saved model can be loaded successfully.

The application must also start and complete a prediction successfully.

### Deployment validation

After deployment, verify the following:

1. The application starts without errors.
2. The model artifact loads successfully.
3. The input form is displayed correctly.
4. A valid reservation can be submitted.
5. The prediction is returned successfully.
6. The displayed result corresponds to the model output.
7. The application handles supported input combinations without unexpected exceptions.

The deployment should be considered complete only after these checks have been performed.

---

# 🧠 Important Machine Learning Concepts Applied

This project demonstrates several important concepts in applied machine learning.

## 1. Supervised Learning

The training dataset contains both input features and known target labels.

The model learns a relationship between reservation characteristics and historical cancellation outcomes.

## 2. Binary Classification

The target contains two classes:

```text
0 = Not canceled
1 = Canceled
```

The objective is to classify each reservation into one of these categories.

## 3. Exploratory Data Analysis

EDA helps investigate distributions, missingness, unusual records, class imbalance, and relationships between variables.

It informs modeling decisions rather than directly creating the prediction model.

## 4. Feature Engineering

Feature engineering creates new representations from existing data.

Examples include:

* `total_stay_nights`
* `total_guests`
* `zero_guests`
* `total_guest_nights`
* `arrival_day_of_week`

## 5. Categorical Encoding

OneHotEncoder converts categorical values into numerical indicator features.

This allows the classifier to process categorical booking information.

## 6. Feature Scaling

StandardScaler standardizes numerical features using statistics learned from the training data.

It is included in the pipeline, although Random Forest generally does not require this transformation.

## 7. ColumnTransformer

ColumnTransformer applies different preprocessing operations to different groups of columns.

It keeps the preprocessing logic organized and reusable.

## 8. Pipeline

Pipeline combines preprocessing and model fitting into a single estimator.

It helps prevent inconsistencies between training and prediction.

## 9. Stratified Sampling

The train-test split uses stratification to preserve approximately the same class proportions in both subsets.

## 10. Model Evaluation

To evaluate the performance of the cancellation prediction system, two classification algorithms were compared:

Logistic Regression — A linear classification algorithm used as a baseline model.
Random Forest Classifier — An ensemble learning algorithm that combines multiple decision trees to capture nonlinear relationships and feature interactions.

Both models were evaluated using five metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.

1. Performance Comparison
Metric	Logistic Regression	Random Forest	Improvement
Accuracy	79.05%	82.50%	+3.45 percentage points
Precision	67.33%	75.65%	+8.32 percentage points
Recall	46.20%	53.59%	+7.39 percentage points
F1-score	54.80%	62.74%	+7.94 percentage points
ROC-AUC	83.39%	88.76%	+5.37 percentage points

Result: Random Forest achieved better performance than Logistic Regression across all five evaluation metrics and was selected as the final model.

Note: The improvements above are absolute percentage-point differences, not relative percentage increases.

2. Understanding the Evaluation Metrics
Accuracy — 82.50%

Accuracy measures the proportion of all test observations classified correctly.

[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
]

The Random Forest model achieved an accuracy of 82.50%, compared with 79.05% for Logistic Regression.

This indicates that Random Forest correctly classified a larger proportion of the test observations.

However, accuracy alone is insufficient because the dataset contains more non-canceled bookings than canceled bookings.

Precision — 75.65%

Precision measures the proportion of bookings predicted as canceled that were actually canceled.

[
Precision = \frac{TP}{TP+FP}
]

The Random Forest model achieved 75.65% precision.

This means that, among the bookings it predicted as canceled, approximately 75.65% belonged to the canceled class in the evaluated test set.

Higher precision can help reduce false alarms when a hotel uses predictions to identify bookings for additional review.

Recall — 53.59%

Recall measures the proportion of actual cancellations correctly identified by the model.

[
Recall = \frac{TP}{TP+FN}
]

The Random Forest model achieved 53.59% recall.

This means that the model identified approximately 53.59% of the actual cancellations in the test set.

Recall is important because missed cancellations can affect occupancy forecasting and operational planning.

However, the current result also reveals an area for improvement: the model did not identify approximately 46.41% of actual cancellations at the evaluated operating threshold.

F1-score — 62.74%

The F1-score is the harmonic mean of precision and recall.

[
F1 = 2 \times
\frac{Precision \times Recall}
{Precision+Recall}
]

The Random Forest model achieved an F1-score of 62.74%, compared with 54.80% for Logistic Regression.

This indicates a better balance between precision and recall for Random Forest at the evaluated classification threshold.

F1-score is particularly useful when both false positives and false negatives matter.

ROC-AUC — 88.76%

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.

The Random Forest model achieved a ROC-AUC of 88.76%, compared with 83.39% for Logistic Regression.

This suggests that Random Forest has better class-ranking discrimination than Logistic Regression on the evaluated data.

ROC-AUC does not, by itself, establish the best classification threshold or guarantee that predicted probabilities are calibrated.

3. Why Did Random Forest Perform Better?

Logistic Regression models the log-odds of the target as a linear combination of the input features.

Random Forest combines multiple decision trees, allowing it to learn nonlinear decision boundaries and interactions between features.

Hotel cancellation behavior may involve interactions between variables such as:

Lead time.
Deposit type.
Market segment.
Customer type.
Previous cancellations.
Length of stay.
Special requests.

For example, the relationship between lead time and cancellation risk may differ depending on the deposit type or market segment.

Random Forest can learn such interactions without requiring every interaction term to be manually specified.

Its stronger results suggest that it captured useful patterns in this dataset more effectively than the Logistic Regression baseline.

However, the comparison does not prove that Random Forest will outperform Logistic Regression on every hotel booking dataset or in every deployment environment.

4. Model Selection Decision

Random Forest was selected as the final model because it achieved the highest score across all five reported metrics.

The comparison showed improvements in:

Overall classification accuracy.
Precision for predicted cancellations.
Recall of actual cancellations.
Balance between precision and recall.
Discrimination between canceled and non-canceled bookings.

The selected model was subsequently integrated into the project's preprocessing pipeline and Streamlit application.

5. Remaining Improvement Opportunity

Although Random Forest achieved the strongest results, its recall of 53.59% indicates that many actual cancellations remain undetected.

Future experiments could investigate:

Hyperparameter tuning using cross-validation.
Class-weighting strategies.
Threshold selection using validation data.
Additional feature engineering.
Alternative tree-based algorithms.
Time-based validation to simulate predictions on future bookings.

Threshold selection should reflect the relative costs of false positives and false negatives. A lower threshold may improve recall but can also reduce precision.

Any threshold should be selected using training or validation data rather than repeatedly optimized against the final test set.

6. Evaluation Summary

Random Forest achieved an accuracy of 82.50%, precision of 75.65%, recall of 53.59%, F1-score of 62.74%, and ROC-AUC of 88.76%.

Compared with Logistic Regression, it provided stronger performance across all reported metrics and was selected as the final model for the deployed prediction application.

The results provide a useful baseline for further optimization, particularly for improving cancellation detection while controlling false-positive predictions.

## 11. Model Persistence

Joblib saves the fitted pipeline so it can be reused without retraining.

## 12. Application Integration

Streamlit provides an interface for supplying reservation information and viewing model predictions.

---

# ⚠️ Challenges and Solutions

The following table summarizes important implementation challenges addressed by the project.

| Challenge                                         | Approach                                                  | Reason                                                                              |
| ------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Duplicate observations                            | Remove exact duplicate rows                               | Reduce repeated influence from identical records                                    |
| Missing values                                    | Apply column-specific handling                            | Preserve useful information and maintain valid inputs                               |
| High missingness in company and agent identifiers | Exclude raw identifiers and retain missingness indicators | Avoid directly modeling sparse identifiers while preserving missingness information |
| High-cardinality country feature                  | Group rare categories and retain an unknown category      | Control categorical dimensionality                                                  |
| Outcome-related information                       | Exclude reservation status and status date                | Reduce target leakage                                                               |
| Multiple numerical scales                         | Apply StandardScaler                                      | Maintain a consistent numerical transformation                                      |
| Categorical variables                             | Apply OneHotEncoder                                       | Represent categories numerically without imposing an arbitrary ordering             |
| Different preprocessing requirements              | Use ColumnTransformer                                     | Apply the appropriate transformation to each feature group                          |
| Training-prediction inconsistency                 | Save the complete fitted Pipeline                         | Reuse the same preprocessing and model                                              |
| Class imbalance                                   | Use stratified splitting and evaluate multiple metrics    | Preserve class proportions and avoid relying on accuracy alone                      |
| Deployment environment differences                | Pin critical library versions and verify compatibility    | Reduce serialization and dependency problems                                        |
| Feature-schema mismatch                           | Validate the expected input features                      | Prevent invalid model inputs                                                        |

---

# 🔬 Limitations and Future Improvements

The current project provides an end-to-end baseline, but several areas could be improved before using the system in a real hotel operation.

## 1. Compare Multiple Models

Random Forest is the primary classifier.

Future work could compare it with:

* Logistic Regression.
* Decision Tree.
* Gradient Boosting.
* HistGradientBoostingClassifier.
* Other suitable gradient-boosted tree implementations.

The comparison should use the same evaluation strategy and feature availability assumptions.

A more complex model should not be selected solely because it is more sophisticated.

## 2. Hyperparameter Tuning

Random Forest performance depends on its hyperparameters.

Useful parameters to investigate include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
```

Cross-validation and a separate validation strategy can be used to evaluate alternative configurations.

The test set should remain reserved for the final evaluation.

## 3. Evaluate Class-Weighting Strategies

The target is imbalanced.

One possible experiment is to compare the current model with a Random Forest using:

```python
class_weight="balanced"
```

This changes how the classifier accounts for the class frequencies during fitting.

However, class weighting is not guaranteed to improve every metric. The resulting precision, recall, F1-score, ROC-AUC, and business trade-offs should be evaluated.

## 4. Calibrate Predicted Probabilities

The application can display estimated cancellation probabilities.

However, probability calibration should be evaluated if these values will be interpreted as real-world risks.

Calibration methods can help determine whether predictions assigned a given probability correspond to approximately the same observed event frequency.

## 5. Improve the Evaluation Strategy

The current workflow uses a stratified random split.

A time-based evaluation could better simulate deployment on future reservations.

This would help investigate whether performance changes over time and whether the model generalizes beyond randomly held-out observations.

## 6. Define the Prediction Time Precisely

The intended prediction time should be documented explicitly.

For example:

* Immediately after a reservation is created.
* A fixed number of days before arrival.
* At another defined point in the booking lifecycle.

The appropriate features differ depending on this choice.

A feature that is valid several days before arrival may not be available immediately after booking.

The model should use only information available at the chosen prediction time.

## 7. Improve Rare-Category Handling

Rare-category grouping should ideally be learned from training data and then applied consistently to validation, test, and production data.

This prevents evaluation data from influencing preprocessing decisions.

## 8. Add Model Monitoring

A production system should monitor:

* Changes in input distributions.
* Changes in cancellation rates.
* Prediction distributions.
* Model performance when actual outcomes become available.
* Changes in the frequency of previously unseen categories.

This can help identify data drift and performance degradation.

## 9. Improve Application Testing

Future testing should include:

* Typical reservation inputs.
* Unusual guest counts.
* Missing or unknown categorical values.
* Different hotel types.
* Different deposit types.
* Boundary values for numerical inputs.
* Feature-schema mismatches.
* Model-loading failures.

Input validation should prevent invalid reservation values from being passed to the classifier.

## 10. Improve Reproducibility

Future versions should document:

* The dataset source and version.
* The cleaning operations.
* The final feature schema.
* The train-test split.
* The model configuration.
* The Python and library versions.
* The evaluation results.
* The model artifact version.

These details make it easier to reproduce, audit, and maintain the model.

---

# 📚 Key Learnings

This project demonstrates how individual machine learning techniques fit into a complete prediction workflow.

### Data Analysis

Understanding the data before training helps identify quality issues, unusual observations, and potentially informative patterns.

### Feature Engineering

Combining related columns can produce more interpretable representations of reservation size, duration, and arrival timing.

### Leakage Prevention

A feature can be statistically informative but still be unsuitable if it reveals information unavailable at prediction time.

### Preprocessing

Numerical and categorical features often require different transformations.

### ColumnTransformer

ColumnTransformer coordinates feature-specific transformations and returns a single transformed feature matrix.

### Pipeline

Pipeline connects preprocessing and classification, making the trained workflow easier to reproduce and deploy.

### Model Evaluation

Accuracy alone does not adequately describe performance on an imbalanced classification problem.

### Deployment

A trained model becomes an application component when it is packaged with compatible dependencies, a stable input schema, and reliable prediction logic.

### End-to-End Thinking

The success of a machine learning project depends on more than selecting an algorithm. Data quality, evaluation design, feature availability, preprocessing consistency, and deployment compatibility all affect the final result.

---

# 👨‍💻 Author

**Siddharth Kashyap**

GitHub: [Siddharthcdt25](https://github.com/Siddharthcdt25)

Project Repository: [Hotel Booking Cancellation Prediction](https://github.com/Siddharthcdt25/hotel-booking-cancellation)

---

## ⭐ Final Note

This project demonstrates an end-to-end machine learning workflow for predicting hotel booking cancellations, from historical data analysis to a deployed prediction interface.

The main objective is to develop a reusable, understandable, and maintainable prediction system while applying core machine learning engineering principles.

The current implementation serves as a baseline that can be improved through rigorous model comparison, time-aware validation, probability calibration, and production monitoring.
