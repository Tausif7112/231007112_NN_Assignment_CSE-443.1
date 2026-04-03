
# Problem Set 02: Predicting Term Deposit Subscription

## Overview

For this problem set, our goal was to build a machine learning model for a banking institution. They want to predict whether a potential customer will subscribe to a term deposit based on their banking behavior and demographic information. We tackled this as a binary classification problem using **Logistic Regression**, which is a great choice for 'Yes'/'No' type predictions.

## Dataset

We used the "Bank Marketing Data Set," which contains 17 attributes about past customers. This dataset includes details like age, job, marital status, education, account balance, and campaign-related information, ultimately culminating in whether the customer subscribed to a term deposit ('y' - yes/no).

The dataset can be downloaded from this Google Drive link: `https://drive.google.com/file/d/1Jg0hqQQBOAt3GevTYh9SVXB7tZWlAQ_Q/view?usp=drive_link`

## Approach

Our journey through this problem followed a standard machine learning pipeline, carefully considering each step. The overall approach involved Data Loading, Exploratory Data Analysis (EDA), Data Preprocessing, Model Training, and iterative Model Improvement, all aimed at building a robust Logistic Regression model.

## Methodology

### 1. Data Loading and Initial Exploration (EDA)

First, we loaded the dataset and took an initial look (`df.head()`, `df.info()`). A crucial step here was identifying and handling missing values, which were represented as 'unknown' strings or '-1' in some columns (e.g., `poutcome`, `pdays`, `contact`, `education`, `job`, `balance`). We decided to drop columns with a very high percentage of missing values (`poutcome`, `pdays`) and imputed others using the mode (for categorical) or median (for numerical) to keep our dataset clean. We also checked for and removed any duplicate rows.

We then dove into some visualizations to understand our data better:

*   **Correlation Matrix:** Examined relationships between numerical features and our target variable (`y`). We found `duration` had the strongest positive correlation with a 'yes' subscription, while `campaign` had a weak negative correlation.

    *To embed here: Correlation Heatmap of Numerical Features*

*   **Categorical Features:** Visualized the proportion of 'Yes' responses across different categories (`job`, `marital`, `education`, `contact`, etc.) to see which segments were more likely to subscribe. For instance, students and retired individuals showed higher subscription rates.

    *To embed here: Example Bar Plot of 'Yes' Proportion by a Categorical Feature (e.g., 'job' or 'education')*

*   **Numerical Features:** Used KDE plots and box plots to see how the distribution of features like `age`, `balance`, and `duration` varied between 'yes' and 'no' subscribers.

    *To embed here: Example KDE/Box Plot of a Numerical Feature by Target (e.g., 'duration')*

*   **Temporal Features:** Explored subscription rates across `month` and `day` to identify potential trends.

    *To embed here: Bar Plot of 'Yes' Proportion by 'month'*

*   **Target Variable Distribution:** An important finding was the severe class imbalance, with far fewer 'yes' subscriptions than 'no's. This meant we'd need strategies to address it.

    *To embed here: Count Plot of Target Variable Distribution*

### 2. Data Preprocessing

With our insights from EDA, we prepared the data for modeling:

*   **Feature Engineering:** We created new features like `has_credit` (combining `default`, `housing`, `loan`), `duration_per_campaign`, and `log_balance` (to handle the skewed `balance` distribution).
*   **Categorical Encoding:** Converted all categorical features into a numerical format suitable for our model using **One-Hot Encoding**. We opted for `drop_first=True` to avoid multicollinearity.
*   **Train-Test Split:** Divided our dataset into training (80%) and testing (20%) sets, ensuring stratification to maintain the original class distribution in both sets.
*   **Feature Scaling:** Applied `StandardScaler` to our numerical features to normalize their range, which is crucial for Logistic Regression.
*   **Handling Imbalanced Data (SMOTE):** To counter the class imbalance observed earlier, we used **SMOTE (Synthetic Minority Over-sampling Technique)** on our training data to create synthetic samples of the minority class ('yes'), balancing the dataset for model training.

### 3. Model Training and Evaluation (Logistic Regression)

*   **Initial Model:** We trained a Logistic Regression model using the SMOTE-resampled training data. We used `class_weight='balanced'` and `solver='liblinear'` (or `saga`) for robustness.
*   **Hyperparameter Tuning:** Employed `HalvingGridSearchCV` to efficiently search for the best hyperparameters (`C`, `penalty`, `solver`) that maximize the `roc_auc` score. The best model was then selected.
*   **Model Evaluation:** Assessed the model's performance on the unseen test set using:
    *   **Classification Report:** Providing precision, recall, f1-score, and support for both classes.
    *   **ROC AUC Score:** A measure of the model's ability to distinguish between classes.
    *   **Confusion Matrix:** Visualizing true positives, true negatives, false positives, and false negatives.

        *To embed here: Confusion Matrix for Initial Best Model*

    *   **ROC Curve:** Plotting the Receiver Operating Characteristic curve.

        *To embed here: ROC Curve for Initial Best Model*

    *   **Feature Importance:** Examined the coefficients of the Logistic Regression model to understand which features were most influential in predicting subscription.

        *To embed here: Bar Plot of Top 10 Feature Coefficients*

    *   **Learning Curve:** Plotted the learning curve to diagnose potential overfitting or underfitting (ours indicated some overfitting).

        *To embed here: Learning Curve Plot*

### 4. Model Improvement Attempts

Given the initial results, especially the lower precision for the minority class, we tried several strategies to improve the model:

*   **Feature Selection (RFE):** Used Recursive Feature Elimination to select a subset of features. This actually led to a slight decrease in AUC and precision, suggesting the original feature set (or a different selection) was better.
*   **Optimizing Classification Threshold:** This was a game-changer! By analyzing the precision-recall curve, we found an optimal threshold (around 0.58) that significantly boosted the precision for the 'yes' class from **0.41 to 0.45** and the F1-score from **0.51 to 0.52** compared to the default 0.5 threshold.
*   **ADASYN Resampling:** Explored ADASYN as an alternative to SMOTE for handling imbalance. While it slightly improved recall, it decreased precision and AUC compared to our SMOTE + threshold optimized model.
*   **Ensemble Methods (Bagging):** Implemented `BaggingClassifier` with Logistic Regression as the base estimator. This approach yielded performance metrics very similar to our initial model but was outperformed by the threshold-optimized model in terms of 'yes' class precision and F1-score.

## Key Findings

Our analysis clearly showed that **optimizing the classification threshold was the most effective strategy** for improving our Logistic Regression model for this problem. It allowed us to significantly increase the precision for the 'yes' class from **0.41 to 0.45** and the F1-score from **0.51 to 0.52** for predicting 'yes' subscriptions, which is often crucial for business applications (e.g., reducing wasted marketing efforts).

While other techniques like RFE, ADASYN, and Bagging were explored, they either didn't offer a substantial improvement or slightly degraded performance compared to our best approach (SMOTE-resampled data with an optimized classification threshold).

*To embed here: Performance Comparison Table of all models*

## Conclusion

The final model, a Logistic Regression model trained on SMOTE-resampled data and employing an optimized classification threshold (0.58), provides a good balance. It demonstrates a strong ability to identify potential subscribers while significantly improving the reliability of those 'yes' predictions (higher precision) without sacrificing too much recall. This optimized model is the best performer among the tested approaches for this problem.

*To embed here: Confusion Matrix for the Optimal Threshold Model (our best model)*

*To embed here: ROC Curve for the Optimal Threshold Model (our best model)*

## How to Run the Code

1.  **Clone the Repository:** If you haven't already, clone this repository to your local machine or open it directly in Google Colab.
2.  **Open the Notebook:** Navigate to the `problem_set_02` directory and open the `.ipynb` file in Google Colab.
3.  **Run Cells Sequentially:** Execute all code cells in sequential order. Make sure to address any package installation prompts (e.g., `gdown`, `imblearn`).

## Figures and Visualizations

Throughout the Colab notebook, various plots and visualizations are generated (e.g., correlation heatmaps, categorical bar plots, numerical distributions, confusion matrices, ROC curves, learning curves). For your submission, you should export the most relevant figures (e.g., as PNGs) from the notebook's output and embed them here in your `README.md` to visually support your findings. You can do this using standard Markdown image syntax:

```markdown
![Description of Figure](path/to/your/figure.png)
```
