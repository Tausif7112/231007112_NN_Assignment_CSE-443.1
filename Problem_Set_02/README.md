
# Predicting Term Deposit Subscription

## Overview

For this problem set, our goal was to build a machine learning model for a banking institution. They want to predict whether a potential customer will subscribe to a term deposit based on their banking behavior and demographic information. We tackled this as a binary classification problem using **Logistic Regression**, which is a great choice for 'Yes'/'No' type predictions.

## Dataset

We used the "Bank Marketing Data Set," which contains 17 attributes about past customers. This dataset includes details like age, job, marital status, education, account balance, and campaign-related information, ultimately culminating in whether the customer subscribed to a term deposit or not ('y' - yes/no).

The dataset can be downloaded from this Google Drive link: [Bank Marketing Data Set (Google Drive)](https://drive.google.com/file/d/1Jg0hqQQBOAt3GevTYh9SVXB7tZWlAQ_Q/view?usp=drive_link)

## Approach

Our journey through this problem followed a standard machine learning pipeline, carefully considering each step. The overall approach involved:
* Loading data
* Exploratory Data Analysis (EDA),
* Data Preprocessing
* Model Training and
* Iterative Model Improvement

All aimed at building a robust Logistic Regression model.

## Methodology

### 1. Data Loading and Initial Exploration (EDA)

First, we loaded the dataset and took an initial look (`df.head()`, `df.info()`). A crucial step here was identifying and handling missing values, which were represented as 'unknown' strings or '-1' in some columns (e.g., `poutcome`, `pdays`, `contact`, `education`, `job`, `balance`). We decided to drop columns with a very high percentage of missing values (`poutcome`, `pdays`) and imputed others using the mode (for categorical) or median (for numerical) to keep our dataset clean. We also checked for and removed any duplicate rows.

We then dove into some visualizations to understand our data better:

*   **Correlation Matrix:** Examined relationships between numerical features and our target variable (`y`). We found `duration` had the strongest positive correlation with a 'yes' subscription, while `campaign` had a weak negative correlation.

<img width="897" height="827" alt="heatmap_numercial" src="https://github.com/user-attachments/assets/096bd63f-b8b1-4304-a623-0acd350e3570" />

<p align="center">
  <em>Figure 1:  Correlation Heatmap of Numerical Features. </em>
</p>
 

*   **Categorical Features:** Visualized the proportion of 'Yes' responses across different categories (`job`, `marital`, `education`, `contact`, etc.) to see which segments were more likely to subscribe. For instance, students and retired individuals showed higher subscription rates.

<img width="1217" height="682" alt="Yes proportion with respect to Job" src="https://github.com/user-attachments/assets/65bd0ac5-184c-460b-a52c-e07ae76c7575" />
<p align="center">
  <em>Figure 2: Bar Plot of 'Yes' Proportion by a Categorical Feature (e.g., 'job' )   </em>
</p>

     

*   **Numerical Features:** Used KDE plots and box plots to see how the distribution of features like `age`, `balance`, and `duration` varied between 'yes' and 'no' subscribers.
      
<img width="1407" height="593" alt="Boxplot of duration vs target" src="https://github.com/user-attachments/assets/459a0a3a-e110-4248-808b-24f71758b699" />

<p align="center">
  <em>Figure 3: KDE and Box Plot of a Numerical Feature by Target (e.g., 'duration')  </em>
</p>
     

*   **Temporal Features:** Explored subscription rates across `month` and `day` to identify potential trends.
<img width="1213" height="673" alt="Yes proportion by Month" src="https://github.com/user-attachments/assets/711a12bd-66a1-412b-9187-e94cb4ef06e5" />

<p align="center">
  <em>Figure 4: Bar Plot of 'Yes' Proportion by 'month'.  </em>
</p>
 

*   **Target Variable Distribution:** An important finding was the severe class imbalance, with far fewer 'yes' subscriptions than 'no's. This meant we'd need strategies to address it.
<p align="center">
<img width="576" height="383" alt="Count_target" src="https://github.com/user-attachments/assets/8eabdff9-18fd-4232-ba1f-5e6dd4ae92b5" />
</p>
<p align="center">
  <em>Figure 5: Count Plot of Target Variable Distribution .  </em>
</p>
    

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
        <p align="center">
        <img width="555" height="462" alt="Confusion-matrix(initial model)" src="https://github.com/user-attachments/assets/f549adea-048b-429c-9c62-473f66f1cdf3" />
        </p>
        <p align="center"><em>Figure 6: Confusion Matrix for Initial Best Model.  </em></p>
        

    *   **ROC Curve:** Plotting the Receiver Operating Characteristic curve.

        <p align="center">
        <img width="715" height="461" alt="ROC(initial model)" src="https://github.com/user-attachments/assets/52330da4-eaae-49ae-9fbd-3cb61638be2f" />
        </p>
         <p align="center"><em>Figure 7: ROC Curve for Initial Best Model.  </em></p>

    *   **Feature Importance:** Examined the coefficients of the Logistic Regression model to understand which features were most influential in predicting subscription.
         <img width="1422" height="791" alt="Feature-importance" src="https://github.com/user-attachments/assets/03b354ee-307e-4801-82d9-f7c9180c92a0" />
         <p align="center">
              <em>Figure 8: Bar Plot of Top 10 Feature Coefficients.  </em>
         </p>

    *   **Learning Curve:** Plotted the learning curve to diagnose potential overfitting or underfitting (ours indicated some overfitting).
    
          <img width="862" height="543" alt="Learning curve" src="https://github.com/user-attachments/assets/5b301fa0-3085-4702-b1df-4e7c1acca1a3" />
          <p align="center"><em>Figure 9: Learning Curve Plot for Logistic Regression.  </em></p>
         

### 4. Model Improvement Attempts

Given the initial results, especially the lower precision for the minority class, we tried several strategies to improve the model:

*   **Feature Selection (RFE):** Used Recursive Feature Elimination to select a subset of features. This actually led to a slight decrease in AUC and precision, suggesting the original feature set (or a different selection) was better.
*   **Optimizing Classification Threshold:** This was a game-changer! By analyzing the precision-recall curve, we found an optimal threshold (around 0.58) that significantly boosted the precision for the 'yes' class from **0.41 to 0.45** and the F1-score from **0.51 to 0.52** compared to the default 0.5 threshold.
*   **ADASYN Resampling:** Explored ADASYN as an alternative to SMOTE for handling imbalance. While it slightly improved recall, it decreased precision and AUC compared to our SMOTE + threshold optimized model.
*   **Ensemble Methods (Bagging):** Implemented `BaggingClassifier` with Logistic Regression as the base estimator. This approach yielded performance metrics very similar to our initial model but was outperformed by the threshold-optimized model in terms of 'yes' class precision and F1-score.

## Key Findings

Our analysis clearly showed that **optimizing the classification threshold was the most effective strategy** for improving our Logistic Regression model for this problem. It allowed us to significantly increase the precision for the 'yes' class from **0.41 to 0.45** and the F1-score from **0.51 to 0.52** for predicting 'yes' subscriptions, which is often crucial for business applications (e.g., reducing wasted marketing efforts).

While other techniques like RFE, ADASYN, and Bagging were explored, they either didn't offer a substantial improvement or slightly degraded performance compared to our best approach (SMOTE-resampled data with an optimized classification threshold).
    <p align="center">
    <img width="728" height="220" alt="Comparison-Table" src="https://github.com/user-attachments/assets/187d72e7-f4b7-41dc-9a76-be43f10664b0" />
    </p>
    <p align="center">
    <em>Figure 10: Performance Comparison Table of all models.  </em>
    </p>
 

## Conclusion

The final model, a Logistic Regression model trained on SMOTE-resampled data and employing an optimized classification threshold (0.58), provides a good balance. It demonstrates a strong ability to identify potential subscribers while significantly improving the reliability of those 'yes' predictions (higher precision) without sacrificing too much recall. This optimized model is the best performer among the tested approaches for this problem.

<p align="center">
  <img width="536" height="467" alt="Confusion_matrix(optimal threshold)" src="https://github.com/user-attachments/assets/d231dd75-b1d2-460d-a527-b114aa78c0c9" />
</p>   
 <p align="center"><em>Figure 11: Confusion Matrix for the Optimal Threshold Model (our best model).  </em></p>

<p align="center">
<img width="635" height="467" alt="ROC(optimal threshold)" src="https://github.com/user-attachments/assets/147cd2cc-cd69-4964-9d66-cfec86b5e83b" />
</p>
<p align="center">
  <em>Figure 12: ROC Curve for the Optimal Threshold Model (our best model).  </em>
</p>
 
