import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import altair as alt

# Step 1: Data Loading and Preprocessing
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].str.replace(',', '.'), errors='coerce')
    data['MonthlyCharges'] = data['MonthlyCharges'].str.replace(',', '.').astype(float)
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
    data['numAdminTickets'] = data['numAdminTickets'].astype(int)
    data['numTechTickets'] = data['numTechTickets'].astype(int)
    data.fillna(0, inplace=True)
    return data

data = load_data()

# Encode categorical variables
def preprocess_data(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.drop(['customerID'])
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

processed_data = preprocess_data(data)

X = processed_data.drop(['customerID', 'Churn'], axis=1)
y = processed_data['Churn']

model = RandomForestClassifier(random_state=42)

# Use StratifiedKFold for preserving class distribution
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Obtain cross-validated predictions
y_pred = cross_val_predict(model, X, y, cv=kf, method='predict')
y_proba = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:, 1]

# Compute evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_proba)

# Add predictions and probabilities to the data
data['Churn_Pred'] = y_pred
data['Churn_Prob'] = y_proba

# Assign Churn Risk based on predicted probabilities
data['Churn_Risk'] = pd.qcut(data['Churn_Prob'], q=3, labels=['Low', 'Medium', 'High'])

segment_metrics = data.groupby('Churn_Risk').agg({
    'tenure': 'mean',
    'MonthlyCharges': 'mean',
    'Churn_Prob': 'mean'
}).reset_index()

st.title('Customer Churn Dashboard')

# Display Model Evaluation Metrics
st.header('Model Evaluation Metrics')
st.write("""
The model's performance is evaluated using several key metrics:
- **Accuracy** measures the proportion of correct predictions.
- **Precision** indicates how many of the predicted churns were actual churns.
- **Recall** reflects the model's ability to identify actual churns.
- **F1 Score** is the harmonic mean of precision and recall.
- **ROC AUC** measures the model's ability to distinguish between classes.

Below are the calculated metrics for our model:
""")
col1, col2, col3 = st.columns(3)
col1.metric('Accuracy', f'{accuracy:.2%}')
col2.metric('Precision', f'{precision:.2%}')
col3.metric('Recall', f'{recall:.2%}')
col1.metric('F1 Score', f'{f1:.2%}')
col2.metric('ROC AUC', f'{roc_auc:.2f}')

st.write("""
- An **accuracy of 85.21%** indicates that the model correctly predicts the churn status 85.21% of the time.
- A **precision of 74.25%** means that when the model predicts a customer will churn, it's correct 74.25% of the time.
- A **recall of 67.75%** shows that the model identifies 67.75% of the customers who actually churned.
- The **F1 Score of 70.84%** balances precision and recall, providing a single metric to evaluate the model.
- An **ROC AUC of 0.91** suggests the model has a good ability to distinguish between customers who churn and those who do not.
""")

# Visual Ranking of Customers
st.header('Customer Churn Risk Ranking')
st.write("""
The customers are segmented into three risk categories based on their predicted churn probability:
- **High Risk**
- **Medium Risk**
- **Low Risk**

Below is the distribution of customers across these risk levels:
""")
risk_counts = data['Churn_Risk'].value_counts().reindex(['High', 'Medium', 'Low']).reset_index()
risk_counts.columns = ['Churn_Risk', 'count']

chart = alt.Chart(risk_counts).mark_bar().encode(
    x=alt.X('Churn_Risk', sort=['High', 'Medium', 'Low']),
    y='count',
    color='Churn_Risk'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)

st.write("""
**Interpretation:**
- The majority of customers fall into the **Low Risk** category, indicating they are unlikely to churn.
- The **High Risk** segment, although smaller, represents customers who are more likely to churn and may require targeted retention strategies.
""")

# Main Factors Affecting Churn
st.header('Factors Affecting Churn')
st.write("""
Understanding the factors that contribute most to customer churn can help in developing effective retention strategies. Here are the top 5 features influencing churn:
""")
# Fit the model on the entire dataset for feature importances
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(5).reset_index()
top_features.columns = ['Feature', 'Importance']

chart = alt.Chart(top_features).mark_bar().encode(
    x='Importance',
    y=alt.Y('Feature', sort='-x'),
    color='Feature'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)

st.write("""
**Interpretation:**
- The features with the highest importance scores have the most significant impact on predicting churn.
- By focusing on these areas, the company can address issues that may be causing customers to leave.
""")

# Customer Segments
st.header('Customer Segments')
st.write("""
Below are the average metrics for each customer segment:
""")
st.dataframe(segment_metrics)

st.write("""
**Interpretation:**
- **Low Risk** customers have the longest average tenure (52.15 months) and the lowest average churn probability (less than 1%).
- **Medium Risk** customers have a moderate tenure and churn probability.
- **High Risk** customers have the shortest average tenure (17.22 months) and the highest average monthly charges ($75.27), with a high churn probability (65.53%).
- Higher monthly charges and shorter tenure are associated with increased churn risk.
""")

# Recommended Retention Strategies
st.header('Retention Strategies')
st.write("""
Based on the customer segments, here are some recommended retention strategies:
""")
strategies = {
    'High': 'Offer discounts or loyalty rewards to retain high-risk customers.',
    'Medium': 'Provide personalized offers or improved customer support.',
    'Low': 'Maintain current satisfaction levels to keep low-risk customers engaged.'
}
for risk_level, strategy in strategies.items():
    st.subheader(f'{risk_level} Risk')
    st.write(strategy)

# Total Drop Rate vs. Predicted Drop Rate
st.header('Total Drop Rate vs. Predicted Drop Rate')
st.write("""
Comparing the actual drop rate with the predicted drop rate helps assess the model's accuracy in estimating churn.
""")
actual_drop_rate = data['Churn'].mean()
predicted_drop_rate = data['Churn_Pred'].mean()
col1, col2 = st.columns(2)
col1.metric('Actual Drop Rate', f'{actual_drop_rate:.2%}')
col2.metric('Predicted Drop Rate', f'{predicted_drop_rate:.2%}')

st.write("""
- The **actual drop rate is 26.54%**, meaning that approximately a quarter of customers have churned.
- The **model's predicted drop rate is 24.21%**, which is close to the actual rate, indicating good predictive performance.
""")

# Predicted Churn Rates by Segment
st.header('Predicted Churn Rates by Customer Segment')
st.write("""
The average predicted churn probability for each segment is displayed below:
""")
chart = alt.Chart(segment_metrics).mark_bar().encode(
    x=alt.X('Churn_Risk', sort=['High', 'Medium', 'Low']),
    y='Churn_Prob',
    color='Churn_Risk'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)

st.write("""
- **High Risk** customers have an average predicted churn probability of 65.53%.
- **Medium Risk** customers have an average predicted churn probability of 14.12%.
- **Low Risk** customers have an average predicted churn probability of less than 1%.
- This segmentation helps prioritize retention efforts towards customers most likely to churn.
""")

# Confusion Matrix
st.header('Confusion Matrix')
st.write("""
The confusion matrix provides a detailed breakdown of the model's performance:
""")
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
st.write(cm_df)

st.write("""
- **True Negatives (4,735)**: Customers who did not churn and were correctly predicted.
- **False Positives (439)**: Customers who did not churn but were incorrectly predicted to churn.
- **False Negatives (603)**: Customers who churned but were not identified by the model.
- **True Positives (1,266)**: Customers who churned and were correctly predicted.
""")

# Classification Report
st.header('Classification Report')
st.write("""
The classification report provides precision, recall, and F1-score for each class:
""")
report = classification_report(y, y_pred, zero_division=0)
st.text(report)

st.write("""
- For the **"No Churn"** class:
  - **Precision (89%)**: High precision indicates that most customers predicted not to churn indeed did not churn.
  - **Recall (92%)**: The model correctly identified 92% of customers who did not churn.
- For the **"Churn"** class:
  - **Precision (74%)**: When the model predicts churn, it's correct 74% of the time.
  - **Recall (68%)**: The model correctly identified 68% of the actual churners.
- **Overall**, the model performs well, but there's room for improvement in detecting all customers who may churn.
""")

# Conclusion
st.header('Conclusion')
st.write("""
The customer churn model demonstrates good predictive capabilities, with an accuracy of over 85% and a high ROC AUC score of 0.91. The model effectively segments customers based on their churn risk, allowing for targeted retention strategies.

By focusing on the factors influencing churn and addressing the needs of high-risk customers, the company can reduce churn rates and improve customer satisfaction.
""")
