import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

file_path = "fraudTest.csv"
data = pd.read_csv(file_path)

missing_values = data.isnull().sum()

data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])

data["merchant"] = data["merchant"].astype('category')
data["category"] = data["category"].astype('category')
data["gender"] = data["gender"].astype('category')

data["amt"] = data["amt"].astype('float')

data["log_amt"] = np.log1p(data["amt"])
data["transaction_hour"] = data["trans_date_trans_time"].dt.hour
data["transaction_day"] = data["trans_date_trans_time"].dt.dayofweek
data["transaction_count_last_month"] = data.groupby("cc_num")['trans_num'].transform(lambda x: x.count())
data["merchant_transaction_count"] = data.groupby("merchant")['trans_num'].transform(lambda x: x.count())

data = data.drop(columns=['cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num'])


min_max_scaler = MinMaxScaler()
data[['amt_minmax', 'log_amt_minmax']] = min_max_scaler.fit_transform(data[['amt', 'log_amt']])

standard_scaler = StandardScaler()
data[['amt_standard', 'log_amt_standard']] = standard_scaler.fit_transform(data[['amt', 'log_amt']])



x = data.drop(columns=['is_fraud', 'trans_date_trans_time', 'merchant', 'category', 'gender', 'job'])
y = data['is_fraud']

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

data_resampled = pd.DataFrame(x_resampled, columns=x.columns)
data_resampled['is_fraud'] = y_resampled

data_majority = data[data['is_fraud'] == 0]
data_minority = data[data['is_fraud'] == 1]

data_majority_downsampled = resample(data_majority, replace=False, n_samples=len(data_minority), random_state=42)

data_balanced = pd.concat([data_majority_downsampled, data_minority])

class_distribution = data_balanced['is_fraud'].value_counts()
print(class_distribution)

# class_distribution = data['is_fraud'].value_counts()
# print(class_distribution)

# class_distribution.plot(kind='bar')
# plt.title('Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('No. of Transactions')
# plt.xticks(ticks=[0,1], labels=['Not Fraud', 'Fraud'], rotation=0)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cross_val_scores = cross_val_score(model, x_resampled, y_resampled, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cross_val_scores)
print("Mean Accuracy:", cross_val_scores.mean())


# Feature importance for Random Forest
feature_importances = model.feature_importances_
features = X_train.columns

# Create a DataFrame for better visualization
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot the top features
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'][:10], feature_df['Importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features')
plt.show()

# print(data[['amt', 'amt_minmax', 'amt_standard', 'log_amt', 'log_amt_minmax', 'log_amt_standard']].head())
# print(data[['amt_minmax', 'amt_standard']].describe())
# print(data.dtypes)
# print(data.head())

