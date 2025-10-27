import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, "../cyber_physic_dataset", "data_version_2.csv")
data = pd.read_csv(dataset_path)

# Separate features and labels
X = data.drop("class", axis=1)
y = data["class"]

# Encode non-numeric columns if necessary
for column in X.select_dtypes(include=["object"]).columns:
    X[column] = LabelEncoder().fit_transform(X[column])

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit a RandomForest model to get feature importances
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances)
plt.title("Feature Importances")
plt.show()
