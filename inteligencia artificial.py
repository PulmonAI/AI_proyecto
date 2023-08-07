import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

url = "http://dibresources.jcbose.ac.in/ssaha4/pulmopred/public/training.csv.txt"
ds = pd.read_csv(url)

X = ds.drop('target', axis=1)
y = ds['target']

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 5: Evaluate the Decision Tree
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Make Predictions
new_data = pd.DataFrame(...)  # New data to be predicted
predictions = clf.predict(new_data)
print("Predictions:", predictions)






