#  STUDENT PERFORMANCE PREDICTION SYSTEM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sleep_Hours':   [8, 7, 7, 6, 6, 5, 5, 5, 4, 4],
    'Attendance_%':  [60, 65, 70, 75, 80, 82, 85, 90, 95, 97],
    'Marks_Scored':  [25, 35, 45, 50, 60, 65, 70, 78, 85, 90]
}

df = pd.DataFrame(data)
print(" Student Performance Data:\n", df)


print("\nSummary Statistics:\n", df.describe())
print("\nCorrelation Matrix:\n", df.corr())

plt.figure(figsize=(6,4))
plt.scatter(df['Hours_Studied'], df['Marks_Scored'], color='blue', label='Hours vs Marks')
plt.scatter(df['Sleep_Hours'], df['Marks_Scored'], color='green', label='Sleep vs Marks')
plt.xlabel("Hours / Sleep")
plt.ylabel("Marks Scored")
plt.title("Student Performance Relation")
plt.legend()
plt.show()

X = df[['Hours_Studied', 'Sleep_Hours', 'Attendance_%']]
y = df['Marks_Scored']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

new_student = np.array([[7, 6, 90]])
predicted_marks = model.predict(new_student)
print("\n Predicted Marks for new student:", round(predicted_marks[0], 2))


plt.figure(figsize=(6,4))
plt.scatter(df['Hours_Studied'], df['Marks_Scored'], color='blue')
plt.plot(df['Hours_Studied'], model.predict(df[['Hours_Studied','Sleep_Hours','Attendance_%']]), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Prediction Line")
plt.show()


print("\n Project Complete: Student Performance Prediction System")
