import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
num_students = 50
study_time = np.random.uniform(1, 10, size=num_students)
exam_scores = 50 + 10 * study_time + np.random.randn(num_students) * 5
data = pd.DataFrame({'StudyTime': study_time, 'ExamScore': exam_scores})
correlation = data['StudyTime'].corr(data['ExamScore'])
# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['StudyTime'], data['ExamScore'])
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.title('Scatter Plot of Study Time vs Exam Score')
plt.show()
# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
print(f"Correlation coefficient: {correlation:.2f}")
