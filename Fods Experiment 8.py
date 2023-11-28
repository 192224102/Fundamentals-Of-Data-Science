import numpy as np
student_scores = np.array([
    [90, 85, 92, 88],
    [78, 92, 80, 85],
    [88, 87, 90, 89],
    [95, 80, 88, 91]
])
average_scores = np.mean(student_scores, axis=0)
subject_with_highest_avg = np.argmax(average_scores)
subjects = ['Math', 'Science', 'English', 'History']
highest_avg_subject = subjects[subject_with_highest_avg]

print("Average Scores for Each Subject:", average_scores)
print("Subject with Highest Average Score:", highest_avg_subject)
