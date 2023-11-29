import numpy as np
student_scores = np.array([
    [80, 75, 90, 85],
    [92, 88, 78, 95],
    [85, 90, 92, 88],
    [78, 82, 80, 75]
])
average_scores = np.mean(student_scores, axis=0)
highest_avg_subject_index = np.argmax(average_scores)
print("Average Scores for Each Subject:", average_scores)
print("Subject with the Highest Average Score:", highest_avg_subject_index)
