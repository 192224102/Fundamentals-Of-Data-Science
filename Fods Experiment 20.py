import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
    'Age': [25, 28, 23, 30, 22, 26, 27, 29, 24, 31],
    'Position': ['Forward', 'Midfielder', 'Forward', 'Defender', 'Midfielder', 'Forward', 'Midfielder', 'Defender', 'Forward', 'Goalkeeper'],
    'GoalsScored': [15, 10, 12, 5, 8, 18, 9, 4, 14, 3],
    'WeeklySalary': [5000, 6000, 5500, 7000, 4500, 8000, 7500, 9000, 5000, 10000]
}
df = pd.DataFrame(data)
df.to_csv('soccer_players.csv', index=False)
df = pd.read_csv('soccer_players.csv')
top_goals_players = df.nlargest(5, 'GoalsScored')
top_salary_players = df.nlargest(5, 'WeeklySalary')
average_age = df['Age'].mean()
above_average_age_players = df[df['Age'] > average_age]['Name']
position_distribution = df['Position'].value_counts()
position_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Players Based on Positions')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.show()
print("Top 5 Players with the Highest Number of Goals Scored:")
print(top_goals_players[['Name', 'GoalsScored']])

print("\nTop 5 Players with the Highest Salaries:")
print(top_salary_players[['Name', 'WeeklySalary']])

print(f"\nAverage Age of Players: {average_age:.2f}")

print("\nNames of Players Above the Average Age:")
print(above_average_age_players)
