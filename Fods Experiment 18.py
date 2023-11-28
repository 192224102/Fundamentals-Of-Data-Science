import numpy as np

def calculate_mean(temperatures):
    return np.mean(temperatures)

def calculate_std_dev(temperatures):
    return np.std(temperatures)

def calculate_temperature_range(temperatures):
    return np.max(temperatures) - np.min(temperatures)

def find_most_consistent_city(std_deviations):
    return np.argmin(std_deviations)

def analyze_temperature_data(city_temperatures):
    city_stats = []

    for city, temperatures in city_temperatures.items():
        mean_temp = calculate_mean(temperatures)
        std_dev = calculate_std_dev(temperatures)
        temp_range = calculate_temperature_range(temperatures)
        city_stats.append({
            'city': city,
            'mean_temperature': mean_temp,
            'std_deviation': std_dev,
            'temperature_range': temp_range
        })
    print("Mean Temperatures:")
    for stat in city_stats:
        print(f"{stat['city']}: {stat['mean_temperature']}")
    print("\nStandard Deviations:")
    for stat in city_stats:
        print(f"{stat['city']}: {stat['std_deviation']}")
    max_range_city = max(city_stats, key=lambda x: x['temperature_range'])
    print(f"\nCity with highest temperature range: {max_range_city['city']} "
          f"({max_range_city['temperature_range']} degrees)")
    most_consistent_city_index = find_most_consistent_city([stat['std_deviation'] for stat in city_stats])
    most_consistent_city = city_stats[most_consistent_city_index]['city']
    print(f"\nCity with most consistent temperature: {most_consistent_city} "
          f"({city_stats[most_consistent_city_index]['std_deviation']} standard deviation)")

city_temperatures = {
    'City1': [25, 26, 24, 23, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'City2': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    'City3': [22, 22, 23, 24, 23, 22, 22, 23, 24, 25, 26, 27, 26, 25, 24, 23, 22, 21, 20, 21, 22, 23, 24, 25, 26, 25, 24, 23, 22, 21, 20],
}

analyze_temperature_data(city_temperatures)
