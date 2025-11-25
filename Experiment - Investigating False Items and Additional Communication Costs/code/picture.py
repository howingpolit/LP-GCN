import pandas as pd
import matplotlib.pyplot as plt

#
df = pd.read_csv("../experiment/gowalla-results-2024-07-22-20-29-20.csv")

#
plt.figure(figsize=(6, 6))

#
# plt.plot(df['rate of convolution client'], df['neighboring_users_number_sum'], label='Neighboring Users Number', marker='o')

#
# plt.plot(df['rate of convolution client'], df['average_neighboring_users_number'], label='Average Neighboring Users Number', marker='o')

#
plt.plot(df['rate of convolution client'], df['average_computation'], label='Average Computation', marker='o')

#
plt.legend()

#
plt.title('Relationship between Ratio of Convolution Clients and Various Metrics')
plt.xlabel('Ratio of Convolution Clients')
plt.ylabel('Metrics')

#
plt.show()
