import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(6, 5))

# set height of bar
DEV = [0.83, 0.88]
TEST = [0.8278906797477225, 0.8995585143658024]

# Set position of bar on X axis
br1 = np.arange(len(DEV))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, DEV, color='blue', width=barWidth, edgecolor='grey', label='DEV')
plt.bar(br2, TEST, color='orange', width=barWidth, edgecolor='grey', label='TEST')

# Adding Xticks
plt.xlabel('Model', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks([r + barWidth for r in range(len(DEV))], ['T5-small', 'T5-base'])

plt.legend()
plt.show()
