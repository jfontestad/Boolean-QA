import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(6, 5))

# set height of bar
models = ['RoBERTa-base', 'BERT-base-cased', 'BERT-base-uncased']
DEV = [0.6217125382262997, 0.7033639143730887, 0.708868501529052]
TEST = [0.6278906797477225, 0.6995585143658024, 0.7217939733707078]

# Set position of bar on X axis
br1 = np.arange(len(DEV))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, DEV, color='blue', width=barWidth, edgecolor='grey', label='DEV')
plt.bar(br2, TEST, color='orange', width=barWidth, edgecolor='grey', label='TEST')

# Adding Xticks
plt.xlabel('Model', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks([r + barWidth for r in range(len(DEV))], models)

plt.legend()
plt.show()
