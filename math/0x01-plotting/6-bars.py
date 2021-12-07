#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ['Farrah', 'Fred', 'Felicia']
legend = [
    ('apples', 'red'),
    ('bananas', 'yellow'),
    ('oranges', '#ff8000'),
    ('peaches', '#ffe5b4'),
]

bottom = fruit[0] * 0
for i, row in enumerate(fruit):
    label, color = legend[i]
    plt.bar(labels, row, .5, label=label, bottom=bottom, color=color)
    bottom += row

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.axis([None, None, 0, 80])
plt.yticks(range(0, 81, 10))
plt.legend()

plt.show()
