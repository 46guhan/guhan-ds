import matplotlib.pyplot as plt
import numpy as np

# Line Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='Sine Wave', color='b', linestyle='-', linewidth=2)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Line Plot')
plt.legend()
plt.grid()
plt.show()

# Scatter Plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
plt.figure(figsize=(8, 5))
plt.scatter(x_scatter, y_scatter, color='r', marker='o', label='Data Points')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter Plot')
plt.legend()
plt.grid()
plt.show()

# Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]
plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Horizontal Bar Chart
plt.figure(figsize=(8, 5))
plt.barh(categories, values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Chart')
plt.show()


 # Histogram
data = np.random.randn(1000)
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='c', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

 # Pie Chart
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [40, 30, 20, 10]
plt.figure(figsize=(8, 5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['gold', 'blue', 'red', 'green'])
plt.title('Pie Chart')
plt.show()

 # Box Plot
data_box = [np.random.randn(100), np.random.randn(100), np.random.randn(100)]
plt.figure(figsize=(8, 5))
plt.boxplot(data_box, patch_artist=True, labels=['A', 'B', 'C'])
plt.title('Box Plot')
plt.show()

 # Violin Plot
plt.figure(figsize=(8, 5))
plt.violinplot(data_box)
plt.title('Violin Plot')
plt.show()

 # Stem Plot
plt.figure(figsize=(8, 5))
plt.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('Stem Plot')
plt.show()


 # Subplots for Multiple Graphs
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(x, y, 'b')
axes[0, 0].set_title('Line Plot')

axes[0, 1].scatter(x_scatter, y_scatter, color='r')
axes[0, 1].set_title('Scatter Plot')

axes[1, 0].bar(categories, values, color='g')
axes[1, 0].set_title('Bar Chart')

axes[1, 1].hist(data, bins=20, color='c')
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
