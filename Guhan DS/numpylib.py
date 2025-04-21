import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 2))
identity = np.eye(3)
random_arr = np.random.rand(3, 3)
print(arr1)
print(arr2)
print(zeros)
print(ones)
print(identity)
print(random_arr)

# Array properties
print("Shape:", arr2.shape)
print("Size:", arr2.size)
print("Datatype:", arr2.dtype)
print("Dimensions:", arr2.ndim)

# Indexing & Slicing
print("First row:", arr2[0])
print("Element at (1,2):", arr2[1, 2])
print("Sliced array:", arr1[1:4])

# Reshaping & Transposing
reshaped = arr1.reshape(5, 1)
transposed = arr2.T
print("Reshaped:", reshaped)
print("Transposed:", transposed)

# Mathematical Operations
arr3 = np.array([10, 20, 30, 40, 50])
sum_arr = arr1 + arr3
prod_arr = arr1 * arr3
div_arr = arr3 / arr1
sqrt_arr = np.sqrt(arr3)
power_arr = np.power(arr1, 2)
print("Sum:", sum_arr)
print("Product:", prod_arr)
print("Division:", div_arr)
print("Square Root:", sqrt_arr)
print("Power:", power_arr)
""" 
# Statistical Functions
mean_val = np.mean(arr3)
median_val = np.median(arr3)
std_dev = np.std(arr3)
var_val = np.var(arr3)
min_val = np.min(arr3)
max_val = np.max(arr3)
print("Mean:", mean_val)
print("Median:", median_val)
print("Standard Deviation:", std_dev)
print("Variance:", var_val)
print("Min:", min_val)
print("Max:", max_val)

# Concatenation & Stacking
concat_arr = np.concatenate((arr1, arr3))
stacked_arr = np.vstack((arr1, arr3))
print("Concatenated:", concat_arr)
print("Stacked:", stacked_arr)

# Linear Algebra
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(matrix_a, matrix_b)
inverse_matrix = np.linalg.inv(matrix_a)
determinant = np.linalg.det(matrix_a)
print("Matrix Product:", matrix_product)
print("Inverse:", inverse_matrix)
print("Determinant:", determinant)

# Random Number Generation
rand_ints = np.random.randint(1, 100, (3, 3))
normals = np.random.normal(0, 1, (3, 3))
print("Random Integers:", rand_ints)
print("Normal Distribution:", normals)


 """