import numpy as np

a = [
	[1, 2],
	[3, 4],
	[5, 6]
]

b = [
	[7, 8],
	[9, 0]
]

def matmul(a, b):
	dim_a = (len(a), len(a[0]))
	dim_b = (len(b), len(b[0]))

	result = []
	for row_a in range(dim_a[0]):
		result.append([])
		for col_b in range(dim_b[1]):
			result[row_a] += [
				sum([a[
					row_a][index] * b[index][col_b] for index in range(dim_a[1])
				])
			]

	return result

print("NumPy ", np.matmul(np.array(a), np.array(b)))
print("Our: ", matmul(a, b))

import time

results = []
for size in range(1, 101):
	m = np.random.rand(size, size)
	a = m.tolist()

	start = time.time()
	np.matmul(m, m)
	duration_np = time.time() - start

	start = time.time()
	matmul(a, a)
	duration_our = time.time() - start

	results.append((size, duration_np, duration_our))
	print(f"Size: {size}\tNumPy: {duration_np}\tOur: {duration_our}")

import matplotlib.pyplot as plt

sizes = [r[0] for r in results]
np_durations = [r[1] for r in results]
our_durations = [r[2] for r in results]

plt.plot(sizes, np_durations, label="NumPy")
plt.plot(sizes, our_durations, label="Our")
plt.show()
