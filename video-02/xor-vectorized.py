import numpy as np

layer1 = np.array([
	[1.5, 	-0.5],
	[-1, 	   1],
	[-1, 	   1]
])

layer2 = np.array([
	[-1],
	[1],
	[1]
])

def step(x):
	return np.where(x>0, 1, 0)

def neural_net(inputs, layers, activation_function):
	outputs = inputs
	for layer in layers:
		inputs = np.hstack([np.ones(shape=(outputs.shape[0], 1)), outputs])
		outputs = activation_function(np.matmul(inputs, layer))

	return outputs

inputs = [
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1]
]

for i in inputs:
	print(
		i,
		" -> ",
		neural_net(
			inputs=np.array([i]),
			layers=[layer1, layer2],
			activation_function=step
		)
	)

print("\nAll the inputs at once:")
print(
	neural_net(
		inputs=np.array(inputs),
		layers=[layer1, layer2],
		activation_function=step
	)
)
