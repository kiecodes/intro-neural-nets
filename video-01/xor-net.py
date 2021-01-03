from typing import Callable


Vector = [float]
ActivationFunc = Callable[[float], float]

def neuron(inputs: Vector, weights: Vector, activation_func: ActivationFunc) -> float:
	return activation_func(
		sum(
			z[0] * z[1] for z in zip([1.0] + inputs, weights)
		)
	)

def step(x: float) -> float:
	return 1 if x > 0 else 0


NAND = [1.5, -1, -1]
OR = [-0.5, 1, 1]
AND = [-1, 1, 1]

def xor_net(inputs: Vector) -> float:
	return neuron(
		inputs=[
			neuron(inputs, NAND, step),
			neuron(inputs, OR, step)
		],
		weights=AND,
		activation_func=step
	)

inputs = [
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1]
]

for i in inputs:
	print(i, xor_net(i))
