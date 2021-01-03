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


weights = [-0.25, 1, -0.45]

glenmorangie = [-0.21, 0.18]
talisker = [0.6, -0.31]

print("Glenmorangie:", neuron(glenmorangie, weights, step))
print("Talisker:", neuron(talisker, weights, step))