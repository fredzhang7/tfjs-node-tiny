# tfjs node tiny
A light-weight, 193MB version of `@tensorflow/tfjs-node` to perform inference on any TensorFlow model in the SavedModel format. 
This repository trims all built-in TensorFlow components used for model training, while still allowing for quicker model inference.

With ≈450 MB reduction in module size, ≈150%-200% the speed to load a model, and slightly faster model inference, this repository outperforms the `@tensorflow/tfjs-node` module in model inference.
