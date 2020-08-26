### The current folder contains the following content:
1. code and written report for: pneumonia detection using convolutional neural network with cyclical learning rate
2. code and written report for: evaluation of network robustness by applying Universal Adversarial Attack and Fast Gradient Sign Method with Gaussian noises
3. pictures included in the report

The first technical report contains the following sections. First, it provides a description of the current dataset, along with the data transformation techniques applied to the input. It then outlines the current neural network architecture and the rationale for the associated hyperparameter choices. Following this section, the optimisation method is then discussed in the context of the algorithm, learning rate choice, and the stopping criteria. Finally, we present the results obtained from our current model and discuss its potential implications.

The goal of adversarial attacks is to "break" or "trick" classifiers by introducing some perturbation to the input. Ideally, this perturbation should be barely perceptible by a person, keeping the original image mostly intact, yet able to deceive a model into classifying it incorrectly. These attacks often make use of the discontinuity involved in deep neural networks, allowing perturbations to "push" data points across decision boundaries.

The attacks tackled here are examples of white box attacks, adversarial attacks that require one to have access to the model itself as it is involved in the computation of the perturbations. Two particular attacks are implemented and expounded upon in this notebook, a particular example of a Universal Adversarial Perturbation (UAP) and the Fast Gradient Sign Method (FGSM)

