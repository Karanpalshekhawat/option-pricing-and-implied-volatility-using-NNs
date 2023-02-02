
## Neural Network model for Pricing Options and computing Implied Volatility

---
An `approach` to learn non-linear relationship between option parameters, it's price and implied volatility using Deep Neural Network

### Main Features

---

- `Black-Scholes` pricing NN model
- `Heston pricing` NN model
- `Implied Volatility` NN model

Apart from above 3 main features, repository also contains python scripts to compute heston price using numerical method under simulation combining both underlying price and volatility stochastic process. It also brent method to compute implied volatility for a given option price and it's parameters.
### Common features across all models

--- 

- Dataset that is used to train each model was created using Latin Hypercube Sampling technique for a range of input parameters. Details about using this range of input parameter is available under `model/static_data` directory, module is present under `model/pricing/utils`. 
- Same tuned hyperparameter was used across all 3 models. Approach used to identify hyper parameters was Random Search using cross validation. Module is available under `model/pricing/core` 

### Documentation

---

Implemented `Pricing options and computing implied volatilities
using neural networks` research paper published by Shuaiqiang Liu 1, Cornelis W. Oosterlee and Sander M.Bohte. Document is present under the `literature` directory.


### Notebooks 

---

Added some jupyter notebooks as below to better understand the flow of implementation