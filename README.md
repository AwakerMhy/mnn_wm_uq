# Uncertainty Quantification in Working Memory via Moment Neural Networks
###  [Paper](https://arxiv.org/abs/2411.14196)
> [**Uncertainty Quantification in Working Memory via Moment Neural Networks**](https://arxiv.org/abs/2411.14196),            
> Hengyuan Ma, Wenlian Lu, and Jianfeng Feng   

## Abstract
Humans possess a finely tuned sense of uncertainty that helps anticipate potential errors, vital for adaptive behavior and survival. However, the underlying neural mechanisms remain unclear. This study applies moment neural networks (MNNs) to explore the neural mechanism of uncertainty quantification in working memory (WM). The MNN captures nonlinear coupling of the first two moments in spiking neural networks (SNNs), identifying firing covariance as a key indicator of uncertainty in encoded information. Trained on a WM task, the model demonstrates coding precision and uncertainty quantification comparable to human performance. Analysis reveals a link between the probabilistic and sampling-based coding for uncertainty representation. Transferring the MNN's weights to an SNN replicates these results. Furthermore, the study provides testable predictions demonstrating how noise and heterogeneity enhance WM performance, highlighting their beneficial role rather than being mere biological byproducts. These findings offer insights into how the brain effectively manages uncertainty with exceptional accuracy.


## Requirement
- Python 3.7
- numpy 1.21.6
- torch 1.11.0


## Running the Experiments

Experiments on the moment neurak network: `working_memory_mnn.py`.
```
$ python working_memory_mnn.py
```

Experiments on the spiking neural network: `snn_verification.py`.
```
$ python snn_verification.py
```

## License

[MIT](LICENSE)


## Acknowledgments

Some parts of this codebase are adapted or derived from [moment-neural-network](https://github.com/BrainsoupFactory/moment-neural-network) under the [MIT] license. Please check their repository for more details.


```