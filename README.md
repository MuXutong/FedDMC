# FedDMC: Efficient and Robust Federated Learning via Detecting Malicious Clients

> Xutong Mu, Ke Cheng, Yulong Shen, Xiaoxiao Li, Zhao Chang, Tao Zhang, and Xindi Ma,  
> *TDSC, 2024*

## Abstract

Federated learning (FL) has gained popularity in the field of machine learning, which allows multiple participants to collaboratively learn a highly-accurate global model without exposing their sensitive data. However, FL is susceptible to poisoning attacks, in which malicious clients manipulate local model parameters to corrupt the global model. Existing FL frameworks based on detecting malicious clients suffer from unreasonable assumptions (e.g., clean validation datasets) or fail to balance robustness and efficiency. To address these deficiencies, we propose FedDMC, which implements robust federated learning by efficiently and precisely detecting malicious clients. Specifically, FedDMC first applies principal component analysis to reduce the dimensionality of the model parameters, which retains the primary parameter feature and reduces the computational overhead for subsequent clustering. Then, a binary tree-based clustering method with noise is designed to eliminate the effect of noisy points in the clustering process, facilitating accurate and efficient malicious client detection. Finally, we design a self-ensemble detection correction module that utilizes historical results via exponential moving averages to improve the robustness of malicious client detection. Extensive experiments conducted on three benchmark datasets demonstrate that FedDMC outperforms state-of-the-art methods in terms of detection precision, global model accuracy, and computational complexity.
[[paper]](https://ieeexplore.ieee.org/abstract/document/10458320).

## Citation

Please cite our paper if you find this code useful for your research.

```
@article{mu2024feddmc,
  title={FedDMC: Efficient and Robust Federated Learning via Detecting Malicious Clients},
  author={Mu, Xutong and Cheng, Ke and Shen, Yulong and Li, Xiaoxiao and Chang, Zhao and Zhang, Tao and Ma, Xindi},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024},
  publisher={IEEE}
}
```
