# Deep Learning @ KAIST

## Course Information

**Description**


**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  

**TAs:** Seanie Lee, Taewook Nam, Jinheon Baek, Minki Kang

**Office:** 
This is an online course.
Building #9, 9201, KAIST Seoul Campus. 
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading** - You will be graded by the total absolute score, and not the relative ranking.
* Mid-term Exam: 40% - The exam will cover the basic topics of deep learning taught in the first half of the course.
* Programming Assignments: 40% - You will need to work on programming assignments.
* Attendance and Participation: 20% - There will be a team presentation on one of the selected papers.

## Tentative Schedule

| Dates | Topic | 
|---|:---|
|3/2| Course Introduction |
|3/4| Review of Machine Learning Basics **(Video Lecture)** |
|3/9| Feedforward Neural Networks (Cost Function, Activations)
|3/11| Feedforward Neural Networks (Backpropagation)
|3/16| Convolutional Neural Networks
|3/18| Advanced CNN Architectures (GoogLeNet, ResNet, DenseNet)
|3/23| Recurrent Neural Networks (LSTM, Seq2Seq)
|3/25| Attentional RNNs and Memory Networks
|3/30| Regularization Techniques for Deep Learning (L2/L1, Ensemble) 
|4/1| Regularization Techniques for Deep Learning (Dropout, Data augmentation)
|4/6| Optimization Techniques for Deep Learning (Challenges) 
|4/8| Optimization Techniques for Deep Learning (Adaptive SGD and Second-order Methods) 
|4/13| Autoencoders
|4/15| Variational Autoencoders
|4/20| **Mid-term Exam**
|4/27| Generative Adversarial Networks
|4/29| Advanced GANs (WGAN, StyleGAN)
|5/4| Image Generation and Translation with GANs
|5/6| Efficient CNN Architectures (MobileNets, ShuffltNets) 
|5/11| Efficient CNN Architectures (EfficientNet, NFNets)
|5/13| Transformers
|5/18| Pretrained Language Models (BERT, GPT) 
|5/20| Vision and Multi-modal Transformers (VIT, DALLE)
|5/25| Graph Neural Networks (Node Embedding Methods)
|5/27| Graph Neural Networks (Pooling Methods)
|6/15| Final Project Due


## Reading List

### Bayesian Deep Learning
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114.pdf) Auto-Encoding Variational Bayes, ICLR 2014.   
[[Kingma et al. 15]](https://arxiv.org/pdf/1506.02557.pdf) Variational Dropout and the Local Reparameterization Trick, NIPS 2015.   
[[Blundell et al. 15]](https://arxiv.org/pdf/1505.05424.pdf) Weight Uncertainty in Neural Networks, ICML 2015.   
[[Gal and Ghahramani 16]](http://proceedings.mlr.press/v48/gal16.pdf) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016.   
[[Liu et al. 16]](https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm.pdf) Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm, NIPS 2016.  
[[Mandt et al. 17]](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf) Stochastic Gradient Descent as Approximate Bayesian Inference, JMLR 2017.  
[[Kendal and Gal 17]](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, ICML 2017.  
[[Gal et al. 17]](https://papers.nips.cc/paper/6949-concrete-dropout.pdf) Concrete Dropout, NIPS 2017.  
[[Gal et al. 17]](http://proceedings.mlr.press/v70/gal17a/gal17a.pdf) Deep Bayesian Active Learning with Image Data, ICML 2017.  
[[Teye et al. 18]](http://proceedings.mlr.press/v80/teye18a/teye18a.pdf) Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, ICML 2018.  
[[Garnelo et al. 18]](http://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf) Conditional Neural Process, ICML 2018.  
[[Kim et al. 19]](http://https://arxiv.org/pdf/1901.05761.pdf) Attentive Neural Processes, ICLR 2019.  
[[Sun et al. 19]](https://arxiv.org/pdf/1903.05779.pdf) Functional Variational Bayesian Neural Networks, ICLR 2019.  
***
[[Louizos et al. 19]](http://papers.nips.cc/paper/9079-the-functional-neural-process.pdf) The Functional Neural Process, NeurIPS 2019.  
[[Amersfoort et al. 20]](https://arxiv.org/pdf/2003.02037.pdf) Uncertainty Estimation Using a Single Deep Deterministic Neural Network, ICML 2020.  
[[Dusenberry et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5657-Paper.pdf) Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors, ICML 2020.  
[[Wenzel et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/3581-Paper.pdf) How Good is the Bayes Posterior in Deep Neural Networks Really?, ICML 2020.  
[[Lee et al. 20]](https://arxiv.org/abs/2008.02956) Bootstrapping Neural Processes, arXiv preprint 2020.  


### Deep Generative Models
#### VAEs, Autoregressive and Flow-Based Generative Models
[[Rezende and Mohamed 15]](http://proceedings.mlr.press/v37/rezende15.pdf) Variational Inference with Normalizing Flows, ICML 2015.   
[[Germain et al. 15]](http://proceedings.mlr.press/v37/germain15.pdf) MADE: Masked Autoencoder for Distribution Estimation, ICML 2015.  
[[Kingma et al. 16]](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf) Improved Variational Inference with Inverse Autoregressive Flow, NIPS 2016.  
[[Oord et al. 16]](http://proceedings.mlr.press/v48/oord16.pdf) Pixel Recurrent Neural Networks, ICML 2016.  
[[Dinh et al. 17]](https://openreview.net/pdf?id=HkpbnH9lx) Density Estimation Using Real NVP, ICLR 2017.  
[[Papamakarios et al. 17](https://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation.pdf) Masked Autoregressive Flow for Density Estimation, NIPS 2017.  
[[Huang et al.18]](http://proceedings.mlr.press/v80/huang18d/huang18d.pdf) Neural Autoregressive Flows, ICML 2018.  
[[Kingma and Dhariwal 18]](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf) Glow: Generative Flow with Invertible 1x1 Convolutions, NeurIPS 2018.  
[[Ho et al. 19]](http://proceedings.mlr.press/v97/ho19a/ho19a.pdf) Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design, ICML 2019.    
***
[[Chen et al. 19]](https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling.pdf) Residual Flows for Invertible Generative Modeling, NeurIPS 2019.  
[[Tran et al. 19]](https://papers.nips.cc/paper/9612-discrete-flows-invertible-generative-models-of-discrete-data.pdf) Discrete Flows: Invertible Generative
Models of Discrete Data, NeurIPS 2019.  
[[Ping et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/647-Paper.pdf) WaveFlow: A Compact Flow-based Model for Raw Audio, ICML 2020.  
[[Vahdat and Kautz 20]](https://arxiv.org/pdf/2007.03898v1.pdf) NVAE: A Deep Hierarchical Variational Autoencoder, arXiv preprint, 2020.  

#### Generative Adversarial Networks
[[Goodfellow et al. 14]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) Generative Adversarial Nets, NIPS 2014.   
[[Radford et al. 15]](https://arxiv.org/abs/1511.06434) Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.   
[[Chen et al. 16]](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016.   
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017.  
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.  
[[Zhang et al. 17]](https://arxiv.org/pdf/1706.03850.pdf) Adversarial Feature Matching for Text Generation, ICML 2017.  
[[Karras et al. 18]](https://openreview.net/forum?id=Hk99zCeAb) Progressive Growing of GANs for Improved Quality, Stability, and Variation, ICLR 2018.  
[[Brock et al. 19]](https://openreview.net/pdf?id=B1xsqj09Fm) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019.  
[[Karras et al. 19]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019.  
[[Xu et al. 19]](https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf) Modeling Tabular Data using Conditional GAN, NeurIPS 2019.   
***
[[Karras et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) Analyzing and Improving the Image Quality of StyleGAN, CVPR 2020.  
[[Zhao et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/4902-Paper.pdf) Feature Quantization Improves GAN Training, ICML 2020.  
[[Sinha et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/1324-Paper.pdf) Small-GAN: Speeding up GAN Training using Core-Sets, ICML 2020.  


### Deep Reinforcement Learning 
[[Mnih et al. 13]](https://arxiv.org/pdf/1312.5602.pdf) Playing Atari with Deep Reinforcement Learning, NIPS Deep Learning Workshop 2013.  
[[Silver et al. 14]](http://proceedings.mlr.press/v32/silver14.pdf) Deterministic Policy Gradient Algorithms, ICML 2014.      
[[Schulman et al. 15]](https://arxiv.org/pdf/1502.05477.pdf) Trust Region Policy Optimization, ICML 2015.  
[[Lillicrap et al. 16]](https://arxiv.org/pdf/1509.02971.pdf) Continuous Control with Deep Reinforcement Learning, ICLR 2016.    
[[Schaul et al. 16]](https://arxiv.org/pdf/1511.05952.pdf) Prioritized Experience Replay, ICLR 2016.  
[[Wang et al. 16]](http://proceedings.mlr.press/v48/wangf16.pdf) Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016.    
[[Mnih et al. 16]](http://proceedings.mlr.press/v48/mniha16.pdf) Asynchronous Methods for Deep Reinforcement Learning, ICML 2016.  
[[Schulman et al. 17]](https://arxiv.org/pdf/1707.06347.pdf) Proximal Policy Optimization Algorithms, arXiv preprint, 2017.  
[[Nachum et al. 18]](https://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf) Data-Efficient Hierarchical Reinforcement Learning, NeurIPS 2018.  
[[Ha et al. 18]](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf) Recurrent World Models Facilitate Policy Evolution, NeurIPS 2018.  
[[Burda et al. 19]](https://openreview.net/forum?id=rJNwDjAqYX) Large-Scale Study of Curiosity-Driven Learning, ICLR 2019.  
[[Vinyals et al. 19]](https://www.nature.com/articles/s41586-019-1724-z) Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature, 2019.  
[[Bellemare et al. 19]](https://papers.nips.cc/paper/8687-a-geometric-perspective-on-optimal-representations-for-reinforcement-learning.pdf) A Geometric Perspective on Optimal Representations for Reinforcement Learning, NeurIPS 2019.  
[[Janner et al. 19]](http://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization.pdf) When to Trust Your Model: Model-Based Policy Optimization, NeurIPS 2019.  
[[Fellows et al. 19]](https://papers.nips.cc/paper/8934-virel-a-variational-inference-framework-for-reinforcement-learning.pdf) VIREL: A Variational Inference Framework for Reinforcement Learning, NeurIPS 2019.  
***
[[Kaiser et al. 20]](https://openreview.net/pdf?id=S1xCPJHtDB) Model Based Reinforcement Learning for Atari, ICLR 2020.  
[[Agarwal et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5394-Paper.pdf) An Optimistic Perspective on Offline Reinforcement Learning, ICML 2020.  
[[Fedus et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/6110-Paper.pdf) Revisiting Fundamentals of Experience Replay, ICML 2020.  
[[Lee et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/3705-Paper.pdf) Batch Reinforcement Learning with Hyperparameter Gradients, ICML 2020.  
[[Raileanu et al. 20]](https://arxiv.org/pdf/2006.12862v1.pdf) Automatic Data Augmentation for Generalization in Deep Reinforcement Learning, arXiv preprint, 2020.  



