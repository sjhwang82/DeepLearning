# Deep Learning @ KAIST

## Course Information

**Description**

This course covers various models and algorithms for deep learning, including both the fundemantal concepts and recent advances. In the first half of the course, we will learn about the most basic topics such as multi-layer preceptrons, backpropagation, convolutional neural networks, recurrent neural networks, autoencoders and variational autoencoders, optimization and regularization of deep learning models. Then, in the second half of the course, we will go over more advanced models, such as generative adversarial networks, efficient CNN architectures, transformers, and graph neural networks, while solving a real-world problem utilizing one of them.  

**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  

**TAs:** Seanie Lee, Taewook Nam, Jinheon Baek, Minki Kang

**Office:** 
Building #9, 9201, KAIST Seoul Campus. 
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading** - You will be graded by the total absolute score, and not the relative ranking.
* Mid-term Exam: 40% - The exam will cover the basic topics of deep learning taught in the first half of the course.
* Final Project: 40% - You will need to work on a final project utilizing one of the advanced models, present it, and submit the report.
* Attendance and Participation: 20% - Active participation during or off-class hours will be highly appreciated.

## Tentative Schedule

| Dates | Topic | 
|---|:---|
|3/2| Course Introduction |
|3/4| Review of Machine Learning Basics **(Video Lecture)** |
|3/9| Feedforward Neural Networks (Cost Function, Activations)
|3/11| Feedforward Neural Networks (Backpropagation)
|3/16| **Pytorch Basics** 
|3/18| Convolutional Neural Networks
|3/23| Advanced CNN Architectures (GoogLeNet, ResNet, DenseNet)
|3/25| Recurrent Neural Networks (LSTM, Seq2Seq)
|3/30| Attentional RNNs and Memory Networks
|4/1| CNN & RNN **(Programming)**
|4/6| Regularization Techniques for Deep Learning (L2/L1, Ensemble)  
|4/8| Regularization Techniques for Deep Learning (Dropout, Data augmentation)
|4/13| Optimization Techniques for Deep Learning (Challenges, Lottery Ticket Hypothesis) 
|4/15| Optimization Techniques for Deep Learning (Adaptive SGD and Second-order Methods) 
|4/20| **Mid-term Exam**
|4/27| Regularization & Optimization **(Programming)**
|4/29| Autoencoders and Variational Autoencoders
|5/4| Generative Adversarial Networks
|5/6| Advanced GANs (WGAN, StyleGAN)
|5/11| VAE and GAN **(Programming)**
|5/13| Efficient CNN Architectures (MobileNets, ShuffltNets) 
|5/18| Efficient CNN Architectures (EfficientNet, NFNets)
|5/20| Transformers
|5/25| Pretrained Language Models (BERT, GPT)
|6/1| Transformer **(Programming)**
|6/3| Vision and Multi-modal Transformers (VIT, DALLE)
|6/8| Graph Neural Networks (Node Embeddings)
|6/10| Graph Neural Networks (Pooling Methods)
|6/15| **Final Presentation**

## Reading List
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

