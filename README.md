# Deep Learning @ KAIST

## Course Information

**Description**

This course covers various models and algorithms for deep learning, including both the fundemantal concepts and recent advances. In the first half of the course, we will learn about the most basic topics such as multi-layer preceptrons, backpropagation, convolutional neural networks, recurrent neural networks, Transformers, autoencoders and variational autoencoders, optimization and regularization of deep learning models. We will also have multiple lab sessions in which the TAs demonstrate how to perform training and inference learned models at the code level. Then, in the second half of the course, we will go over more advanced models, such as generative adversarial networks, efficient CNN architectures, self-supervised learning and pretrained language models, and graph neural networks, while working on a project that aims to solve a real-world problem utilizing one of them.  

**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  

**TAs** 

Taewook Nam (namsan@kaist.ac.kr)  
Jinheon Baek (jinheon.baek@kaist.ac.kr)  
Jaehyeong Jo (harryjo97@kaist.ac.kr)  
Wonyong Jeong (wyjeong@kaist.ac.kr)  
Dongchan Min (alsehdcks95@kaist.ac.kr)  


**Programming Environment:**
We will be using [Pytorch](https://pytorch.org/) as the official ML library.

**Office:** 
Building #9, 9201, KAIST Seoul Campus. 
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading** - You will be graded by the total absolute score, and not the relative ranking. Every project groups will be also graded solely based on the absolute quality of the final project outcome, not in comparison to others.
* Mid-term Exam: 20% - The exam will cover the basic topics of deep learning taught in the first half of the course.
* Final Exam: 20% - This will cover the advanced topics taught in the second half of the course. 
* Programming Assignments: 30% - You will work on two to three programming assignments and submit the code as well as the report.
* Attendance and Participation: 20% - Active participation during or off-class hours will be highly appreciated.

## Course Schedule

| Dates | Topic | 
|---|:---|
|3/3| Course Introduction |
|3/8| Review of Machine Learning Basics **(Video Lecture)** |
|3/10| Feedforward Neural Networks (Cost Function, Activations)
|3/15| Feedforward Neural Networks (Backpropagation, Why deep learning works)
|3/17| Pytorch Basics, Setting Up AWS, and Feedforward Neural Networks **(Programming)** 
|3/22| Convolutional Neural Networks
|3/24| Modern CNNs (GoogLeNet, ResNet, DenseNet)
|3/29| Recurrent Neural Networks (LSTM, Seq2Seq)
|3/31| Attention Mechanisms and Attentional RNNs
|4/5| Regularization for Deep Learning (L2/L1, Ensemble)  
|4/7| Regularization for Deep Learning (Dropout, Data Augmentations)
|4/12| Optimization Techniques for Deep Learning (Challenges, Adpative SGDs) 
|4/14| Optimization Techniques for Deep Learning (Adaptive SGDs and Second-order Methods) 
|4/19| CNN, RNN, Regularization **(Programming)**
|4/21| **Mid-term Exam**
|4/26| Autoencoders
|4/28| Introduction to Deep Generative Models, Variational Autoencoders 
|5/3| Generative Adversarial Networks
|5/10| Advanced GANs (WGAN, StyleGAN)
|5/12|  Optimization, VAEs and GANs **(Programming)** 
|5/17| Advanced CNN Architectures (MobileNets, ShuffleNets) 
|5/19| Advanced CNN Architectures (EfficientNets, NFNets) **(1st Assignment Due)**
|5/24| Object Detection and Segmentation 
|5/26| Transformers and Pretrained Language Models (BERT, GPT)
|5/31| Transfomers and Pretrained Language Models **(Programming)**
|6/2| Vision Transformers
|6/7| Graph Neural Networks (GCN, GAT, GIN)
|6/9| Graph Neural Networks (Pooling Methods, Graph Generation)  
|6/14| Q&A Session for the Final Exam
|6/16| **Final Exam** **(2nd Assignment Due)**

## Reading List
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114.pdf) Auto-Encoding Variational Bayes, ICLR 2014.   
[[Goodfellow et al. 14]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) Generative Adversarial Nets, NIPS 2014.   
[[Radford et al. 15]](https://arxiv.org/abs/1511.06434) Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.   
[[Chen et al. 16]](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016.   
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017.  
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.  
[[Karras et al. 18]](https://openreview.net/forum?id=Hk99zCeAb) Progressive Growing of GANs for Improved Quality, Stability, and Variation, ICLR 2018.  
[[Brock et al. 19]](https://openreview.net/pdf?id=B1xsqj09Fm) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019.  
[[Karras et al. 19]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019.  
[[Li et al. 16]](https://arxiv.org/pdf/1511.05493.pdf) Gated Graph Sequence Neural Networks, ICLR 2016.  
[[Hamilton et al. 17]](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) Inductive Representation Learning on Large Graphs, NIPS 2017.  
[[Kipf and Welling 17]](https://openreview.net/pdf?id=SJU4ayYgl) Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.  
[[Velickovic et al. 18]](https://openreview.net/pdf?id=rJXMpikCZ) Graph Attention Networks, ICLR 2018.   
[[Xu et al. 19]](https://openreview.net/forum?id=ryGs6iA5Km) How Powerful are Graph Neural Networks?, ICLR 2019.  
[[Ying et al. 18]](https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf) Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS 2018.  
[[Dosovitskiy et al. 14]](https://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf) Discriminative Unsupervised Feature Learning with Convolutional Neural Networks, NIPS 2014.  
[[Gidaris et al. 18]](https://openreview.net/pdf?id=S1v4N2l0-) Unsupervised Representation Learning by Predicting Image Rotations, ICLR 2018.  
[[He et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020.  
[[Chen et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/6165-Paper.pdf) A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020.  
[[Devlin et al. 19]](https://www.aclweb.org/anthology/N19-1423.pdf) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.  
[[Clark et al. 20]](https://openreview.net/pdf?id=r1xMH1BtvB) ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, ICLR 2020.  


