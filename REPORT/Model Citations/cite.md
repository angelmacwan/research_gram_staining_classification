```bibtex
@article{yuan2022volo,
  title={Volo: Vision outlooker for visual recognition},
  author={Yuan, Li and Hou, Qibin and Jiang, Zihang and Feng, Jiashi and Yan, Shuicheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}

@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}

@misc{gu2022convformercombiningcnntransformer,
      title={ConvFormer: Combining CNN and Transformer for Medical Image Segmentation},
      author={Pengfei Gu and Yejia Zhang and Chaoli Wang and Danny Z. Chen},
      year={2022},
      eprint={2211.08564},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2211.08564},
}

@misc{simonyan2015deepconvolutionalnetworkslargescale,
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
      author={Karen Simonyan and Andrew Zisserman},
      year={2015},
      eprint={1409.1556},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1409.1556},
}

@INPROCEEDINGS {9711046,
author = { Chen, Zhengsu and Xie, Lingxi and Niu, Jianwei and Liu, Xuefeng and Wei, Longhui and Tian, Qi },
booktitle = { 2021 IEEE/CVF International Conference on Computer Vision (ICCV) },
title = {{ Visformer: The Vision-friendly Transformer }},
year = {2021},
volume = {},
ISSN = {},
pages = {569-578},
abstract = { The past year has witnessed the rapid development of applying the Transformer module to vision problems. While some researchers have demonstrated that Transformer-based models enjoy a favorable ability of fitting data, there are still growing number of evidences showing that these models suffer over-fitting especially when the training data is limited. This paper offers an empirical study by performing step-by-step operations to gradually transit a Transformer-based model to a convolution-based model. The results we obtain during the transition process deliver useful messages for improving visual recognition. Based on these observations, we propose a new architecture named Visformer, which is abbreviated from the ‘Vision-friendly Transformer’. With the same computational complexity, Visformer outperforms both the Transformer-based and convolution-based models in terms of ImageNet classification accuracy, and the advantage becomes more significant when the model complexity is lower or the training set is smaller. The code is available at https://github.com/danczs/Visformer. },
keywords = {Convolutional codes;Training;Visualization;Computer vision;Protocols;Computational modeling;Fitting},
doi = {10.1109/ICCV48922.2021.00063},
url = {https://doi.ieeecomputersociety.org/10.1109/ICCV48922.2021.00063},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Oct}

@article{d2021convit,
  title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  author={d'Ascoli, St{\'e}phane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  journal={arXiv preprint arXiv:2103.10697},
  year={2021}
}

@misc{li2023rethinkingvisiontransformersmobilenet,
      title={Rethinking Vision Transformers for MobileNet Size and Speed},
      author={Yanyu Li and Ju Hu and Yang Wen and Georgios Evangelidis and Kamyar Salahi and Yanzhi Wang and Sergey Tulyakov and Jian Ren},
      year={2023},
      eprint={2212.08059},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.08059},
}


@misc{xie2017aggregatedresidualtransformationsdeep,
      title={Aggregated Residual Transformations for Deep Neural Networks},
      author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
      year={2017},
      eprint={1611.05431},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1611.05431},
}

@INPROCEEDINGS {9711309,
author = { Chen, Chun-Fu Richard and Fan, Quanfu and Panda, Rameswar },
booktitle = { 2021 IEEE/CVF International Conference on Computer Vision (ICCV) },
title = {{ CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification }},
year = {2021},
volume = {},
ISSN = {},
pages = {347-356},
abstract = { The recently developed vision transformer (ViT) has achieved promising results on image classification compared to convolutional neural networks. Inspired by this, in this paper, we study how to learn multi-scale feature representations in transformer models for image classification. To this end, we propose a dual-branch transformer to com-bine image patches (i.e., tokens in a transformer) of different sizes to produce stronger image features. Our approach processes small-patch and large-patch tokens with two separate branches of different computational complexity and these tokens are then fused purely by attention multiple times to complement each other. Furthermore, to reduce computation, we develop a simple yet effective token fusion module based on cross attention, which uses a single token for each branch as a query to exchange information with other branches. Our proposed cross-attention only requires linear time for both computational and memory complexity instead of quadratic time otherwise. Extensive experiments demonstrate that our approach performs better than or on par with several concurrent works on vision transformer, in addition to efficient CNN models. For example, on the ImageNet1K dataset, with some architectural changes, our approach outperforms the recent DeiT by a large margin of 2% with a small to moderate increase in FLOPs and model parameters. Our source codes and models are available at https://github.com/IBM/CrossViT. },
keywords = {Image segmentation;Computer vision;Image recognition;Computational modeling;Semantics;Memory management;Object detection},
doi = {10.1109/ICCV48922.2021.00041},
url = {https://doi.ieeecomputersociety.org/10.1109/ICCV48922.2021.00041},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Oct}


@misc{touvron2022deitiiirevengevit,
      title={DeiT III: Revenge of the ViT},
      author={Hugo Touvron and Matthieu Cord and Hervé Jégou},
      year={2022},
      eprint={2204.07118},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.07118},
}

@inproceedings{beit,
    title={{BEiT}: {BERT} Pre-Training of Image Transformers},
    author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=p-BhZSz59o4}
}

@article{beitv2,
    title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
    author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
    year={2022},
    eprint={2208.06366},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}


@inproceedings{cai2023efficientvit,
  title={Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17302--17313},
  year={2023}
}
@article{zhang2024efficientvit,
  title={EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss},
  author={Zhang, Zhuoyang and Cai, Han and Han, Song},
  journal={arXiv preprint arXiv:2402.05008},
  year={2024}
}
@article{chen2024deep,
  title={Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models},
  author={Chen, Junyu and Cai, Han and Chen, Junsong and Xie, Enze and Yang, Shang and Tang, Haotian and Li, Muyang and Lu, Yao and Han, Song},
  journal={arXiv preprint arXiv:2410.10733},
  year={2024}
}

@misc{trockman2022patchesneed,
      title={Patches Are All You Need?},
      author={Asher Trockman and J. Zico Kolter},
      year={2022},
      eprint={2201.09792},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.09792},
}

@inproceedings{vasufastvit2023,
  author = {Pavan Kumar Anasosalu Vasu and James Gabriel and Jeff Zhu and Oncel Tuzel and Anurag Ranjan},
  title = {FastViT:  A Fast Hybrid Vision Transformer using Structural Reparameterization},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2023}
}

@misc{touvron2021trainingdataefficientimagetransformers,
      title={Training data-efficient image transformers & distillation through attention},
      author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
      year={2021},
      eprint={2012.12877},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2012.12877},
}


@inproceedings{yu2024inceptionnext,
  title={Inceptionnext: When inception meets convnext},
  author={Yu, Weihao and Zhou, Pan and Yan, Shuicheng and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5672--5683},
  year={2024}
}


@misc{szegedy2016inceptionv4inceptionresnetimpactresidual,
      title={Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning},
      author={Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke and Alex Alemi},
      year={2016},
      eprint={1602.07261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1602.07261},
}

@misc{huang2018denselyconnectedconvolutionalnetworks,
      title={Densely Connected Convolutional Networks},
      author={Gao Huang and Zhuang Liu and Laurens van der Maaten and Kilian Q. Weinberger},
      year={2018},
      eprint={1608.06993},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1608.06993},
}

@article{han2020model,
  title={Model rubik’s cube: Twisting resolution, depth and width for tinynets},
  author={Han, Kai and Wang, Yunhe and Zhang, Qiulin and Zhang, Wei and Xu, Chunjing and Zhang, Tong},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={19353--19364},
  year={2020}
}

@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
