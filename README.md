# Awesome Video Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=Xiefan-Guo/Awesome-Video-Generation) ![GitHub stars](https://img.shields.io/github/stars/Xiefan-Guo/Awesome-Video-Generation?color=green)  ![GitHub forks](https://img.shields.io/github/forks/Xiefan-Guo/Awesome-Video-Generation?color=9cf)

A curated list of ***Video Generation*** papers and resources.

## Contents

- [Awesome Video Generation](#awesome-video-generation)
  - [Contents](#contents)
  - [Unconditional Video Generation](#unconditional-video-generation)
  - [Video Prediction](#video-prediction)
  - [Video-to-Video](#video-to-video)

## Unconditional Video Generation

**Generating Videos with Scene Dynamics.**<br>
*Carl Vondrick, Hamed Pirsiavash, Antonio Torralba.*<br>
*NeurIPS 2016.* [[PDF]](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf) [[Project]](http://www.cs.columbia.edu/~vondrick/tinyvideo/) [[Official Lua]](https://github.com/cvondrick/videogan) [[Unofficial PyTorch]](https://github.com/batsa003/videogan)

**Temporal Generative Adversarial Nets with Singular Value Clipping.**<br>
*Masaki Saito, Eiichi Matsumoto, Shunta Saito.*<br>
*ICCV 2017.* [[PDF]](https://arxiv.org/pdf/1611.06624.pdf) [[Project]](https://pfnet-research.github.io/tgan/) [[Official Chainer]](https://github.com/pfnet-research/tgan) [[Unofficial PyTorch]](https://github.com/proceduralia/tgan-pytorch)

**MoCoGAN: Decomposing Motion and Content for Video Generation.**<br>
*Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz.*<br>
*CVPR 2018.* [[PDF]](https://arxiv.org/pdf/1707.04993.pdf) [[Official PyTorch]](https://github.com/sergeytulyakov/mocogan)

**G³AN:  Disentangling appearance and motion for video generation.**<br>
*Yaohui Wang, Piotr Bilinski, Francois Bremond, Antitza Dantcheva.*<br>
*CVPR 2020.* [[PDF]](https://arxiv.org/pdf/1912.05523.pdf) [[Project]](https://wyhsirius.github.io/G3AN/) [[Official PyTorch]](https://github.com/wyhsirius/g3an-project)

**A Good Image Generator Is What You Need for High-Resolution Video Synthesis.**<br>
*Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, Sergey Tulyakov.*<br>
*ICLR 2021.* [[PDF]](https://arxiv.org/pdf/2104.15069.pdf) [[Project]](https://bluer555.github.io/MoCoGAN-HD/) [[Official PyTorch]](https://github.com/snap-research/MoCoGAN-HD) [[Talk]](https://papertalk.org/papertalks/29015) [[Slide]](https://iclr.cc/media/Slides/iclr/2021/virtual(03-08-00)-03-08-00UTC-2810-a_good_image.pdf)

**InMoDeGAN: Interpretable Motion Decomposition Generative Adversarial Network for Video Generation.**<br>
*Yaohui Wang, François Brémond, Antitza Dantcheva.*<br>
*arXiv 2021.* [[PDF]](https://arxiv.org/pdf/2101.03049.pdf) [[Project]](https://wyhsirius.github.io/InMoDeGAN/) [[Official PyTorch]](https://github.com/wyhsirius/InMoDeGAN-project)

## Video Prediction

**Probabilistic Video Prediction From Noisy Data With a Posterior Confidence.**<br>
*Yunbo Wang, Jiajun Wu, Mingsheng Long, Joshua B. Tenenbaum.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Probabilistic_Video_Prediction_From_Noisy_Data_With_a_Posterior_Confidence_CVPR_2020_paper.pdf)

**Disentangling Physical Dynamics From Unknown Factors for Unsupervised Video Prediction.**<br>
*Vincent Le Guen, Nicolas Thome.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/vincent-leguen/PhyDNet) [[Video]](https://www.youtube.com/watch?v=_edOGTNSC1U)

**Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction.**<br>
*Beibei Jin, Yu Hu, Qiankun Tang, Jingyu Niu, Zhiping Shi, Yinhe Han, Xiaowei Li.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Exploring_Spatial-Temporal_Multi-Frequency_Analysis_for_High-Fidelity_and_Temporal-Consistency_Video_Prediction_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/Bei-Jin/STMFANet) [[Video]](https://www.youtube.com/watch?v=cemetXMOInU)

**Multi-view Action Recognition using Cross-view Video Prediction.**<br>
*Shruti Vyas, Yogesh S Rawat, Mubarak Shah.*<br>
*ECCV 2020.* [[PDF]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720426.pdf) [[Official Keras]](https://github.com/svyas23/cross-view-action)

**Greedy Hierarchical Variational Autoencoders for Large-Scale Video Prediction.**<br>
*Bohan Wu, Suraj Nair, Roberto Martin-Martin, Li Fei-Fei, Chelsea Finn.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Greedy_Hierarchical_Variational_Autoencoders_for_Large-Scale_Video_Prediction_CVPR_2021_paper.pdf) [[Project]](https://sites.google.com/view/ghvae)

**Hierarchical Video Prediction Using Relational Layouts for Human-Object Interactions.**<br>
*Navaneeth Bodla, Gaurav Shrivastava, Rama Chellappa, Abhinav Shrivastava.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Bodla_Hierarchical_Video_Prediction_Using_Relational_Layouts_for_Human-Object_Interactions_CVPR_2021_paper.pdf) [[Project]](https://horn-video.github.io/)

**Learning Semantic-Aware Dynamics for Video Prediction.**<br>
*Xinzhu Bei, Yanchao Yang, Stefano Soatto.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Bei_Learning_Semantic-Aware_Dynamics_for_Video_Prediction_CVPR_2021_paper.pdf)

**Video Prediction Recalling Long-Term Motion Context via Memory Alignment Learning.**<br>
*Sangmin Lee, Hak Gu Kim, Dae Hwi Choi, Hyung-Il Kim, Yong Man Ro.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Video_Prediction_Recalling_Long-Term_Motion_Context_via_Memory_Alignment_Learning_CVPR_2021_paper.pdf) [[Official PyTorch]](https://github.com/sangmin-git/LMC-Memory)

**MotionRNN: A Flexible Model for Video Prediction With Spacetime-Varying Motions.**<br>
*Haixu Wu, Zhiyu Yao, Jianmin Wang, Mingsheng Long.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_MotionRNN_A_Flexible_Model_for_Video_Prediction_With_Spacetime-Varying_Motions_CVPR_2021_paper.pdf)

**Deep Learning in Latent Space for Video Prediction and Compression.**<br>
*Bowen Liu, Yu Chen, Shiyu Liu, Hun-Seok Kim.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Deep_Learning_in_Latent_Space_for_Video_Prediction_and_Compression_CVPR_2021_paper.pdf) [[Code]](https://github.com/BowenL0218/Video-compression)

**A Hierarchical Variational Neural Uncertainty Model for Stochastic Video Prediction.**<br>
*Moitreya Chatterjee, Narendra Ahuja, Anoop Cherian.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chatterjee_A_Hierarchical_Variational_Neural_Uncertainty_Model_for_Stochastic_Video_Prediction_ICCV_2021_paper.pdf)

## Video-to-Video

**Video-to-Video Synthesis.**<br>
*Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro.*<br>
*NeurIPS 2018.* [[PDF]](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf) [[Project]](https://tcwang0509.github.io/vid2vid/) [[Official PyTorch]](https://github.com/NVIDIA/vid2vid) [[Video(short)]](https://www.youtube.com/watch?v=5zlcXTCpQqM) [[Video(full)]](https://www.youtube.com/watch?v=GrP_aOSXt5U)

**Few-shot Video-to-Video Synthesis.**<br>
*Ting-Chun Wang, Ming-Yu Liu, Andrew Tao, Guilin Liu, Jan Kautz, Bryan Catanzaro.*<br>
*NeurIPS 2019.* [[PDF]](https://arxiv.org/pdf/1910.12713.pdf) [[Project]](https://nvlabs.github.io/few-shot-vid2vid/) [[Official PyTorch]](https://github.com/NVLabs/few-shot-vid2vid) [[Video]](https://www.youtube.com/watch?v=8AZBuyEuDqc)

**World-Consistent Video-to-Video Synthesis.**<br>
*Arun Mallya, Ting-Chun Wang, Karan Sapra, Ming-Yu Liu.*<br>
*ECCV 2020.* [[PDF]](https://nvlabs.github.io/wc-vid2vid/files/wc-vid2vid.pdf) [[Project]](https://nvlabs.github.io/wc-vid2vid/) [[Official PyTorch]](https://github.com/NVlabs/imaginaire)


