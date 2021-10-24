# Awesome Video Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=Xiefan-Guo/Awesome-Video-Generation) ![GitHub stars](https://img.shields.io/github/stars/Xiefan-Guo/Awesome-Video-Generation?color=green)  ![GitHub forks](https://img.shields.io/github/forks/Xiefan-Guo/Awesome-Video-Generation?color=9cf)

A curated list of ***Video Generation*** papers and resources.

## Contents

- [Awesome Video Generation](#awesome-video-generation)
  - [Contents](#contents)
  - [Unconditional Video Generation](#unconditional-video-generation)
  - [Video Prediction](#video-prediction)
  - [Image-to-Video](#image-to-video)
  - [Video-to-Video](#video-to-video)
  - [Video Generation](#video-generation)
  - [Talking Head](#talking-head)

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

**Probabilistic Video Generation using Holistic Attribute Control.**<br>
*He, Jiawei and Lehrmann, Andreas and Marino, Joseph and Mori, Greg and Sigal, Leonid.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Jiawei_He_Probabilistic_Video_Generation_ECCV_2018_paper.pdf)

**TwoStreamVAN: Improving Motion Modeling in Video Generation.**<br>
*Ximeng Sun, Huijuan Xu, Kate Saenko.*<br>
*WACV 2020.* [[PDF]](https://openaccess.thecvf.com/content_WACV_2020/papers/Sun_TwoStreamVAN_Improving_Motion_Modeling_in_Video_Generation_WACV_2020_paper.pdf) [[Official PyTorch]](https://github.com/sunxm2357/TwoStreamVAN/) [[Video]](https://www.youtube.com/watch?v=mIG0WXolM7A&t=3110s)

**Jointly Trained Image and Video Generation using Residual Vectors.**<br>
*Yatin Dandi, Aniket Das, Soumye Singhal, Vinay Namboodiri, Piyush Rai.*<br>
*WACV 2020.* [[PDF]](https://openaccess.thecvf.com/content_WACV_2020/papers/Dandi_Jointly_Trained_Image_and_Video_Generation_using_Residual_Vectors_WACV_2020_paper.pdf) [[Video]](https://www.youtube.com/watch?v=3usvANd5vyE&t=182s)

**ImaGINator: Conditional Spatio-Temporal GAN for Video Generation.**<br>
*Yaohui WANG, Piotr Bilinski, Francois Bremond, Antitza Dantcheva.*<br>
*WACV 2020.* [[PDF]](https://openaccess.thecvf.com/content_WACV_2020/papers/WANG_ImaGINator_Conditional_Spatio-Temporal_GAN_for_Video_Generation_WACV_2020_paper.pdf) [[Official PyTorch]](https://github.com/wyhsirius/ImaGINator) [[Video]](https://www.youtube.com/watch?v=0RYqfwR5YNk&t=3098s)

**G³AN:  Disentangling appearance and motion for video generation.**<br>
*Yaohui Wang, Piotr Bilinski, Francois Bremond, Antitza Dantcheva.*<br>
*CVPR 2020.* [[PDF]](https://arxiv.org/pdf/1912.05523.pdf) [[Project]](https://wyhsirius.github.io/G3AN/) [[Official PyTorch]](https://github.com/wyhsirius/g3an-project)

**Non-Adversarial Video Synthesis With Learned Priors.**<br>
*Abhishek Aich, Akash Gupta, Rameswar Panda, Rakib Hyder, M. Salman Asif, Amit K. Roy-Chowdhury.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Aich_Non-Adversarial_Video_Synthesis_With_Learned_Priors_CVPR_2020_paper.pdf) [[Project]](https://abhishekaich27.github.io/navsynth.html) [[Official PyTorch]](https://github.com/abhishekaich27/Navsynth) [[Slide]](https://abhishekaich27.github.io/data/Project_pages/CVPR_2020/CVPR_2020_Navsynth_slides.pdf) [[Poster]](https://abhishekaich27.github.io/data/Project_pages/CVPR_2020/CVPR_2020_Navsynth_poster.pdf)

**Temporal Shift GAN for Large Scale Video Generation.**<br>
*Andres Munoz, Mohammadreza Zolfaghari, Max Argus, Thomas Brox.*<br>
*WACV 2021.* [[PDF]](https://openaccess.thecvf.com/content/WACV2021/papers/Munoz_Temporal_Shift_GAN_for_Large_Scale_Video_Generation_WACV_2021_paper.pdf)

**A Good Image Generator Is What You Need for High-Resolution Video Synthesis.**<br>
*Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, Sergey Tulyakov.*<br>
*ICLR 2021.* [[PDF]](https://arxiv.org/pdf/2104.15069.pdf) [[Project]](https://bluer555.github.io/MoCoGAN-HD/) [[Official PyTorch]](https://github.com/snap-research/MoCoGAN-HD) [[Talk]](https://papertalk.org/papertalks/29015) [[Slide]](https://iclr.cc/media/Slides/iclr/2021/virtual(03-08-00)-03-08-00UTC-2810-a_good_image.pdf)

**InMoDeGAN: Interpretable Motion Decomposition Generative Adversarial Network for Video Generation.**<br>
*Yaohui Wang, François Brémond, Antitza Dantcheva.*<br>
*arXiv 2021.* [[PDF]](https://arxiv.org/pdf/2101.03049.pdf) [[Project]](https://wyhsirius.github.io/InMoDeGAN/) [[Official PyTorch]](https://github.com/wyhsirius/InMoDeGAN-project)

## Video Prediction

**Unsupervised Learning for Physical Interaction through Video Prediction.**<br>
*Chelsea Finn, Ian Goodfellow, Sergey Levine.*<br>
*NeurIPS 2016.* [[PDF]](https://papers.nips.cc/paper/2016/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf) [[PyTorch]](https://github.com/NjuHaoZhang/VP_goodfellow_nips2016_pytorch)

**Deep Multi-scale Video Prediction Beyond Mean Square Error.**<br>
*Michael Mathieu, Camille Couprie, Yann LeCun.*<br>
*ICLR 2016.* [[PDF]](https://arxiv.org/pdf/1511.05440.pdf) [[Project]](https://cs.nyu.edu/~mathieu/iclr2016.html) [[Official Lua]](https://github.com/coupriec/VideoPredictionICLR2016) [[Dataset]](http://perso.esiee.fr/~coupriec/MathieuICLR16TestCode.zip)

**Decomposing Motion and Content for Natural Video Sequence Prediction.**<br>
*Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin and Honglak Lee.*<br>
*ICLR 2017.* [[PDF]](https://openreview.net/pdf?id=rkEFLFqee) [[Official TensorFlow]](https://github.com/rubenvillegas/iclr2017mcnet)

**Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning.**<br>
*William Lotter, Gabriel Kreiman, David D. Cox.*<br>
*ICLR 2017.* [[PDF]](https://arxiv.org/pdf/1605.08104.pdf) [[Official Keras]](https://github.com/coxlab/prednet)

**Flexible Spatio-Temporal Networks for Video Prediction.**<br>
*Chaochao Lu, Michael Hirsch, Bernhard Scholkopf.*<br>
*CVPR 2017.* [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Flexible_Spatio-Temporal_Networks_CVPR_2017_paper.pdf)

**Learning to Generate Long-term Future via Hierarchical Prediction.**<br>
*Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, Honglak Lee.*<br>
*ICML 2017.* [[PDF]](http://proceedings.mlr.press/v70/villegas17a/villegas17a.pdf) [[Project]](https://sites.google.com/a/umich.edu/rubenevillegas/hierch_vid) [[Official TensorFlow]](https://github.com/rubenvillegas/icml2017hierchvid)

**Dual Motion GAN for Future-Flow Embedded Video Prediction.**<br>
*Xiaodan Liang, Lisa Lee, Wei Dai, Eric P. Xing.*<br>
*ICCV 2017.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liang_Dual_Motion_GAN_ICCV_2017_paper.pdf)

**Stochastic Variational Video Prediction.**<br>
*Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, Sergey Levine.*<br>
*ICLR 2018.* [[PDF]](https://arxiv.org/pdf/1710.11252.pdf)

**Structure Preserving Video Prediction.**<br>
*Jingwei Xu, Bingbing Ni, Zefan Li, Shuo Cheng, Xiaokang Yang.*<br>
*CVPR 2018.* [[PDF]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Structure_Preserving_Video_CVPR_2018_paper.pdf)

**Folded Recurrent Neural Networks for Future Video Prediction.**<br>
*Oliu, Marc and Selva, Javier and Escalera, Sergio.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Marc_Oliu_Folded_Recurrent_Neural_ECCV_2018_paper.pdf) [[Official TensorFlow]](https://github.com/moliusimon/frnn)

**Flow-Grounded Spatial-Temporal Video Prediction from Still Images.**<br>
*Li, Yijun and Fang, Chen and Yang, Jimei and Wang, Zhaowen and Lu, Xin and Yang, Ming-Hsuan.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yijun_Li_Flow-Grounded_Spatial-Temporal_Video_ECCV_2018_paper.pdf) [[Official Lua]](https://github.com/Yijunmaverick/FlowGrounded-VideoPrediction)

**DYAN: A Dynamical Atoms-Based Network For Video Prediction.**<br>
*Liu, Wenqian and Sharma, Abhishek and Camps, Octavia and Sznaier, Mario.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Wenqian_Liu_DYAN_A_Dynamical_ECCV_2018_paper.pdf) [[Official PyTorch]](https://github.com/liuem607/DYAN)

**SDC-Net: Video prediction using spatially-displaced convolution.**<br>
*Reda, Fitsum A. and Liu, Guilin and Shih, Kevin J. and Kirby, Robert and Barker, Jon and Tarjan, David and Tao, Andrew and Catanzaro, Bryan.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Fitsum_Reda_SDC-Net_Video_prediction_ECCV_2018_paper.pdf)

**ContextVP: Fully Context-Aware Video Prediction.**<br>
*Byeon, Wonmin and Wang, Qin and Kumar Srivastava, Rupesh and Koumoutsakos, Petros.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Wonmin_Byeon_ContextVP_Fully_Context-Aware_ECCV_2018_paper.pdf)

**Video Prediction via Selective Sampling.**<br>
*Jingwei Xu, Bingbing Ni, Xiaokang Yang.*<br>
*NeurIPS 2018.* [[PDF]](https://papers.nips.cc/paper/2018/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf) [[Official Code]](https://github.com/xjwxjw/VPSS)

**Learning to Decompose and Disentangle Representations for Video Prediction.**<br>
*Jun-Ting Hsieh, Bingbin Liu, De-An Huang, Li F. Fei-Fei, Juan Carlos Niebles.*
*NeurIPS 2018.* [[PDF]](https://papers.nips.cc/paper/2018/file/496e05e1aea0a9c4655800e8a7b9ea28-Paper.pdf) [[Official PyTorch]](https://github.com/jthsieh/DDPAE-video-prediction)

**Stochastic Adversarial Video Prediction.**<br>
*Alex X. Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, Sergey Levine.*<br>
*ICLR 2019. (Reject)* [[PDF]](https://arxiv.org/pdf/1804.01523.pdf) [[Project]](https://alexlee-gk.github.io/video_prediction/) [[Official TensorFlow]](https://github.com/alexlee-gk/video_prediction)

**Time-Agnostic Prediction: Predicting Predictable Video Frames.**<br>
*Dinesh Jayaraman, Frederik Ebert, Alexei Efros, Sergey Levine.*<br>
*ICLR 2019.* [[PDF]](https://openreview.net/pdf?id=SyzVb3CcFX)

**Eidetic 3D LSTM: A Model for Video Prediction and Beyond.**<br>
*Yunbo Wang, Lu Jiang, Ming-Hsuan Yang, Li-Jia Li, Mingsheng Long, Li Fei-Fei.*<br>
*ICLR 2019.* [[PDF]](https://openreview.net/pdf?id=B1lKS2AqtX) [[Official TensorFlow]](https://github.com/google/e3d_lstm)

**Compositional Video Prediction.**<br>
*Yufei Ye, Maneesh Singh, Abhinav Gupta, Shubham Tulsiani.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Compositional_Video_Prediction_ICCV_2019_paper.pdf) [[Project]](https://judyye.github.io/CVP/) [[Official PyTorch]](https://github.com/JudyYe/CVP)

**Improved Conditional VRNNs for Video Prediction.**<br>
*Lluis Castrejon, Nicolas Ballas, Aaron Courville.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Castrejon_Improved_Conditional_VRNNs_for_Video_Prediction_ICCV_2019_paper.pdf)

**Disentangling Propagation and Generation for Video Prediction.**<br>
*Hang Gao, Huazhe Xu, Qi-Zhi Cai, Ruth Wang, Fisher Yu, Trevor Darrell.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_Disentangling_Propagation_and_Generation_for_Video_Prediction_ICCV_2019_paper.pdf)

**SME-Net: Sparse Motion Estimation for Parametric Video Prediction Through Reinforcement Learning.**<br>
*Yung-Han Ho, Chuan-Yuan Cho, Wen-Hsiao Peng, Guo-Lun Jin.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ho_SME-Net_Sparse_Motion_Estimation_for_Parametric_Video_Prediction_Through_Reinforcement_ICCV_2019_paper.pdf) [[Official TensorFlow]](https://github.com/hectorho0409/SME_release)

**Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction.**<br>
*Yunji Kim, Seonghyeon Nam, In Cho, Seon Joo Kim.*<br>
*NeurIPS 2019.* [[PDF]](https://papers.nips.cc/paper/2019/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf) [[Offical TensorFlow]](https://github.com/YunjiKim/Unsupervised-Keypoint-Learning-for-Guiding-Class-conditional-Video-Prediction)

**High Fidelity Video Prediction with Large Stochastic Recurrent Neural Networks.**<br>
*Ruben Villegas, Arkanath Pathak, Harini Kannan, Dumitru Erhan, Quoc V. Le, Honglak Lee.*<br>
*NeurIPS 2019.* [[PDF]](https://papers.nips.cc/paper/2019/file/f7177163c833dff4b38fc8d2872f1ec6-Paper.pdf)

**VideoFlow: A Conditional Flow-Based Model for Stochastic Video Generation.**<br>
*Manoj Kumar, Mohammad Babaeizadeh, Dumitru Erhan, Chelsea Finn, Sergey Levine, Laurent Dinh, Durk Kingma.*<br>
*ICLR 2020.* [[PDF]](https://arxiv.org/pdf/1903.01434.pdf) [[Code]](https://github.com/tensorflow/tensor2tensor)

**Probabilistic Video Prediction From Noisy Data With a Posterior Confidence.**<br>
*Yunbo Wang, Jiajun Wu, Mingsheng Long, Joshua B. Tenenbaum.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Probabilistic_Video_Prediction_From_Noisy_Data_With_a_Posterior_Confidence_CVPR_2020_paper.pdf)

**Future Video Synthesis With Object Motion Prediction.**<br>
*Yue Wu, Rongrong Gao, Jaesik Park, Qifeng Chen.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Future_Video_Synthesis_With_Object_Motion_Prediction_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/YueWuHKUST/FutureVideoSynthesis)

**Disentangling Physical Dynamics From Unknown Factors for Unsupervised Video Prediction.**<br>
*Vincent Le Guen, Nicolas Thome.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/vincent-leguen/PhyDNet) [[Video]](https://www.youtube.com/watch?v=_edOGTNSC1U)

**Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction.**<br>
*Beibei Jin, Yu Hu, Qiankun Tang, Jingyu Niu, Zhiping Shi, Yinhe Han, Xiaowei Li.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Exploring_Spatial-Temporal_Multi-Frequency_Analysis_for_High-Fidelity_and_Temporal-Consistency_Video_Prediction_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/Bei-Jin/STMFANet) [[Video]](https://www.youtube.com/watch?v=cemetXMOInU)

**Stochastic Latent Residual Video Prediction.**<br>
*Jean-Yves Franceschi, Edouard Delasalles, Mickael Chen, Sylvain Lamprier, Patrick Gallinari.*<br>
*ICML 2020.* [[PDF]](http://proceedings.mlr.press/v119/franceschi20a/franceschi20a.pdf) [[Official PyTorch]](https://github.com/edouardelasalles/srvp)

**Multi-view Action Recognition using Cross-view Video Prediction.**<br>
*Shruti Vyas, Yogesh S Rawat, Mubarak Shah.*<br>
*ECCV 2020.* [[PDF]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720426.pdf) [[Official Keras]](https://github.com/svyas23/cross-view-action)

**Revisiting Hierarchical Approach for Persistent Long-Term Video Prediction.**<br>
*Wonkwang Lee, Whie Jung, Han Zhang, Ting Chen, Jing Yu Koh, Thomas Huang, Hyungsuk Yoon, Honglak Lee, Seunghoon Hong.*<br>
*ICLR 2021.* [[PDF]](https://openreview.net/pdf?id=3RLN4EPMdYd) [[Project]](https://1konny.github.io/HVP/) [[Official PyTorch]](https://github.com/1Konny/HVP)

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

**Autoregressive Latent Video Prediction with High-Fidelity Image Generator.**<br>
*Anonymous authors.*<br>
*Submitted to ICLR 2022.* [[PDF]](https://openreview.net/pdf?id=K-hiHQXEQog)

## Image-to-Video

**Pose Guided Human Video Generation.**<br>
*Yang, Ceyuan and Wang, Zhe and Zhu, Xinge and Huang, Chen and Shi, Jianping and Lin, Dahua.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Ceyuan_Yang_Pose_Guided_Human_ECCV_2018_paper.pdf)

**Learning to Forecast and Refine Residual Motion for Image-to-Video Generation.**<br>
*Zhao, Long and Peng, Xi and Tian, Yu and Kapadia, Mubbasir and Metaxas, Dimitris.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Long_Zhao_Learning_to_Forecast_ECCV_2018_paper.pdf) [[Project]](https://garyzhao.github.io/FRGAN/) [[Official PyTorch]](https://github.com/garyzhao/FRGAN)

**Video Generation From Single Semantic Label Map.**<br>
*Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, Xiaogang Wang.*<br>
*CVPR 2019.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Video_Generation_From_Single_Semantic_Label_Map_CVPR_2019_paper.pdf) [[Official PyTorch]](https://github.com/junting/seg2vid)

**Time Flies: Animating a Still Image With Time-Lapse Video As Reference.**<br>
*Chia-Chi Cheng, Hung-Yu Chen, Wei-Chen Chiu.**<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Time_Flies_Animating_a_Still_Image_With_Time-Lapse_Video_As_CVPR_2020_paper.pdf) [[Official PyTorch]](https://github.com/angelwmab/Time-Flies)

**Painting Many Pasts: Synthesizing Time Lapse Videos of Paintings.**<br>
*Amy Zhao, Guha Balakrishnan, Kathleen M. Lewis, Fredo Durand, John V. Guttag, Adrian V. Dalca.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Painting_Many_Pasts_Synthesizing_Time_Lapse_Videos_of_Paintings_CVPR_2020_paper.pdf) [[Project]](https://xamyzhao.github.io/timecraft/) [[Official TensorFlow]](https://github.com/xamyzhao/timecraft)

**Point-to-Point Video Generation.**<br>
*Tsun-Hsuan Wang, Yen-Chi Cheng, Chieh Hubert Lin, Hwann-Tzong Chen, Min Sun*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Point-to-Point_Video_Generation_ICCV_2019_paper.pdf) [[Official PyTorch]](https://github.com/yccyenchicheng/p2pvg)

**High-Quality Video Generation from Static Structural Annotations.**<br>
*Lu Sheng, Junting Pan, Jiaming Guo, Jing Shao, Chen Change Loy.*<br>
*IJCV 2020.* [[PDF]](https://link.springer.com/content/pdf/10.1007/s11263-020-01334-x.pdf) [[Official PyTorch]](https://github.com/junting/seg2vid)

**DTVNet: Dynamic Time-lapse Video Generation via Single Still Image.**<br>
*Jiangning Zhang, Chao Xu, Liang Liu, Mengmeng Wang, Xia Wu, Yong Liu, Yunliang Jiang.*<br>
*ECCV 2020.* [[PDF]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500290.pdf) [[Official PyTorch]](https://github.com/zhangzjn/DTVNet)

**Stochastic Image-to-Video Synthesis Using cINNs.**<br>
*Michael Dorkenwald, Timo Milbich, Andreas Blattmann, Robin Rombach, Konstantinos G. Derpanis, Bjorn Ommer.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Dorkenwald_Stochastic_Image-to-Video_Synthesis_Using_cINNs_CVPR_2021_paper.pdf) [[Project]](https://compvis.github.io/image2video-synthesis-using-cINNs/) [[Official PyTorch]](https://github.com/CompVis/image2video-synthesis-using-cINNs) [[Official PyTorch]](https://github.com/CompVis/interactive-image2video-synthesis)

**Understanding Object Dynamics for Interactive Image-to-Video Synthesis.**<br>
*Andreas Blattmann, Timo Milbich, Michael Dorkenwald, Bjorn Ommer.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Blattmann_Understanding_Object_Dynamics_for_Interactive_Image-to-Video_Synthesis_CVPR_2021_paper.pdf) [[Project]](https://compvis.github.io/interactive-image2video-synthesis/)

**Animating Pictures With Eulerian Motion Fields.**<br>
*Aleksander Holynski, Brian L. Curless, Steven M. Seitz, Richard Szeliski.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Holynski_Animating_Pictures_With_Eulerian_Motion_Fields_CVPR_2021_paper.pdf) [[Project]](https://eulerian.cs.washington.edu/)

**Pose-Guided Human Animation From a Single Image in the Wild.**<br>
*Jae Shin Yoon, Lingjie Liu, Vladislav Golyanik, Kripasindhu Sarkar, Hyun Soo Park, Christian Theobalt.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yoon_Pose-Guided_Human_Animation_From_a_Single_Image_in_the_Wild_CVPR_2021_paper.pdf)

**iPOKE: Poking a Still Image for Controlled Stochastic Video Synthesis.**<br>
*Andreas Blattmann, Timo Milbich, Michael Dorkenwald, Bjorn Ommer.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Blattmann_iPOKE_Poking_a_Still_Image_for_Controlled_Stochastic_Video_Synthesis_ICCV_2021_paper.pdf) [[Project]](https://compvis.github.io/ipoke/) [[Offficial PyTorch]](https://github.com/CompVis/ipoke)

## Video-to-Video

**Video-to-Video Synthesis.**<br>
*Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro.*<br>
*NeurIPS 2018.* [[PDF]](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf) [[Project]](https://tcwang0509.github.io/vid2vid/) [[Official PyTorch]](https://github.com/NVIDIA/vid2vid) [[Video(short)]](https://www.youtube.com/watch?v=5zlcXTCpQqM) [[Video(full)]](https://www.youtube.com/watch?v=GrP_aOSXt5U)

**Recycle-GAN: Unsupervised Video Retargeting.**<br>
*Aayush Bansal, Shugao Ma, Deva Ramanan, and Yaser Sheikh.*<br>
*ECCV 2018.* [[PDF]](https://arxiv.org/pdf/1808.05174.pdf) [[Project]](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/) [[Official PyTorch]](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/)

**Video-to-Video Translation with Global Temporal Consistency.**<br>
*Xingxing Wei, Jun Zhu, Sitong Feng, Hang Su.*<br>
*ACM MM 2018.* [[PDF]](https://ml.cs.tsinghua.edu.cn/~jun/pub/video2video.pdf )

**Unsupervised Video-to-Video Translation.**<br>
*Dina Bashkirova, Ben Usman, Kate Saenko.*<br>
*ICLR 2019.* [[PDF]](https://arxiv.org/pdf/1806.03698.pdf) [[Official TensorFlow]](https://github.com/dbash/CycleGAN3D)

**Mocycle-GAN: Unpaired Video-to-Video Translation.**<br>
*Yang Chen, Yingwei Pan, Ting Yao, Xinmei Tian, Tao Mei.*<br>
*ACM MM 2019.* [[PDF]](https://arxiv.org/pdf/1908.09514.pdf)

**Few-shot Video-to-Video Synthesis.**<br>
*Ting-Chun Wang, Ming-Yu Liu, Andrew Tao, Guilin Liu, Jan Kautz, Bryan Catanzaro.*<br>
*NeurIPS 2019.* [[PDF]](https://arxiv.org/pdf/1910.12713.pdf) [[Project]](https://nvlabs.github.io/few-shot-vid2vid/) [[Official PyTorch]](https://github.com/NVLabs/few-shot-vid2vid) [[Video]](https://www.youtube.com/watch?v=8AZBuyEuDqc)

**World-Consistent Video-to-Video Synthesis.**<br>
*Arun Mallya, Ting-Chun Wang, Karan Sapra, Ming-Yu Liu.*<br>
*ECCV 2020.* [[PDF]](https://nvlabs.github.io/wc-vid2vid/files/wc-vid2vid.pdf) [[Project]](https://nvlabs.github.io/wc-vid2vid/) [[Official PyTorch]](https://github.com/NVlabs/imaginaire)

**HyperCon: Image-to-Video Model Transfer for Video-to-Video Translation Tasks.**<br>
*Ryan Szeto, Mostafa El-Khamy, Jungwon Lee, Jason J. Corso.*<br>
*WACV 2021.* [[PDF]](https://openaccess.thecvf.com/content/WACV2021/papers/Szeto_HyperCon_Image-to-Video_Model_Transfer_for_Video-to-Video_Translation_Tasks_WACV_2021_paper.pdf)

**Unsupervised Multimodal Video-to-Video Translation via Self-Supervised Learning.**<br>
*Kangning Liu, Shuhang Gu, Andres Romero, Radu Timofte.*<br>
*WACV 2021.* [[PDF]](https://openaccess.thecvf.com/content/WACV2021/papers/Liu_Unsupervised_Multimodal_Video-to-Video_Translation_via_Self-Supervised_Learning_WACV_2021_paper.pdf)

## Video Generation

**Controllable Video Generation With Sparse Trajectories.**<br>
*Zekun Hao, Xun Huang, Serge Belongie.*
*CVPR 2018.* [[PDF]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hao_Controllable_Video_Generation_CVPR_2018_paper.pdf) [[Official PyTorch]](https://github.com/zekunhao1995/ControllableVideoGen)

**Attentive Semantic Video Generation Using Captions.**<br>
*Tanya Marwah, Gaurav Mittal, Vineeth N. Balasubramanian.*<br>
*ICCV 2017.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Marwah_Attentive_Semantic_Video_ICCV_2017_paper.pdf) [[Official TensorFlow]](https://github.com/Singularity42/cap2vid)

**Deep Video Generation, Prediction and Completion of Human Action Sequences.**<br>
*Cai, Haoye and Bai, Chunyan and Tai, Yu-Wing and Tang, Chi-Keung.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Chunyan_Bai_Deep_Video_Generation_ECCV_2018_paper.pdf) [[Project]](https://iamacewhite.github.io/supp/)

**X2Face: A network for controlling face generation using images, audio, and pose codes.**<br>
*Olivia Wiles, A. Sophia Koepke, Andrew Zisserman.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Olivia_Wiles_X2Face_A_network_ECCV_2018_paper.pdf) [[Official PyTorch]](https://github.com/oawiles/X2Face)

**Animating Arbitrary Objects via Deep Motion Transfer.**<br>
*Aliaksandr Siarohin, Stephane Lathuiliere, Sergey Tulyakov, Elisa Ricci, Nicu Sebe.*<br>
*CVPR 2019.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Siarohin_Animating_Arbitrary_Objects_via_Deep_Motion_Transfer_CVPR_2019_paper.pdf) [[Official PyTorch]](https://github.com/AliaksandrSiarohin/monkey-net)

**Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks.**<br>
*Lin Wang, S. Mohammad Mostafavi I., Yo-Sung Ho, Kuk-Jin Yoon.*<br>
*CVPR 2019.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf)

**End-To-End Time-Lapse Video Synthesis From a Single Outdoor Image.**<br>
*Seonghyeon Nam, Chongyang Ma, Menglei Chai, William Brendel, Ning Xu, Seon Joo Kim.*<br>
*CVPR 2019.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_End-To-End_Time-Lapse_Video_Synthesis_From_a_Single_Outdoor_Image_CVPR_2019_paper.pdf)

**View-LSTM: Novel-View Video Synthesis Through View Decomposition.**<br>
*Mohamed Ilyes Lakhal, Oswald Lanz, Andrea Cavallaro.*<br>
*ICCV 2019.* [[PDFhttps://openaccess.thecvf.com/content_ICCV_2019/papers/Lakhal_View-LSTM_Novel-View_Video_Synthesis_Through_View_Decomposition_ICCV_2019_paper.pdf]]

**Everybody Dance Now.**<br>
*Caroline Chan, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chan_Everybody_Dance_Now_ICCV_2019_paper.pdf) [[Project]](https://carolineec.github.io/everybody_dance_now/) [[Official PyTorch]](https://github.com/carolineec/EverybodyDanceNow) [[Dataset]](https://carolineec.github.io/everybody_dance_now/#data)

**First Order Motion Model for Image Animation.**<br>
*liaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, Nicu Sebe.*<br>
*NeurIPS 2019.* [[PDF]](https://arxiv.org/pdf/2003.00196.pdf) [[Official PyTorch]](https://github.com/AliaksandrSiarohin/first-order-model)

**Unsupervised object-centric video generation and decomposition in 3D.**<br>
*Paul Henderson, Christoph H. Lampert.*<br>
*NeurIPS 2020.* [[PDF]](https://papers.nips.cc/paper/2020/file/20125fd9b2d43e340a35fb0278da235d-Paper.pdf) [[Official TensorFlow]](https://github.com/pmh47/o3v)

**Diverse Video Generation using a Gaussian Process Trigger .**<br>
*Gaurav Shrivastava, Abhinav Shrivastava.*<br>
*ICLR 2021.* [[PDF]](https://openreview.net/pdf?id=Qm7R_SdqTpT) [[Project]](http://www.cs.umd.edu/~gauravsh/dvg.html) [[Official PyTorch]](https://github.com/shgaurav1/DVG)

**Playable Video Generation.**<br>
*Willi Menapace, Stephane Lathuiliere, Sergey Tulyakov, Aliaksandr Siarohin, Elisa Ricci.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Menapace_Playable_Video_Generation_CVPR_2021_paper.pdf)

**Deep Animation Video Interpolation in the Wild.**<br>
*Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, Ziwei Liu.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Siyao_Deep_Animation_Video_Interpolation_in_the_Wild_CVPR_2021_paper.pdf) [[Official PyTorch]](https://github.com/lisiyao21/AnimeInterp/)

**Motion Representations for Articulated Animation.**<br>
*Aliaksandr Siarohin, Oliver J. Woodford, Jian Ren, Menglei Chai, Sergey Tulyakov.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Siarohin_Motion_Representations_for_Articulated_Animation_CVPR_2021_paper.pdf) [[Official PyTorch]](https://github.com/snap-research/articulated-animation)

**Generative Video Transformer: Can Objects be the Words?.**<br>
*Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn.*<br>
*ICML 2021.* [[PDF]](http://proceedings.mlr.press/v139/wu21h/wu21h.pdf) [[Official PyTorch]](https://github.com/ahn-ml/OCVT)

**Click To Move: Controlling Video Generation With Sparse Motion.**<br>
*Pierfrancesco Ardino, Marco De Nadai, Bruno Lepri, Elisa Ricci, Stephane Lathuiliere.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ardino_Click_To_Move_Controlling_Video_Generation_With_Sparse_Motion_ICCV_2021_paper.pdf) [[Code]](https://github.com/PierfrancescoArdino/C2M)

**Sat2Vid: Street-View Panoramic Video Synthesis From a Single Satellite Image.**<br>
*Zuoyue Li, Zhenqiang Li, Zhaopeng Cui, Rongjun Qin, Marc Pollefeys, Martin R. Oswald.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Sat2Vid_Street-View_Panoramic_Video_Synthesis_From_a_Single_Satellite_Image_ICCV_2021_paper.pdf)


## Talking Head

**GANimation: Anatomically-aware Facial Animation from a Single Image.**<br>
*Albert Pumarola, Antonio Agudo, Aleix M. Martinez, Alberto Sanfeliu, Francesc Moreno-Noguer.*<br>
*ECCV 2018.* [[PDF]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.pdf) [[Official Code]](https://github.com/ageitgey/face_recognition)

**Hierarchical Cross-Modal Talking Face Generation With Dynamic Pixel-Wise Loss.**<br>
*Lele Chen, Ross K. Maddox, Zhiyao Duan, Chenliang Xu.*<br>
*CVPR 2019.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hierarchical_Cross-Modal_Talking_Face_Generation_With_Dynamic_Pixel-Wise_Loss_CVPR_2019_paper.pdf) [[Official PyTorch]](https://github.com/lelechen63/ATVGnet)

**Few-Shot Adversarial Learning of Realistic Neural Talking Head Models.**<br>
*Egor Zakharov, Aliaksandra Shysheya, Egor Burkov, Victor Lempitsky.*<br>
*ICCV 2019.* [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_Few-Shot_Adversarial_Learning_of_Realistic_Neural_Talking_Head_Models_ICCV_2019_paper.pdf) [[PyTorch]](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models) [[Video]](https://www.youtube.com/watch?v=ByfFufRhuRc)

**MakeItTalk: Speaker-Aware Talking-Head Animation.**<br>
*Yang Zhou, Xintong Han, Eli Shechtman, Jose Echevarria, Evangelos Kalogerakis, Dingzeyu Li.*<br>
*SIGGRAPH Asia 2020 and ToG 2020.* [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3414685.3417774) [[Project]](https://people.umass.edu/yangzhou/MakeItTalk/) [[Official PyTorch]](https://github.com/yzhou359/MakeItTalk) [[Video]](https://www.youtube.com/watch?v=OU6Ctzhpc6s)

**DAVD-Net: Deep Audio-Aided Video Decompression of Talking Heads.**<br>
*Xi Zhang, Xiaolin Wu, Xinliang Zhai, Xianye Ben, Chengjie Tu.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DAVD-Net_Deep_Audio-Aided_Video_Decompression_of_Talking_Heads_CVPR_2020_paper.pdf)

**PuppeteerGAN: Arbitrary Portrait Animation With Semantic-Aware Appearance Transformation.**<br>
*Zhuo Chen, Chaoyue Wang, Bo Yuan, Dacheng Tao.*<br>
*CVPR 2020.* [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_PuppeteerGAN_Arbitrary_Portrait_Animation_With_Semantic-Aware_Appearance_Transformation_CVPR_2020_paper.pdf)

**Speech-driven Facial Animation using Cascaded GANs for Learning of Motion and Texture.**<br>
*Dipanjan Das, Sandika Biswas, Sanjana Sinha, Brojeshwar Bhowmick.*<br>
*ECCV 2020.* [[PDF]]

**Audio- and Gaze-Driven Facial Animation of Codec Avatars.**<br>
*Alexander Richard, Colin Lea, Shugao Ma, Jurgen Gall, Fernando de la Torre, Yaser Sheikh.*<br>
*WACV 2021.* [[PDF]](https://openaccess.thecvf.com/content/WACV2021/papers/Richard_Audio-_and_Gaze-Driven_Facial_Animation_of_Codec_Avatars_WACV_2021_paper.pdf)

**One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing.**<br>
*Ting-Chun Wang, Arun Mallya, Ming-Yu Liu.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_One-Shot_Free-View_Neural_Talking-Head_Synthesis_for_Video_Conferencing_CVPR_2021_paper.pdf?) [[Project]](https://nvlabs.github.io/face-vid2vid/) [[Talk]](https://www.youtube.com/watch?v=smrcnZ5Eg4A)

**LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces From Video Using Pose and Lighting Normalization.**<br>
*Avisek Lahiri, Vivek Kwatra, Christian Frueh, John Lewis, Chris Bregler.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lahiri_LipSync3D_Data-Efficient_Learning_of_Personalized_3D_Talking_Faces_From_Video_CVPR_2021_paper.pdf)

**Flow-Guided One-Shot Talking Face Generation With a High-Resolution Audio-Visual Dataset.**<br>
*Zhimeng Zhang, Lincheng Li, Yu Ding, Changjie Fan.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf) [[Official PyTorch]](https://github.com/MRzzm/HDTF) [[HR-Dataset]](https://github.com/MRzzm/HDTF)

**Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation.**<br>
*Hang Zhou, Yasheng Sun, Wayne Wu, Chen Change Loy, Xiaogang Wang, Ziwei Liu.*<br>
*CVPR 2021.* [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Pose-Controllable_Talking_Face_Generation_by_Implicitly_Modularized_Audio-Visual_Representation_CVPR_2021_paper.pdf) [[Project]](https://hangz-nju-cuhk.github.io/projects/PC-AVS) [[Official PyTorch]](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)

**MeshTalk: 3D Face Animation From Speech Using Cross-Modality Disentanglement.**<br>
*Alexander Richard, Michael Zollhofer, Yandong Wen, Fernando de la Torre, Yaser Sheikh.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Richard_MeshTalk_3D_Face_Animation_From_Speech_Using_Cross-Modality_Disentanglement_ICCV_2021_paper.pdf) [[Code]](https://github.com/facebookresearch/meshtalk)

**AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis.**<br>
*Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, Juyong Zhang.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_AD-NeRF_Audio_Driven_Neural_Radiance_Fields_for_Talking_Head_Synthesis_ICCV_2021_paper.pdf) [[Official PyTorch]](https://github.com/YudongGuo/AD-NeRF)

**HeadGAN: One-Shot Neural Head Synthesis and Editing.**<br>
*Michail Christos Doukas, Stefanos Zafeiriou, Viktoriia Sharmanska.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Doukas_HeadGAN_One-Shot_Neural_Head_Synthesis_and_Editing_ICCV_2021_paper.pdf) [[Project]](https://michaildoukas.github.io/HeadGAN/)

**FACIAL: Synthesizing Dynamic Talking Face With Implicit Attribute Learning.**<br>
*Chenxu Zhang, Yifan Zhao, Yifei Huang, Ming Zeng, Saifeng Ni, Madhukar Budagavi, Xiaohu Guo.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_FACIAL_Synthesizing_Dynamic_Talking_Face_With_Implicit_Attribute_Learning_ICCV_2021_paper.pdf)

**Learned Spatial Representations for Few-Shot Talking-Head Synthesis.**<br>
*Moustafa Meshry, Saksham Suri, Larry S. Davis, Abhinav Shrivastava.*<br>
*ICCV 2021.* [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Meshry_Learned_Spatial_Representations_for_Few-Shot_Talking-Head_Synthesis_ICCV_2021_paper.pdf) 