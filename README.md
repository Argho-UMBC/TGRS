## SAM-VQA: Supervised Attention-Based Visual Question Answering Model for Post-Disaster Damage Assessment on Remote Sensing Imagery

# Abstract
Each natural disaster leaves a trail of destruction and damage that must be effectively managed to reduce its negative impact on human life. Any delay in making proper decisions at the post-disaster managerial level can increase human suffering and waste resources. Proper managerial decisions after any natural disaster rely on an appropriate assessment of damages using data-driven approaches, which are needed to be efficient, fast, and interactive. The goal of this study is to incorporate a deep interactive data-driven framework for proper damage assessment to speed up the response and recovery phases after a natural disaster. Hence, this article focuses on introducing and implementing the visual question answering (VQA) framework for post-disaster damage assessment based on drone imagery, namely supervised attention-based VQA (SAM-VQA). In VQA, query-based answers from images regarding the situation in disaster-affected areas can provide valuable information for decision-making. Unlike other computer vision tasks, VQA is more interactive and allows one to get instant and effective scene information by asking questions in natural language from images. In this work, we present a VQA dataset and propose a novel SAM-VQA framework for post-disaster damage assessment on remote sensing images. Our model outperforms state-of-the-art attention-based VQA techniques, including stacked attention networks (SANs) and multimodal factorized bilinear (MFB) with Co-Attention. Furthermore, our proposed model can derive appropriate visual attention based on questions to predict answers, making our approach trustworthy.



![SAM-VQA](fig1 (1).png)


# Motivation
Disaster management can be defined as an accountable organization and management for dealing with all humanitarian aspects, particularly post-disaster response, and recovery, to mitigate the impact of a disaster. In the response and recovery stage after any catastrophic event, disaster management requires a fast and interactive data-driven approach to thoroughly comprehend the damaged situation. A rapid and in-depth understanding of the damage in the aftermath of disasters is essential for supporting the decision-making system. The decisions regarding the distribution of relief and food to the highly victimized areas, the operation of the search and rescue missions, the reconstruction of the damaged roads and buildings, etc., are dependent on the proper assessment of the damage. Any delay in the recovery phase can drive human lives toward death and dissipate an abundance of money. Haas et al. established a logarithmic heuristic which suggests that reducing the time spent on each phase of a disaster response reduces the time spent on the next phase by a factor of 10. In this article, we present a supervised attention-based visual question answering (SAM-VQA) framework to provide high-level scene information for proper damage assessment to speed up the response and recovery phases after any natural disaster.

# VQA for remote-sensing images is more Challenging than VQA for human-centric images

Visual question answering (VQA) is a complicated multimodal research problem in which the aim is to answer an image-specified question. In a VQA framework, we generally ask questions about images in natural language. Attention-based VQA models showed remarkable performance on many ground imagery-based VQA datasets. Attention in VQA algorithms is defined as assigning weights within different image regions according to the importance of getting clues for predicting the answer to a given question. Relevant image portions should get higher weights compared to irrelevant portions to answer a question. Although those attention-based VQA frameworks can obtain relevant visual attention weight from many ground imagery-based VQA datasets, they fail to obtain relevant visual attention from remote sensing images. The main reason for not obtaining relevant visual attention weights on remote sensing images is the way those models are learning visual attention weights. Most of the attention-based VQA models are trained in a supervised manner (i.e., minimizing the cross-entropy loss between the ground truth and predicted answers). However, visual attention weights within those models are learned without any additional supervision and solely based on ground-truth answers. By ground-truth answer, we mean the corresponding true text answer to a given question about an image. The estimated visual attention weight distributions for remote sensing images learned solely by minimizing loss between the ground-truth answers and predicted answers in a classification manner could not properly highlight the relevant image regions. An additional learning component as a means of supervision is needed so that the estimated visual attention weight can focus on relevant image portions to answer a question. Thus, to supervise the visual attention weight, we need the visual ground-truth which will highlight the relevant image portions necessary for answering a given question. To address this, we propose a SAM-VQA framework to obtain relevant visual attention weights on remote sensing images in the context of post-disaster damage assessment.


# Model Architecture

![SAM-VQA2](archi.png)

## Results

# Quantitative


# Qualitative

![SAM-VQA3](fig2 (1).png)

