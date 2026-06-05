## Introduction

Breast cancer is the most diagnosed cancer and one of the leading cancer-related deaths among women. Through a process called digital pathology scanning, whole Slide Images (WSIs) are acquired recording potentially cancerous tissue at very high resolution. Skilled physicians are able to make a diagnosis based on examining these images. By leveraging deep learning techniques, this process can be assisted or even automated, making diagnosis faster, more cost-effective, and more consistent while helping physicians focus their expertise on the most challenging cases.

R. Ge et al. propose the Multiscale Multi-head Self-attention Ensemble Network (MMSEN). It is a heterogeneous deep ensemble learning approach. Ensemble techniques show good generalization capabilities. The intermediate feature vectors produced by VGG16 and DenseNet121, pretrained on the ImageNet1k dataset, are combined using a self-attention layer, followed by global average pooling.

Due to a scarcity of data the authors of the original paper use a five-fold cross validation, to train and assess the models performance. Here we simply split the data into disjoint train, test and evaluation sets. This may be the reason for the large deviation in the results reported by R. Ge et al. and the results we show here.

We train the model on the PCam benchmark dataset, while the model achieves high precision and specifity, the sensitivity is very low and even degrades further during training. This is due to an unsettling high number of false negatives. That is patients for whom the cancer would have been remained undetected. In order to mend this problem we experiment with different data augmentation and regularization techniques, to improve sensitivity.

## Results

For a detailed description of the approach and results please visit the following blog:
https://scientific-blog.pages.dev/blog/cancer-detection/