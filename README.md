# dr-segmentation-python
SSMDDeeplabv3plus-based Segmentation model for detecting diabetic retinopathy using unsupervised learning.
**(Semi-Supervised Multi-task Decoders Deeplabv3plus)**

## 모델 설명
1. Trained the encoder of Deeplabv3plus using the EypPACS dataset.
2. One decoder is used for reconstruction using the FGADR dataset, while the other four decoders perform segmentation for Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates, respectively.
   
## Results
1. Table

|||MA||HE||EX||SE|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
||Dice|ROC|Dice|ROC|Dice|ROC|Dice|ROC|
|DL_v3+|0.482|0.934|0.55|0.973|0.598|0.977|0.608|0.967|
|SSMD_DL_v3+|0.566|0.951|0.665|0.981|0.663|0.988|0.729|0.984|

2. Inference Images
![ssmd_inference](https://github.com/Lucia-Jaekyung/dr-segmentation-pytorch/assets/141312157/7825af06-3cb4-4d2a-a689-fe73a3d76384)
