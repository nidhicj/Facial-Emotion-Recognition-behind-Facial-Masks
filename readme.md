# Facial Emotion Recognition behind Facial Masks

InfoTech Research Project

## Abstract
The onset of a pandemic condition owing to Covid-19 has introduced new obstacles in Facial Emotion Recognition (FER), particularly due to the requirement to wear facial masks. In light of this issue, the goal of this study is to train the Attention-based Convolutional Neural Network (ACNN) developed by Yong Li et al.[[1]](#1) and train the model on artificially synthesised masked datasets derived from existing emotion datasets namely AffectNet, RAF-DB and CK+. A hypothesis is proposed with the notion that emotions namely Neutral, Happy, Angry and Surprise are better perceived than Sad, Fear and Disgust emotions. The experiments and results acquired following a successful training and testing regime of the model, indicate that the presented hypothesis is correct. Although the test accuracies obtained in the study do have further scope to be improved. Many difficulties arise as a result of information loss caused by face masks, which may be solved by creating additional datasets collected under controlled and lab circumstances.

## Introduction
To implement an algorithm that can recognize the facial emotion behind the facial mask, this project presents a model that is trained on images of masked emotional faces. A Convolution Neural Network with Attention Mechanism (ACNN) [[1]](#1) is used, which mimics how humans recognize facial expressions. It concentrates and learns on the unobstructed and obstructed portions of the face using two versions of ACNN: patch based ACNN and global-local based ACNN respectively and classifies the emotion according to the standard emotions accepted in FER studies i.e., Neutral, Happy, Angry, Surprise, Sad, Fear and Disgust. Apart from these standard emotions, the model is also tested on only four emotions: Neutral, Happy, Angry and Surprise which are hypothesised to be perceived better than Sad, Fear and Disgust [[2]](#2). The emotion images are obtained from different datasets namely: AffectNet [[3]](#3), CK+ [[4]](#4) and RAF-DB[[5]](#5), that are augmented by artificially overlaying facial mask on the subject in the image. Subsequently, the results of the model’s performance for the experiments conducted to evaluate the model are presented.

A detailed discussion on results and conclusion is documented in the file `Report.pdf` 

![[Emotions]](misc/unmasked.gif) --> ![[Emotions behind Facial Mask]](misc/masked.gif)

## Repository Usage Details

This repository contains the code needed to pre-process the datasets, extract features from it, train, validate and test the model. 
To recreate the project, please follow the steps below after cloning the repo:
1. Use the package manager Anaconda to install all the dependencies from `requirements.txt`.
```python
conda create --name <env> --file requirements.txt
```
2. To artificially overlay the masks on the images of selected dataset, use the `pre_processing/facial_mask_overlaying.ipynb` file.
3. In `pre_processing/` folder, run the `IP_<dataset_name>.ipynb` files to generate train, validation and test sets in .txt files. It also aggregates all the facial landmarks of the images and dumps into .pkl file stored in respective folder (Results/MaskedDatasets or Results/UnmaskedDatasets-> AffectNet/CKplus/RAF-DB)
4. Once the .pkl and .txt files are ready, the scripts from `Scripts.txt` file must be used to train the model for required dataset and number of emotions (4/7)
5. The results obtained from the `main_trainer.py` file are stored in the `<NumberOfEmotions><ConditionOfImages>output.txt` file and also .png files of the confusion matrix are generated.
*NumberOfEmotions - 4/7
*ConditionOfImages - Masked(M) / Unmasked(U)

Note: 
1. The folders checkpoint_<dataset_name> contain the weights of a best trained model.
2. The datasets utilized for this research were given proper acknowledgment and relevant rights were granted.

## References
[1] Y. Li, J. Zeng, S. Shan, X. Chen. “Occlusion Aware Facial Expression Recognition Using CNN With Attention Mechanism.” In: IEEE Transactions on Image Processing 28.5 (2019), pp. 2439–2450. ISSN: 10577149. DOI: 10.1109/TIP.2018.2886767 (cit. on pp. 18, 19, 21).

[2] M. Marini, A. Ansani, F. Paglieri, F. Caruana, M. Viola. “The impact of facemasks on emotion recognition, trust attribution and re-identification.” In: Scientific Reports 11.1 (2021), pp. 1–14. ISSN:20452322. DOI: 10.1038/s41598-021-84806-5. URL: https://doi.org/10.1038/s41598-021-84806-5 (cit. on p. 14)

[3] A. Mollahosseini, B. Hasani, M. H. Mahoor. “AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild.” In: IEEE Transactions on Affective Computing PP.99 (2017), pp. 1–1 (cit. on p. 22).

[4] P. Lucey, J. F. Cohn, T. Kanade, J. Saragih, Z. Ambadar, I. Matthews. “The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression.” In: 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops. 2010, pp. 94–101. DOI: 10.1109/CVPRW.2010.5543262 (cit. on
pp. 22, 25).

[5] S. Li, W. Deng, J. Du. “Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild.” In: Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE. 2017, pp. 2584–2593 (cit. on p. 22).

