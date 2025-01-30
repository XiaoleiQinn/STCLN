# Spatiotemporal masked pre-training for advancing crop mapping on satellite image time series with limited labels



![Flowchart](https://github.com/user-attachments/assets/7bb996a7-7602-432f-be50-44f9116d33be)
Fig. 1. Overview of the proposed pre-training and fine-tuning processes for crop mapping.

![STCLN](https://github.com/user-attachments/assets/49e9065c-d8ac-4d8d-b2ca-499e591277ad)
Fig. 2. The architecture of the proposed SpatioTemporal Collaborative Learning Network (STCLN).




## Data preparation
Please download the PASTIS dataset from https://zenodo.org/record/5012942. Please kindly use the data according to the instruction from https://github.com/VSainteuf/pastis-benchmark.
Please download the MTLCC dataset from https://zenodo.org/records/5712933#.YZdpwXVKhaY for MTLCC dataset. Please kindly use the data according to https://github.com/MarcCoru/MTLCC.
## Implementation
Please use finetuning_STCLN.py in MTLCC or PASTIS folder.

You can replace "--pretrain_pth" with the folder where you have saved our pre-training weight file, either "checkpoint_99.utae.tar" or "checkpoint_99.sttrans.tar."

## Extension experiments on different temporal lengths and resolutions
We initialize the model pre-trained on data from 30 images in 2016 and fine-tune it using the MTLCC dataset collected in 2017, which consists of fewer images with a temporal length of 12. As shown in Table 1, compared to the model without pre-training (STCLN), the pre-trained model (STCLN_wp) enhances performance, demonstrating that the self-supervised method can be applied to temporal images with shorter length and lower temporal resolution. 

Table 1 Experiment of fine-tuning on MTLCC 2017 data with a temporal length of 12.
| Method      | OA |mIoU      | mF1 |
| ----------- | ----------- |----------- | ----------- |
| STCLN      | 0.7674       |0.3896|0.4881
| STCLN_wp   | 0.8018	        |0.4870|0.5914


## Extension experiments on early-season and cross-year crop mapping
We use the model weights pre-trained using data in 2016 to initialize the model and fine-tune it using the MTLCC dataset collected from February to October 2017. As shown in Fig. 3, as the image length increases, the performance improves. Compared to the model without pre-training (STCLN), the pre-trained model (STCLN_wp) consistently enhances the performance. This result demonstrates that the pre-trained method can support cross-year transfer crop mapping tasks. 


![image](https://github.com/user-attachments/assets/8a4574ff-a1ea-49a7-8c87-85b9ea5704e1)

Fig. 3. Cross-year experiment on MTLCC dataset.

## Todo
1. Increasing model'size
2. Pretrain with larger-scale data
3. Extend this method to multi-source (SAR and optical) SITS data for better feature representation.
4. Integrate this method with some sample generation methods, such as active learning, fuzzy clustering, and Dynamic Time Warping, etc.
