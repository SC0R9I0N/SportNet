# SportNet


## Data Source

The data is collected from [AthletePose3D](https://drive.google.com/drive/folders/10YnMJAluiscnLkrdiluIeehNetdry5Ft), specifically the pose_2d data.

## Other Resources

The model_backbone.py is based on [AthletePose3D's code](https://github.com/calvinyeungck/AthletePose3D/tree/main/pose_2d/moganet_b_ap2d_384x288.py), [MogaNet's moganet.py](https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py), and [MMPose's heatmap_head.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/heatmap_heads/heatmap_head.py).

## Our Data

You can download the AthletePose3D checkpoint we used, the Ground Truth JSON files for the train_set and valid_set, and the full trained model from [Google Drive](https://drive.google.com/drive/folders/1jUOJXuShJC8f-el2_GEzSSpOngEj7kvl?usp=drive_link). (Note: The trained model folder has an inner folder where it is semantic_explainer_t5/semantic_explainer_t5/, so you need to move the inner folder out)