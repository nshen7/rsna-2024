import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ----------- Functions for disc detection -----------

def load_model_disc_detection(state_dict=None):
    
    # We use the lastest
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Replace the classifier with a new one, that has Num_classes
    num_classes = 6  # 5 classes (discs) + background

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Import parameters if available
    if state_dict:
        model.load_state_dict(state_dict)
    
    return model


# ----------- Functions for severity classification -----------

def load_model_severity_classification(state_dict=None):
    
    # We use the lastest
    model = torchvision.models.swin_v2_t(weights="DEFAULT")

    # Replace the classifier with a new one, that has Num_classes
    num_classes = 3 

    # Get number of input features for the classifier
    in_features = model.head.in_features

    # Replace the pre-trained head with a new one
    model.head = nn.Linear(in_features, num_classes)
    
    # Import parameters if available
    if state_dict:
        model.load_state_dict(state_dict)
    
    return model