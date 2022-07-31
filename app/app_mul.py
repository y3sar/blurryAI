from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import random
import gradio as gr
import numpy as np

output_dict = {} # this dict is shared between segment and blur_background functions
pred_label_unq = []


def random_color_gen(n):
    return [tuple(random.randint(0,255) for i in range(3)) for i in range(n)]

def segment(input_image):
    
    # prepare image for display
    display_img = torch.tensor(np.asarray(input_image)).unsqueeze(0)
    display_img =  display_img.permute(0, 3, 1, 2).squeeze(0)
    
    # Prepare the RCNN model
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    transforms = weights.transforms()
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model = model.eval();
    
    # Prepare the input image
    input_tensor = transforms(input_image).unsqueeze(0)
    
    # Get the predictions
    output = model(input_tensor)[0] # idx 0 to get the first dictionary of the returned list
    
    
    # Filter by threshold
    score_threshold = 0.75
    mask_threshold = 0.5
    masks = output['masks'][output['scores'] > score_threshold] > mask_threshold;
    boxes = output['boxes'][output['scores'] > score_threshold]
    masks = masks.squeeze(1)
    boxes = boxes.squeeze(1)
    
    pred_labels = [weights.meta["categories"][label] for label in output['labels'][output['scores'] > score_threshold]]
    n_pred = len(pred_labels)
    
    # give unique id to all the predicitons
    pred_label_unq = [pred_labels[i] + str(pred_labels[:i].count(pred_labels[i]) + 1) for i in range(n_pred)]
    
    colors = random_color_gen(n_pred)
    
    # Prepare output_dict
    for i in range(n_pred):
        output_dict[pred_label_unq[i]] = {'mask': masks[i].tolist(), 'color': colors[i]}
        
    
    masked_img = draw_segmentation_masks(display_img, masks, alpha=0.9, colors=colors)
    bounding_box_img = draw_bounding_boxes(masked_img, boxes, labels=pred_label_unq, colors='white')
    masked_img = T.ToPILImage()(masked_img)
    bounding_box_img = T.ToPILImage()(bounding_box_img)
    
    return bounding_box_img;


def blur_object(input_image, label_name):

    label_names = label_name.split(' ')

    
    
    input_tensor = T.ToTensor()(input_image).unsqueeze(0)
    blur = T.GaussianBlur(15, 20)
    blurred_tensor = blur(input_tensor)

    final_img = input_tensor


    for name in label_names:
        mask = output_dict[name.strip()]['mask']
        mask = torch.tensor(mask).unsqueeze(0)
    
        final_img[:, :, mask.squeeze(0)] = blurred_tensor[:, :, mask.squeeze(0)];
    
    final_img = T.ToPILImage()(final_img.squeeze(0))
    
    return final_img;

def blur_background(input_image, label_name):
    label_names = label_name.split(' ')

    
    
    input_tensor = T.ToTensor()(input_image).unsqueeze(0)
    blur = T.GaussianBlur(15, 20)
    blurred_tensor = blur(input_tensor)

    final_img = blurred_tensor


    for name in label_names:
        mask = output_dict[name.strip()]['mask']
        mask = torch.tensor(mask).unsqueeze(0)
    
        final_img[:, :, mask.squeeze(0)] = input_tensor[:, :, mask.squeeze(0)];
    
    final_img = T.ToPILImage()(final_img.squeeze(0))
    
    return final_img;
    
    
    

############################
""" User Interface """
############################

with gr.Blocks() as app:
    
    gr.Markdown("# Blur an objects background with AI")
    
    gr.Markdown("First segment the image and create bounding boxes")
    with gr.Column():
        input_image = gr.Image(type='pil')
        b1 = gr.Button("Segment Image")
        
    
    
    with gr.Row():
        bounding_box_image = gr.Image();
    
    
    gr.Markdown("Now choose a label (eg: person1) from the above image of your desired object and input it below")
    with gr.Column():
        label_name = gr.Textbox()
        with gr.Row():
            b2 = gr.Button("Blur Backbround")
            b3 = gr.Button("Blur Object")
        result = gr.Image()
    
    b1.click(segment, inputs=input_image, outputs=bounding_box_image)
    b2.click(blur_background, inputs=[input_image, label_name], outputs=result)
    b3.click(blur_object, inputs=[input_image, label_name], outputs=result)
    

app.launch(debug=True)
