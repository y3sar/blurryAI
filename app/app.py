from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import random
import gradio as gr
import numpy as np


def random_color_gen(n):
    return [tuple(random.randint(0,255) for i in range(3)) for i in range(n)]

import math

def get_gaussian_kernel(kernel_size=15, sigma=20, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, padding='same', groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


output_dict = {} # this dict is shared between segment and blur_background functions
pred_label_unq = []

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

def blur_background(input_image, label_name):
    mask = output_dict[label_name]['mask']
    mask = torch.tensor(mask).unsqueeze(0)
    
    input_tensor = T.ToTensor()(input_image).unsqueeze(0)
    blur = get_gaussian_kernel()
    blurred_tensor = blur(input_tensor)
    
    final_img = blurred_tensor
    final_img[:, :, mask.squeeze(0)] = input_tensor[:, :, mask.squeeze(0)];
    
    final_img = T.ToPILImage()(final_img.squeeze(0))
    
    return final_img;
    
    
    

    
    
with gr.Blocks() as app:
    
    gr.Markdown("# Blur an objects background with AI")
    
    gr.Markdown("First segment the image and create bounding boxes")
    with gr.Column():
        input_image = gr.Image(type='pil')
        b1 = gr.Button("Segment Image")
        
    
    
    with gr.Row():
#         masked_image = gr.Image();
        bounding_box_image = gr.Image();
    
    
    gr.Markdown("Now choose a label (eg: person1) from the above image of your desired object and input it below")
    with gr.Column():
        label_name = gr.Textbox()
        b2 = gr.Button("Blur Backbround")
        result = gr.Image()
    
    b1.click(segment, inputs=input_image, outputs=bounding_box_image)
    b2.click(blur_background, inputs=[input_image, label_name], outputs=result)
    
    
    
    
#     instance_segmentation = gr.Interface(segment, inputs=input_image, outputs=['json', 'image'])

app.launch(debug=True)