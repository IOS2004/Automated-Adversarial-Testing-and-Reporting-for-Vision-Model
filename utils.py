
# coding: utf-8

# In[2]:


import requests
from io import BytesIO
import urllib.request as url_req
from PIL import Image
import os
import torch
import torch.nn.functional as F


# In[3]:


def urltoImg(url):
    print(url)
    try:
        img = Image.open(url_req.urlopen(url))
        return img
    except Exception as error:
        print("Couldn't load image "+str(error))
        return None 


# In[1]:


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)

def torch_arctanh(x, eps=1e-6):
    x = x*(1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

def save_imgs(key,url_list):
    for i,url in enumerate(url_list):
        img = urltoImg(url)
        if img is not None:
            save_path = os.path.join('imagenet_imgs',str(key)+str('_')+str(i)+str('.png'))
            img.save(save_path)

def predict_top_five(model,img,k=5):

    output = model(img)
    # print(output.size())
    op_probs = F.softmax(output,dim=1)
    top_k = torch.topk(output,k,dim=1)
    labels = top_k[1].squeeze_(0)
    labels_np = labels.cpu().numpy()
    # print(labels)
    op_probs_np = op_probs.squeeze_(0).detach().cpu().numpy()*100
    # print('Probs')
    # print(op_probs_np[labels_np])



    return op_probs_np[labels_np],labels_np

def getPredictionInfo(model,img):
    output = model(img)
    _,pred = torch.max(output.data,1)
    #     print(adv_img.data-img.data)
    op_probs = F.softmax(output, dim=1)                 #get probability distribution over classes
    pred_prob =  ((torch.max(op_probs.data, 1)[0][0]) * 100, 4)      #find probability (confidence) of a predicted class
    return output,pred,op_probs,pred_prob

def checkMatchingLabels(label,pred_label,misclassfns):
    if (int(label)!=int(pred_label)):
        misclassfns+=1
    return misclassfns

def checkMatchingLabelsTop_five(label,pred_label_list,misclassfns):
    if(int(label) not in pred_label_list.astype(int)):
        misclassfns+=1
    return misclassfns

def save_adversarial_image(adv_img_tensor, original_img_name, attack_name, epsilon, output_dir='adversarial_images'):
    """
    Save adversarial image tensor to disk as PNG file.
    
    Args:
        adv_img_tensor: Adversarial image tensor (can be batched or single image)
        original_img_name: Name of the original image file
        attack_name: Name of the attack method (e.g., 'fgsm', 'bim', 'pgd')
        epsilon: Epsilon value used for the attack
        output_dir: Directory to save adversarial images
    """
    import numpy as np
    import torchvision.transforms as transforms
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Remove batch dimension if present
    if adv_img_tensor.dim() == 4:
        adv_img_tensor = adv_img_tensor.squeeze(0)
    
    # Denormalize the image (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Move to CPU and detach
    adv_img_tensor = adv_img_tensor.cpu().detach()
    
    # Denormalize
    adv_img_denorm = adv_img_tensor * std + mean
    
    # Clip to valid range [0, 1]
    adv_img_denorm = torch.clamp(adv_img_denorm, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    adv_img_pil = to_pil(adv_img_denorm)
    
    # Create filename
    base_name = os.path.splitext(original_img_name)[0]
    filename = f"{base_name}_{attack_name}_eps{epsilon}.png"
    save_path = os.path.join(output_dir, filename)
    
    # Save image
    adv_img_pil.save(save_path)
    
    return save_path