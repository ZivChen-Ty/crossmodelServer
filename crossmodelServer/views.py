import argparse

from PIL import Image
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
import torch
import os
import numpy as np
import random
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import json
from crossmodelServer.HTMCModel.util.input_file import read, get_token
from crossmodelServer.HTMCModel.util.load_model import load_harnn, get_clip_model, get_harnn_model


def index(request):
    return HttpResponse('Hello World!')

def encodeImage(imagePath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = imagePath.split('.')
    clip_model, preprocess = get_clip_model()
    if tmp[-1] in ["jpg", "jpeg", "png"]:
        path = imagePath[1:]
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        global_feature = clip_model.encode_image(image)
        global_feature /= global_feature.norm(dim=-1, keepdim=True)
        global_feature = global_feature.repeat(53, 1)
        return global_feature
    else:
        return False

def encodeText(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = get_clip_model()
    text_data = text
    clip_token = clip.tokenize(text_data).to(device)
    features, global_feature = clip_model.encode_text(clip_token)
    global_feature /= global_feature.norm(dim=-1, keepdim=True)
    return global_feature



def test(request):
    path = 'F:\lab\项目实训-多级分类标签\代码\CrossModel\data\origin_data\畜牧兽业\\'
    doc_name = '畜牧兽业-产蛋率-不同光照强度对层叠式笼养鸡舍鸡群产蛋率的影响.pdf'
    text, image = read(path, doc_name)
    print(text, image)
    if len(image) > 0 and image[0] == 'is Image':
        features = encodeImage(text[0])
    else:
        token = get_token(text)
        print(token, image)
        features = encodeText([doc_name] + token[0])
        print(features.shape)

    harnn_model = get_harnn_model()
    print(features.float().shape)
    print(harnn_model(features.float()))
    # js = json.dumps({
    #     'feature': features.cpu()
    # })
    return HttpResponse('test')
