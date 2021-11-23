import asyncio
from natsort import natsorted
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import pickle
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import HairColorType

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('attribute', choices=['gt', 'blonde', 'yaw', 'pitch'], type=str)
args = parser.parse_args()
attribute = args.attribute


def get_accessories(accessories):
    """Helper function for face_detection sample.
    This will return a string representation of a person's accessories.
    """

    accessory_str = ",".join([str(accessory) for accessory in accessories])
    return accessory_str if accessory_str else "No accessories"


def get_emotion(emotion):
    """Helper function for face_detection sample.
    This will determine and return the emotion a person is showing.
    """

    emotions = {}
    for emotion_name, emotion_value in vars(emotion).items():
        if emotion_name == "additional_properties":
            continue
        emotions[emotion_name] = emotion_value
    return emotions


def get_hair(hair):
    """Helper function for face_detection sample.
     This determines and returns the hair color detected for a face in an image.
    """

    if not hair.hair_color:
        return "invisible" if hair.invisible else "bald"

    hairs = {}
    for hair_color in hair.hair_color:
        hairs[hair_color.color.split('\'')[0]] = hair_color.confidence

    return hairs


# TODO: you need to modify these two values
KEY = 'your_key_here'

ENDPOINT = "https://your_endpoint.cognitiveservices.azure.com/"

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

if attribute == 'gt':
    imgs = natsorted(glob.glob('../fake/original/*.jpg'))[:100]
else:
    imgs = natsorted(glob.glob(f'../fake/{attribute}/*.jpg'))[:100]

print(f'num to do: {len(imgs)}')
print(f'----------------------------- doing attribute: {attribute}')

results = {}
ids_done = []

start = time.time()
missing = []

for i, img in enumerate(imgs):
    image = open(img, 'r+b')
    img = img.split('/')[-1].split('.')[0]

    # Pausing for a little bit to avoid triggering any rate limiting
    time.sleep(3)

    faces = face_client.face.detect_with_stream(image, return_face_attributes=['age', 'gender', 'headPose', 'smile', 'facialHair', 'glasses', 'emotion', 'hair', 'makeup', 'occlusion', 'accessories', 'blur', 'exposure', 'noise'])

    if len(faces):
        attributes = {
            'age': faces[0].face_attributes.age,
            'gender': faces[0].face_attributes.gender,

            'pitch': faces[0].face_attributes.head_pose.pitch,
            'yaw': faces[0].face_attributes.head_pose.yaw,
            'roll': faces[0].face_attributes.head_pose.roll,

            'smile': faces[0].face_attributes.smile,

            'moustache': faces[0].face_attributes.facial_hair.moustache,
            'beard': faces[0].face_attributes.facial_hair.beard,
            'sideburns': faces[0].face_attributes.facial_hair.sideburns,

            'glasses': faces[0].face_attributes.glasses,

            'hair_color': get_hair(faces[0].face_attributes.hair),
            'emotion': get_emotion(faces[0].face_attributes.emotion),
            'accessories': get_accessories(faces[0].face_attributes.accessories),

            'blur': faces[0].face_attributes.blur.value,
            'exposure': faces[0].face_attributes.exposure.value,
            'noise': faces[0].face_attributes.noise.value,

            'eye_makeup': faces[0].face_attributes.makeup.eye_makeup,
            'lip_makeup': faces[0].face_attributes.makeup.lip_makeup,
        }

        results[img] = attributes
        ids_done += [img]
    else:
        print(f'no face has been detected in {img}')
        missing += [img]
    print(f'done {i}/{len(imgs)}')

pickle.dump(results, open(f'./predictions-new/preds-{attribute}.pkl', 'wb'))

print('time taken in seconds', time.time() - start)