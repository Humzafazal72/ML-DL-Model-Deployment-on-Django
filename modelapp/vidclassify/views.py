from django.shortcuts import render
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

i3d_model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

# Create your views here.
def vid_main(request):
    return render(request,'vid_classify.html')

def preprocess_video_clip(video_clip, output_size=(224, 224)):
    resized_frames = [cv2.resize(frame, output_size) for frame in video_clip]
    normalized_frames = np.array(resized_frames) / 255.0
    return normalized_frames

with open("label_map.txt", "r") as file:
    class_labels = file.read().splitlines()


def vid_predict(request):
    if request.method == "POST":
        uploaded_vid = request.FILES.get("vid2")
        video_content = uploaded_vid.read()
        with open('uploaded.mp4', 'wb') as file:
            file.write(video_content)
        
        video_path = "uploaded.mp4"
        clip_length = 16
        num_classes = 400  # The number of classes in the Kinetics dataset

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_size=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        frame_indices = np.linspace(0, len(frames) - 1, clip_length, dtype=int)
        video_clip = [frames[i] for i in frame_indices]

        processed_clip = preprocess_video_clip(video_clip)
        processed_clip = np.expand_dims(processed_clip, axis=0)
        processed_clip = tf.constant(processed_clip, dtype=tf.float32)

        predictions = i3d_model(processed_clip)['default']

        prediction_np = np.array(predictions)
        predicted_class = np.argmax(predictions)

        predicted_action = class_labels[predicted_class]
        
        return render(request,'vid_classify.html',{'prediction':predicted_action})
        



