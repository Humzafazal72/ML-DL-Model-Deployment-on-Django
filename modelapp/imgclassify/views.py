from django.shortcuts import render
from keras.models import load_model
import numpy as np
from PIL import Image


model=load_model('imgclassify.h5')

def img_main(request):
    return render(request,'imgclassify.html')

def impredictor(request):
    if request.method=="POST":
        uploaded_image = request.FILES.get("img")

        pil_image = Image.open(uploaded_image)

        pil_image=pil_image.convert("RGB")

        resized_image = pil_image.resize((100, 100))

        image_array = np.array(resized_image)

        image_array = image_array / 255.0

        input_array = image_array.reshape(1, 100, 100, 3) 

        prediction = model.predict(input_array)
        if prediction[0,0]>0.5:
            prediction='Cat'
        else:
            prediction='Dog'
        
        return render(request,'img_result.html',{'prediction':prediction})