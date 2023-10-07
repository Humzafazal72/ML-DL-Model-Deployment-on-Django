
from django.shortcuts import render
import xgboost as xgb
from django.http import HttpResponseRedirect
model = xgb.Booster(model_file='best_model.model')
def main(request):
    return render(request,'index.html')

def prediction(request):
    if request.method == "POST":
        hair = request.POST.get("hair", False) == "on"
        fore_width = float(request.POST.get("fore_width", 0))
        fore_height = float(request.POST.get("fore_height", 0))
        nose_w = request.POST.get("nose_w", False) == "on"
        nose_l = request.POST.get("nose_l", False) == "on"
        lips_t = request.POST.get("lips_t", False) == "on"
        dist_n_l = request.POST.get("dist_n_l", False) == "on"

        # Convert the input values to appropriate data types as required by your model
        # For example, if your model expects numerical features, convert them to float or int.
        input_data = [[int(hair), fore_width, fore_height, int(nose_w), int(nose_l), int(lips_t), int(dist_n_l)]]

        # Create a DMatrix object from the input data
        dmatrix = xgb.DMatrix(data=input_data)

        # Make predictions using the model
        y_predict_b = model.predict(dmatrix)
        
        y_predict_s = "Female" if y_predict_b[0] < 0.5 else "Male"

        # Render the template with the prediction result
        return render(request, 'result.html', {"prediction": y_predict_s,'confidence':y_predict_b[0]*100,'hair':int(hair), 'fore_width':fore_width, 'fore_height':fore_height,
                                               'nose_w':int(nose_w), 'nose_l':int(nose_l),'lips_t': int(lips_t), 
                                               'dist_n_l':int(dist_n_l)})
            
