from django.urls import path
from . import views

urlpatterns = [
    path('',views.main,name='home'),
    path('predictor/',views.prediction,name='predictor')
]
