from django.urls import path
from . import views

urlpatterns = [
    path('',views.img_main,name='imgmain'),
    path('image_prediction',views.impredictor,name='impredictor' )
]
