from django.urls import path
from . import views
urlpatterns = [
    path('',views.vid_main,name='vidmain'),
    path('action_predict/',views.vid_predict,name='vidpredict')
]
