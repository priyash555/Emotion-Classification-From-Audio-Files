from django.contrib import admin
from django.urls import path, include

from home import views
from .views import starting, about, FileView, FileDeleteView, Predict

urlpatterns = [
    path('', starting, name='home-home'),
    path('about/', starting, name='home-about'),
    path('predict/', Predict.as_view(), name='APIpredict'),
    path('upload/', FileView.as_view(), name='APIupload'),
    path('delete/', FileDeleteView.as_view(), name='APIdelete'),
]