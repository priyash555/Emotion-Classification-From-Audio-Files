from django.contrib import admin
from django.urls import path, include

from home import views
from .views import starting, about

urlpatterns = [
    path('', starting, name='home-home'),
    path('about/', starting, name='home-about'),
]