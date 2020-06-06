from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.views.generic import ListView, CreateView, UpdateView, DetailView, DeleteView
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.template.loader import render_to_string
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse


# Create your views here.
def starting(request):
    return render(request, 'home/starting.html', {})


def about(request):
    return render(request, 'home/about.html', {})