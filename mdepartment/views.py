from django.shortcuts import render
from .models import Mdepartments
from django.contrib import messages, auth
import os
from django.http import JsonResponse
from sklearn.externals import joblib


def index(request):
    mdepartments=Mdepartments.objects.all()
    context={
        'mdepartments':mdepartments
    }
