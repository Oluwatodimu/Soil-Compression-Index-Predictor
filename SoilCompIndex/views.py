from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle

# homepage function
def home(request):
    return render(request, 'home.html') 

# prediction function
def result(request):
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    userInupts = []

    userInupts.append(request.GET['mc'])
    userInupts.append(request.GET['ll'])
    userInupts.append(request.GET['pi'])
    userInupts.append(request.GET['ivr'])
    userInupts.append(request.GET['sg'])

    ans = model.predict([userInupts])

    return render(request, 'result.html', {'ans': ans})

# API testing function module
@csrf_exempt
def predictApi(request):

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    userInupts = []

    userInupts.append(request.GET['mc'])
    userInupts.append(request.GET['ll'])
    userInupts.append(request.GET['pi'])
    userInupts.append(request.GET['ivr'])
    userInupts.append(request.GET['sg'])

    ans = model.predict([userInupts])

    ansDict = {'Compression Index': round(ans, 2)}

    return JsonResponse(ansDict)
