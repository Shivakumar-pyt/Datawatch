from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate, login, logout

from mediator import QueryAnalyzer
from model import Model
from connect import connect

import numpy as np

from preprocess_query import PreprocessQuery

from anomaly_detector.models import User
from anomaly_detector.models import Warning

from datetime import date


clf = Model("output.csv").train_naivebayes()

@api_view(['POST'])
def loginAuthenticate(request):
    username = request.data.get('username')
    password = request.data.get('password')

    user = User.objects.get(username=username,password=password)

    if(user):
        return Response(user.role_name)
    else:
        return Response("Incorrect Credentials")

# @api_view(['POST'])
# def logoutUser(request):
#     logout(request)
#     return Response({'message': 'Logout successful'}, status=status.HTTP_200_OK)


@api_view(['POST'])
def processQuery(request):
    query = request.data.get('query')
    role = request.data.get('role')
    user_id = request.data.get('user_id')

    preprocess_query = PreprocessQuery(query,role)

    if(preprocess_query.check_access()==False):
        return Response("Unauthorized Query")

    conn = connect()
    query_analyzer = QueryAnalyzer(query, conn, role)

    command_type, role_score, sensitivity, avg_count, freq_score = query_analyzer.get_query_info()

    query_code=0

    if command_type == "select":
        query_code = 1
    else:
        query_code = 2
    
    q = np.array([query_code,role_score, sensitivity, avg_count, 12.418243725098195]).reshape(1,-1)

    model = Model("output.csv")

    prediction = model.run_naivebayes(clf,q)

    person = User.objects.get(user_id=user_id)

    print(user_id)

    if prediction==1:
        warning = Warning(warning_id=1001, user_id=user_id,query=query,timestamp=date.today(),status="Possible Anomaly")
        warning.save()


    if prediction==0:
        warning = Warning(warning_id=1001,user_id=user_id,query=query,timestamp=date.today(),status="Anomaly")
        warning.save()


    return Response(prediction)



    
    



