from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate, login, logout
import pickle
from mediator import QueryAnalyzer
from model import Model
from connect import connect
import numpy as np
from preprocess_query import PreprocessQuery
from .models import Users, Warnings
from datetime import date
from django.views.decorators.csrf import csrf_exempt

from .serializer import WarningSerializer, UserSerializer
from connect import connect

from load_schema import LoadSchema



@api_view(['POST'])
def loginAuthenticate(request):
    username = request.data.get('username')
    password = request.data.get('password')

    try:
        user = Users.objects.get(username=username,password=password)
    except:
        return Response("Incorrect Credentials")

    user_serializer = UserSerializer(user,many=False)

    return Response(user_serializer.data)
        

# @api_view(['POST'])
# def logoutUser(request):
#     logout(request)
#     return Response({'message': 'Logout successful'}, status=status.HTTP_200_OK)


@api_view(['POST'])
def processQuery(request):
    query = request.data.get('query').lower()
    role = request.data.get('role')
    user_id = request.data.get('user_id')

    # print(query,role,user_id)

    preprocess_query = PreprocessQuery(query,role)

    # print("syntax check: ",preprocess_query.check_access())

    if((preprocess_query.check_access()==0) or (preprocess_query.check_access()==-1)):
        return Response("There is an error in your SQL syntax. Please verify your SQL query.")
    
    if(preprocess_query.check_access()==2):
        return Response("Unauthorized Query")

    conn = connect()
    query_analyzer = QueryAnalyzer(query, conn, role)

    command_type, role_score, sensitivity, avg_count, freq_score = query_analyzer.get_query_info()

    print(command_type, role_score, sensitivity, avg_count, freq_score)

    if avg_count >= 0.4 and command_type == "delete":
        label = "Possible Anomaly"
        if avg_count >= 0.65:
            label = "Anomaly"
        person = Users.objects.get(user_id=user_id)
        warning = Warnings(user=person,query=query,timestamp=date.today(),status=label)
        warning.save()
        return Response("This query is anomalous...")

    if (command_type==-1) or (role_score==-1) or (sensitivity==-1) or (avg_count==-1) or (freq_score==-1):
        return Response("There is an error in your query. Please verify your SQL query.")

    query_code=0

    if command_type == "select":
        query_code = 1
    else:
        query_code = 2
    
    q = np.array([query_code,role_score, sensitivity, avg_count*10,freq_score]).reshape(1,-1)

    with open("anomaly_detection_model2.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    # model = Model("output.csv")

    prediction = loaded_model.predict(q)

    print(prediction)

    person = None

    if prediction==2:
        person = Users.objects.get(user_id=user_id)
        warning = Warnings(user=person,query=query,timestamp=date.today(),status="Possible Anomaly")
        warning.save()
        return Response("This query is possibly anomalous...")


    if prediction==0:
        person = Users.objects.get(user_id=user_id)
        warning = Warnings(user=person,query=query,timestamp=date.today(),status="Anomaly")
        warning.save()
        return Response("This query is anomalous...")

    conn = connect()

    cursor = conn.cursor()

    try:
        cursor.execute(query)
    except:
        return Response("There is an error in your query. Please verify your SQL query.")
    

    if command_type=="select":
        rows = cursor.fetchall()
        return Response(rows)
    
    if command_type=="insert":
        row = cursor.fetchone()
    

    conn.commit()
    conn.close()
    
    return Response("Query successfully executed...")


@csrf_exempt
@api_view(['GET'])
def retrieveWarnings(request):
    warnings = Warnings.objects.all()
    warning_serializer = WarningSerializer(warnings,many=True)
    return Response(warning_serializer.data)



    
    



