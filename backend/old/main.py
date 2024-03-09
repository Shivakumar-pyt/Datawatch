from sql_metadata import Parser
from datetime import datetime, timedelta
import mysql.connector
import json


conn = mysql.connector.connect(
        host="34.16.97.93",
        user="root",
        password="123456789",
        database="CompanyInfo"
)

cursor = conn.cursor()

def read_schema_info():
    
    with open("schema.json","r") as json_file:
        schema = json.load(json_file)
    
    return schema

query="INSERT INTO projects (name, description, start_date, expected_end_date, status) VALUES ('Project y', 'This is a new project', '2024-03-10', '2024-09-10', 'In Progress'), ('Project X', 'This is a new project', '2024-03-10', '2024-09-10', 'In Progress'), ('Project X', 'This is a new project', '2024-03-10', '2024-09-10', 'In Progress');".lower()
parser = Parser(query)


# Varibles Initialization
command_type=str(parser.tokens[0])
tables=parser.tables
subqueries=parser.subqueries
logs={}
schema=read_schema_info()

operation_counts = {
    "select": 50,
    "insert": 50,
    "delete": 50,
    "update": 50,
    "alter": 50
}




# Mediator code

def transform(query):
    index=query.find("from")
    return "select count(*) "+query[index:]

# Calculates average retrieved size of tables 
def avgCount(query):
    
    print(command_type)
    sum=0; dem=0;
    if(command_type=="alter"):
        return 1
    elif(command_type=="select"):
        query2=transform(query)
        cursor.execute(query2)
        row = cursor.fetchone()[0]
        sum+=row

        for query in subqueries:
            query2=transform(query)
            cursor.execute(query2)
            row = cursor.fetchone()[0]
            sum+=row
    elif (command_type=="insert"):
        row=len(parser.values); n=len(parser.values_dict)
        sum=row/n
    else:
        query2="select count(*) "+tables[0]+" "+query[query.find("where"):]
        cursor.execute(query)
        row = cursor.fetchone()[0]
        sum=row

    if(sum==0):
        return 0
    for table in tables:
        dem+=schema[table]["num_rows"]
    
    return float(sum/dem)


# Calculates total sensitivity score of tables 
def sensitivity():
    sensitivity=0
    for table in tables:
        sensitivity+=schema[table]["sensitivity"]

    return sensitivity


# Calculates frequency score
def cal_Freq_Score():
    fScore=0; totalSeconds=608400 
    now=datetime.now()
    for time in logs[command_type]:
        difference=now-time
        if(difference.days>7):
            break
        seconds_difference=difference.seconds
        fScore+=totalSeconds/(seconds_difference*100000)
    
    return fScore

        