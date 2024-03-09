import mysql.connector

def connect():
    conn = mysql.connector.connect(
            host="34.16.97.93",
            user="root",
            password="123456789",
            database="CompanyInfo"
    )
    
    return conn

