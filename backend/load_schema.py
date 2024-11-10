import mysql.connector
import json
from connect import connect

class LoadSchema:
    def __init__(self):
        self.conn = connect()
        self.cursor = self.conn.cursor()
        self.query = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = %s"
        self.query2 = "SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema = %s AND table_name = %s"
        self.query3 = "SELECT * FROM permissions"
        self.tables = ["bug_logs","projects","users","permissions","code_repo","project_tasks","test_cases","warnings",
                       "employee"]
        self.table_sensitivities = self.table_sensitivities()
        
    
    def table_sensitivities():
        table_sensitivities = dict()
        table_sensitivities["projects"] = 1.0
        table_sensitivities["users"] = 0.9
        table_sensitivities["permissions"] = 0.8
        table_sensitivities["code_repo"] = 0.7
        table_sensitivities["bug_logs"] = 0.6
        table_sensitivities["employee"] = 0.5
        table_sensitivities["project_tasks"] = 0.4
        table_sensitivities["test_cases"] = 0.4
        
        return table_sensitivities
    
    
    def table_mappings(table_name):
        if table_name == "bug_logs":
            return ["projects"]
        elif table_name == "code_repo":
            return ["projects"]
        elif table_name == "employee":
            return ["project_tasks","users","permissions"]
        elif table_name == "permissions":
            return ["users","employee"]
        elif table_name == "project_tasks":
            return ["projects","employee"]
        elif table_name == "projects":
            return ["code_repo","project_tasks","bug_logs","test_cases"]
        elif table_name == "test_cases":
            return ["projects"]
        else:
            return ["permissions","employee"]
        
    
    def save_schema_as_json(self):
        data = {}
        for table in self.tables:
            self.cursor.execute(self.query2,(self.conn.database,table))
            row = self.cursor.fetchone()
            data[table] = {}
            data[table]["num_rows"] = row[1]
            data[table]["sensitivity"] = self.table_sensitivities[table]
            data[table]["columns"] = []
            data[table]["table_mappings"] = self.table_mappings(table)
        
        self.cursor.execute(self.query,(self.conn.database,))
        rows = self.cursor.fetchall()
        
        for row in rows:
            data[row[0]]["columns"].append(row[1])
            
        with open("schema.json", "w") as json_file:
            json.dump(data, json_file)

            
    def save_permissions_as_json(self):
        self.cursor.execute(self.query3)
        rows = self.cursor.fetchall()
        
        permissions = {}
        for role in ["Administrator","Project Manager","Team Lead","Developer","Quality Assurance"]:
            permissions[role] = {}
        
        for row in rows:
            permissions[row[1]][row[2]] = [row[3],row[4],row[5]]
        
        with open("permissions.json", "w") as json_file:
            json.dump(permissions, json_file)
        
        