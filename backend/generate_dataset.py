import numpy as np
import random
import json
import pandas as pd

class DatasetGenerator:
    def __init__(self):    
        self.queries = ["SELECT","INSERT","UPDATE","DELETE","ALTER"]
        self.roles = ["Administrator","Project Manager","Team Lead","Developer","Quality Assurance"]
        self.tables = ["users","permissions","employee","projects","project_tasks","bug_logs","test_cases","code_repo"]
        self.schema, self.permissions = self.read_schema_info()
        self.dataset_queries = list()
        self.dataset_roles = list()
        self.dataset_sensitivities = list()
        self.avg_data_sizes = list()
        self.frequency_scores = list()
        
    def read_schema_info(self):
    
        with open("schema.json","r") as json_file:
            schema = json.load(json_file)
        
        with open("permissions.json", "r") as json_file:
            permissions = json.load(json_file)
        
        return schema, permissions
    
    
    def check_access(self,role,table_name,query_type):
        ind=0
        if query_type=="SELECT":
            ind=0
        elif query_type=="INSERT" or query_type=="UPDATE":
            ind=1
        else:
            ind=2
            
        return self.permissions[role][table_name][ind]
    
    def get_avg_size(self,tables_selected):
        avg_size=0.0
        total_size=1.0
        
        percent=random.uniform(0.01,1)
        lower_limit=percent-0.05*percent; lower_limit=max(0.01,lower_limit)
        upper_limit=percent+0.05*percent; upper_limit=min(1,upper_limit)
        
        for table in tables_selected:
            size = self.schema[table]["num_rows"]
            total_size += size
            m=random.random()
            if(m<0.15):
                avg_size+=size*random.uniform(0.01,lower_limit)
            elif(m>0.85):
                avg_size+=size*random.uniform(upper_limit,1)
            else:
                avg_size+=size*random.uniform(lower_limit,upper_limit)
        
        return avg_size/total_size
    
    
    def assign_role_score(self,role):
        if role=="Administrator":
            return 1.0
        elif role=="Project Manager":
            return 0.8
        elif role=="Team Lead":
            return 0.7
        elif role=="Developer":
            return 0.6
        else:
            return 0.5
        
    def generate_data(self,query_type):
        
        rows=0
        
        query_code=0
        
        if query_type == "SELECT":
            rows = 600
            query_code = 1
        else:
            rows = 100
            query_code = 2
        
        for i in range(rows):
            
            role = random.choice(self.roles)
            
            role_score = self.assign_role_score(role)
            
            table = random.choice(self.tables)
            
            table_count = 1
            
            table_score = 0
            
            tables_selected = []
            
            access = False
            
            while access==False:
                if self.check_access(role,table,query_type):
                    break
                        
                table = random.choice(self.tables)
                
            if query_type == "SELECT":
                table_mappings = self.schema[table]["table_mappings"]
                
                allowed_tables = [table]
                
                for table_mapping in table_mappings:
                    if self.check_access(role,table_mapping,query_type):
                        table_count += 1
                        allowed_tables.append(table_mapping)
                
                tab = random.randint(1,table_count)
                
                while len(tables_selected) < table_count:
                    t = random.choice(allowed_tables)
                    if t not in tables_selected:
                        tables_selected.append(t)
                
                for tb in tables_selected:
                    table_score += self.schema[tb]["sensitivity"]
            
            else:
                tables_selected.append(table)
                table_score += self.schema[table]["sensitivity"]
            
            avg_data_size = self.get_avg_size(tables_selected)
            
            frequency_score = random.uniform(0.00001,14.0165348)
            
            self.dataset_queries.append(query_code)
            self.dataset_roles.append(role_score)
            self.dataset_sensitivities(table_score)
            self.avg_data_sizes.append(avg_data_size)
            self.frequency_scores.append(frequency_score)


    def create_dataset(self):
        self.generate_data("SELECT")
        self.generate_data("INSERT")
        self.generate_data("UPDATE")
        self.generate_data("DELETE")
        
        df = pd.DataFrame()
        df["Query"] = self.dataset_queries
        df["Role"] = self.dataset_roles
        df["Table Sensitivity"] = self.dataset_sensitivities
        df["Average Data Size"] = self.avg_data_sizes
        df["Frequency Score"] = self.frequency_scores
        
        df.to_csv('output.csv', index=False)
