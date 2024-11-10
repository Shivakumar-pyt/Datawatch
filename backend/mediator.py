from sql_metadata import Parser
from datetime import datetime, timedelta
import json
from generate_dataset import DatasetGenerator
from error import CustomError

class QueryAnalyzer:
    def __init__(self, query, conn, role, schema_file="schema.json", log_file="logs.json"):
        self.query = query.lower()
        self.conn = conn
        self.parser = Parser(self.query.lower())
        self.cursor = conn.cursor()
        self.schema = self.read_schema_info(schema_file)
        self.logs = self.read_log_info(log_file)
        self.role = role
        
    def read_schema_info(self, file):
        with open(file, "r") as json_file:
            schema = json.load(json_file)
        return schema
    
    def read_log_info(self, file):
        with open(file, "r") as json_file:
            logs = json.load(json_file)
        return logs

    def transform(self):
        try:
            index = self.query.find("from")
            return "select count(*) " + self.query[index:]
        except:
            return -1

    def avg_count(self):
        try:
            command_type = str(self.parser.tokens[0])
            sum_count = 0
            dem = 0
            
            if command_type == "alter":
                return 1
            
            elif command_type == "select":
                query2 = self.transform()
                self.cursor.execute(query2)
                row = self.cursor.fetchone()[0]
                sum_count += row

                for subquery in self.parser.subqueries:
                    query2 = self.transform(subquery)
                    self.cursor.execute(query2)
                    row = self.cursor.fetchone()[0]
                    sum_count += row

            elif command_type == "insert":
                row = len(self.parser.values)
                n = len(self.parser.values_dict)
                sum_count = row / n

            else:
                query2 = "select count(*) from " + self.parser.tables[0] + " " + self.query[self.query.find("where"):]
                self.cursor.execute(query2)
                row = self.cursor.fetchone()[0]
                sum_count = row

            if sum_count == 0:
                return 0
            
            for table in self.parser.tables:
                dem += self.schema[table]["num_rows"]
            
            return float(sum_count / dem)
        except:
            return -1
            # raise CustomError("There is an error in your SQL syntax. Please verify your SQL query.")

    def sensitivity(self):
        try:

            sensitivity_score = 0
            for table in self.parser.tables:
                sensitivity_score += self.schema[table]["sensitivity"]
            return sensitivity_score
        except:
            return -1

    def cal_freq_score(self, command_type):
        try:
            date_format = '%Y-%m-%d %H:%M:%S.%f'
            f_score = 0
            total_seconds = 365.04
            now = datetime.now()

            n = len(self.logs[command_type][self.role])
            
            for i in range(n - 1, -1, -1):
                stime = self.logs[command_type][self.role][i]
                time = datetime.strptime(stime, date_format)
                difference = now - time
                
                if difference.days > 7:
                    break
                
                seconds_difference = difference.seconds
                f_score += total_seconds / (seconds_difference + 1)
            
            self.logs[command_type][self.role].append(str(datetime.now()))
            
            with open('logs.json', 'w') as file:
                json.dump(self.logs, file, indent=4)
            
            return f_score
        except:
            return -1

    def get_query_info(self):
        dataset_gen = DatasetGenerator()
        command_type = str(self.parser.tokens[0])
        role_score = dataset_gen.assign_role_score(self.role)
        sensitivity = self.sensitivity()
        avg_count = self.avg_count()
        freq_score = self.cal_freq_score(command_type)
        return command_type, role_score, sensitivity, avg_count, freq_score
