import json
from sql_metadata import Parser
import sqlvalidator
from error import CustomError

class PreprocessQuery:
    def __init__(self,query,role):
        self.query=query
        self.role=role
        self.parser = Parser(self.query.lower())

        with open("permissions.json","r") as file:
            self.permissions = json.load(file)


    def check_access(self):

        try:

            check_ind=0

            sql_query = sqlvalidator.parse(self.query)

            if (not sql_query.is_valid()) and (str(self.parser.tokens[0])=="select"):
                return 0
            
            if (str(self.parser.tokens[0]) == "select"):
                check_ind = 0
            elif (str(self.parser.tokens[0])=="delete"):
                check_ind = 2
            else:
                check_ind = 1

            for table in self.parser.tables:
                if self.permissions[self.role][table][check_ind]==0:
                    return 2
            
            return 1

        except:
            return -1