import json
from sql_metadata import Parser

class PreprocessQuery:
    def __init__(self,query,role):
        self.query=query
        self.role=role
        self.parser = Parser(self.query.lower())

        with open("permissions.json","r") as file:
            self.permissions = json.load(file)


    def check_access(self):
        check_ind=0

        if self.parser.tokens[0]=="select":
            check_ind=0
        
        if(self.parser.tokens[0]=="insert") or (self.parser.tokens[0]=="update"):
            check_ind=1

        if self.parser.tokens[0]=="delete":
            check_ind=2

        for table in self.parser.tables:
            if self.permissions[self.role][table][check_ind]==0:
                return False
        
        return True