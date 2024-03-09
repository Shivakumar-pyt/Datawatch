from django.db import models


class User(models.Model):   
    user_id = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=255,unique=True)
    password = models.CharField(max_length=255)
    email = models.EmailField()
    role_name = models.CharField(max_length=255)
    last_login = models.DateTimeField()
    active_status = models.SmallIntegerField()

    class Meta:
        db_table = "users"

class Warning(models.Model):
    warning_id = models.IntegerField(primary_key=True)
    user = models.ForeignKey("User",on_delete=models.CASCADE)
    query = models.CharField(max_length=255)
    timestamp = models.DateTimeField()
    status = models.CharField(max_length=255)

    class Meta:
        db_table = "warnings"





