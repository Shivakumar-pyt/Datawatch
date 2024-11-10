from rest_framework import serializers
from .models import Warnings, Users

class WarningSerializer(serializers.ModelSerializer):
    class Meta:
        model = Warnings
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = ['user_id','role_name',]