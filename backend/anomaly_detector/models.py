# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class BugLogs(models.Model):
    bug_id = models.AutoField(primary_key=True)
    project = models.ForeignKey('Projects', models.DO_NOTHING)
    employee = models.ForeignKey('Employee', models.DO_NOTHING, blank=True, null=True)
    description = models.TextField()
    severity = models.CharField(max_length=255)
    status = models.CharField(max_length=255)
    reported_date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bug_logs'


class CodeRepo(models.Model):
    repository_id = models.AutoField(primary_key=True)
    project = models.ForeignKey('Projects', models.DO_NOTHING)
    repository_name = models.CharField(max_length=255)
    repository_url = models.CharField(max_length=255, blank=True, null=True)
    last_commit_date = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'code_repo'


class Employee(models.Model):
    employee_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.CharField(unique=True, max_length=255)
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    position = models.CharField(max_length=255, blank=True, null=True)
    department_id = models.IntegerField()
    salary = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'employee'


class Permissions(models.Model):
    permission_id = models.AutoField(primary_key=True)
    role_name = models.CharField(max_length=255)
    table_name = models.CharField(max_length=255)
    read_access = models.IntegerField()
    write_access = models.IntegerField()
    delete_access = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'permissions'


class ProjectTasks(models.Model):
    task_id = models.AutoField(primary_key=True)
    project = models.ForeignKey('Projects', models.DO_NOTHING)
    description = models.TextField(blank=True, null=True)
    start_date = models.DateField(blank=True, null=True)
    due_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=255, blank=True, null=True)
    employee = models.ForeignKey(Employee, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'project_tasks'


class Projects(models.Model):
    project_id = models.AutoField(primary_key=True)
    name = models.CharField(unique=True, max_length=255)
    description = models.TextField(blank=True, null=True)
    start_date = models.DateField(blank=True, null=True)
    expected_end_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'projects'


class TestCases(models.Model):
    test_case_id = models.AutoField(primary_key=True)
    project = models.ForeignKey(Projects, models.DO_NOTHING)
    description = models.TextField()
    result = models.TextField()

    class Meta:
        managed = False
        db_table = 'test_cases'


class Users(models.Model):
    user_id = models.AutoField(primary_key=True)
    username = models.CharField(unique=True, max_length=255)
    password = models.CharField(max_length=255)
    email = models.CharField(unique=True, max_length=255)
    role_name = models.CharField(max_length=255)
    last_login = models.DateTimeField(blank=True, null=True)
    active_status = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'


class Warnings(models.Model):
    warning_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(Users, models.DO_NOTHING, blank=True, null=True)
    query = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(blank=True, null=True)
    status = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'warnings'
