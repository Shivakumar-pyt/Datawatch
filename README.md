# Anomaly Detection System for Structured Workloads

[Watch our system in action on YouTube!](https://www.youtube.com/watch?v=008oyHMFRUc&t=21s)

## Overview

Our system proposes an innovative approach to anomaly detection tailored for structured workloads such as databases. By extracting essential information from queries, we aim to discern user patterns and identify potential deviations from normal behavior.

## Softwares Used / Tech Stack
- **Frontend: React**
- **Backend: Django**
- **Model and utility programs: Python**
- **Database: MySQL (Hosted on GCP)**
- **API Testing: Postman**

## Dependencies

### Frontend React

- React-router-Dom
- React-BootStrap
- @dotlottie/react-player
- react-lottie
- @react-login-page/base

### Backend Django

- sql
- sql-metadata
- sqlvalidator
- scikitlearn
- numpy
- pandas
- django rest framework
- cors headers


## Roles and Access Rights

The system acknowledges five distinct roles within the database environment:

1. Admin
2. QA
3. Developer
4. Project Manager
5. Team Lead

During evaluation, the associated role of a query is taken into account, as different roles possess varying levels of access rights within the database.

## Command Types

The system recognizes various command types, including:

- **Insert**
- **Update**
- **Delete**
- **Alter**
- **Select**

Weighting is assigned to each command type, with insert, update, and delete commands given higher weights due to their potential to significantly impact the database compared to select queries.

## Sensitivity Score

Each table in the database is assigned a sensitivity score. For every query, the total sensitivity score is computed by summing the sensitivity scores of each involved table.

## Frequency Score

A frequency score is calculated for each query by considering the time difference between the current query and the most recent query within a predetermined threshold range.

## Query Parser

The Query Parser component is responsible for parsing queries and extracting critical details such as command type, table, and table attributes.

## Mediator

The Mediator component simulates the potential impact of query execution on tables within the database. It estimates the average size of data that could be affected by the query.

## ML Model

The model is used for classifying the query as an abnormal, possibly abnormal or a normal query. 
We have made use of the K means clustering algorithm on our unlabelled dataset with 3 clusters corresponding to the three labels.
After running K means on the dataset, we append the labels to the dataset after which the dataset becomes labelled.

Then we have used the Naive Bayes algorithm on this labelled dataset to further improve the accuracy.
