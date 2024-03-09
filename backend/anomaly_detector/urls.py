from django.urls import path
from . import views

urlpatterns = [
    path('login/',views.loginAuthenticate,name="login"),
    # path('/logout/',views.logoutUser,name="logout"),
    path('processQuery/',views.processQuery,name="processQuery")
]