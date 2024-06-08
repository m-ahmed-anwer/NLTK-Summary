from django.urls import path
from .views import summarize

urlpatterns = [
    path('summarize/', summarize, name='summarize')
]
