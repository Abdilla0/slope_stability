# slope/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.UploadDataView.as_view(), name='upload'),
    path('configure/', views.ConfigureAnalysisView.as_view(), name='configure'),
    path('calculate/', views.CalculateView.as_view(), name='calculate'),
    path('export/', views.ExportResultsView.as_view(), name='export'),
    path('download-template/', views.DownloadTemplateView.as_view(), name='download_template'),
]