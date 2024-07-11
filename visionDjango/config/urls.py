from django.contrib import admin
from django.urls import path
from asistencia.views import AsistenciaView  # Importa tu vista

urlpatterns = [
    path('admin/', admin.site.urls),
    path('asistencia/', AsistenciaView.as_view(), name='asistencia'),  # Define la ruta para la vista Asistencia
]
