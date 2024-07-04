from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Alumno)
admin.site.register(Clase)
admin.site.register(Curso)
admin.site.register(Asistencia)