from django.db import models

class Alumno(models.Model):
    id_alumno = models.PositiveIntegerField(unique=True)
    nombre = models.CharField(max_length=30)

    def __str__(self):
        return self.nombre

class Curso(models.Model):
    nombre = models.CharField(max_length=30, unique=True)

    def __str__(self):
        return self.nombre

class Clase(models.Model):
    fecha = models.DateField()
    id_curso = models.ForeignKey(Curso, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.id_curso} - {self.fecha}"

class Asistencia(models.Model):
    alumno = models.ForeignKey(Alumno, on_delete=models.CASCADE)
    clase = models.ForeignKey(Clase, on_delete=models.CASCADE)
    asistio = models.BooleanField(default=False)

    class Meta:
        unique_together = ('alumno', 'clase')  # Esto asegura que cada alumno solo pueda tener una asistencia por clase

    def __str__(self):
        return f"{self.alumno} - {self.clase} - {self.asistio}"
