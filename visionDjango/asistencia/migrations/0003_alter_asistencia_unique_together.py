# Generated by Django 5.0.6 on 2024-07-04 09:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('asistencia', '0002_asistencia_asistio'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='asistencia',
            unique_together={('alumno', 'clase')},
        ),
    ]