# Generated by Django 5.0.6 on 2024-07-04 14:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('asistencia', '0003_alter_asistencia_unique_together'),
    ]

    operations = [
        migrations.AlterField(
            model_name='asistencia',
            name='clase',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='asistencia.clase'),
        ),
    ]