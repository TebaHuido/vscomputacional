from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.generic import View
from django.middleware.csrf import get_token
from .models import Alumno, Clase, Asistencia

class AsistenciaView(View):
    def get(self, request, *args, **kwargs):
        csrf_token = get_token(request)
        print(csrf_token)
        return HttpResponse("100")
    
    def post(self, request, *args, **kwargs):
        person_name = request.POST.get('person', None)
        
        if not person_name:
            return JsonResponse({"error": "Missing person"}, status=400)

        try:
            alumno = Alumno.objects.get(nombre=person_name)
        except Alumno.DoesNotExist:
            return JsonResponse({"error": "Alumno not found"}, status=404)

        clase_id = request.POST.get('clase_id', None)
        clase = None
        if clase_id:
            try:
                clase = Clase.objects.get(pk=clase_id)
            except Clase.DoesNotExist:
                return JsonResponse({"error": "Clase not found"}, status=404)

        asistencia, created = Asistencia.objects.update_or_create(
            alumno=alumno,
            clase=clase,
            defaults={'asistio': True}
        )

        return JsonResponse({"status": "success", "created": created, "asistencia_id": asistencia.id})