# Usar una imagen base de Debian
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de requisitos al contenedor
COPY /requirements.txt /app/requirements.txt

# Instalar las dependencias necesarias para compilar OpenCV y otras librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libqt5gui5 \
    libxcb-xinerama0 \
    && pip install --no-cache-dir -r requirements.txt


# Copiar el resto del código fuente al contenedor
COPY . .

# Ejecutar el script cuando se inicie el contenedor
CMD ["python", "ReconocimientoFacial.py"]
