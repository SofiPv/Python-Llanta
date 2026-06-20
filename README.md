# Python-Llanta | Detección didáctica con OpenCV

Proyecto académico de visión por computadora desarrollado en Python.
El programa utiliza OpenCV para capturar video desde una cámara web, procesar cada imagen en escala de grises y aplicar clasificadores Haar Cascade para identificar objetos en tiempo real.

Este repositorio forma parte de mi portafolio de trabajo y muestra el uso básico de detección de objetos mediante archivos XML preentrenados.

---

## Objetivo del proyecto

Aplicar conceptos introductorios de visión por computadora mediante un programa capaz de:

* Capturar video desde una cámara web.
* Procesar imágenes frame por frame.
* Convertir imágenes a escala de grises.
* Cargar clasificadores Haar Cascade en formato XML.
* Detectar objetos con `detectMultiScale`.
* Dibujar rectángulos y etiquetas sobre las detecciones encontradas.

---

## Tecnologías utilizadas

* Python 3
* OpenCV
* Haar Cascade Classifier
* Cámara web

---

## Estructura del repositorio

```text
Python-Llanta/
├── cascades/
│   ├── cars.xml
│   └── cars_2.xml
├── src/
│   └── detector.py
├── haarcascade_car.xml
├── requirements.txt
└── README.md
```

---

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/SofiPv/Python-Llanta.git
```

Entra a la carpeta del proyecto:

```bash
cd Python-Llanta
```

Crea un entorno virtual:

```bash
python -m venv .venv
```

Activa el entorno virtual en Windows:

```bash
.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Ejecuta el programa principal:

```bash
python src/detector.py
```

Al iniciar, se abrirá una ventana con la imagen capturada por la cámara.
Cuando el programa detecte un objeto, mostrará un rectángulo sobre la zona identificada y una etiqueta visual.

Para cerrar la ventana, presiona la tecla:

```text
Esc
```

---

## Funcionamiento general

El programa realiza el siguiente flujo:

1. Abre la cámara web.
2. Captura cada frame del video.
3. Convierte la imagen a escala de grises.
4. Carga clasificadores Haar Cascade desde archivos XML.
5. Aplica detección de objetos con OpenCV.
6. Dibuja rectángulos sobre las detecciones.
7. Muestra el resultado en tiempo real.

---

## Alcance del proyecto

Este proyecto está orientado al aprendizaje y demostración de conceptos básicos de visión por computadora.
La detección puede variar según la iluminación, la calidad de la cámara, el ángulo del objeto y el clasificador utilizado.

No corresponde a un sistema industrial ni de seguridad; su propósito es académico, didáctico y de portafolio.

