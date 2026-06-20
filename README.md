# Python-Llanta | Detector didáctico con OpenCV

Proyecto académico de visión por computadora creado como práctica de detección de objetos con **Python** y **OpenCV**.

El programa abre una cámara o archivo de video, convierte cada frame a escala de grises, aplica un clasificador **Haar Cascade** y dibuja un rectángulo sobre las detecciones encontradas.


---

## Objetivo del proyecto

Mostrar de forma sencilla cómo una aplicación en Python puede:

- Capturar video desde una cámara;
- Procesar imágenes frame por frame;
- Usar un clasificador XML preentrenado;
- Detectar objetos con `detectMultiScale`;
- Dibujar resultados en pantalla;
- Modificar parámetros para observar cambios en la detección.

---

## Tecnologías utilizadas

- Python 3
- OpenCV
- Haar Cascade Classifier
- Cámara web o archivo de video

---

## Estructura del repositorio

```text
python-llanta/
├── cascades/
│   └── cars.xml
├── docs/
│   └── guia_didactica.md
├── examples/
│   └── README.md
├── outputs/
├── src/
│   └── detector.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Clona el repositorio y entra a la carpeta del proyecto:

```bash
git clone https://github.com/SofiPv/Python-Llanta.git
cd Python-Llanta
```

Crea un entorno virtual:

```bash
python -m venv .venv
```

Actívalo en Windows:

```bash
.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución básica

Para abrir la cámara principal:

```bash
python src/detector.py
```

Si tu computadora detecta otra cámara, puedes probar:

```bash
python src/detector.py --source 1
```

También puedes usar un archivo de video:

```bash
python src/detector.py --source video.mp4
```

---

## Controles

| Tecla | Acción |
|---|---|
| `q` | Cerrar el programa |
| `Esc` | Cerrar el programa |
| `p` | Pausar o reanudar el video |
| `s` | Guardar una captura en `outputs/` |

---

## Parámetros didácticos

Puedes modificar la sensibilidad de la detección desde la terminal:

```bash
python src/detector.py --scale-factor 1.2 --min-neighbors 6
```

También puedes cambiar la etiqueta mostrada:

```bash
python src/detector.py --label "Llanta"
```

## Limitaciones

Este proyecto usa un clasificador Haar Cascade clásico, por lo que puede presentar falsos positivos o no detectar objetos en condiciones de poca luz, ángulos difíciles o cámaras con baja calidad.

No está pensado como sistema de seguridad ni como detector industrial, sino como práctica académica y demostración introductoria.
