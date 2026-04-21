# Expansion multicorpus para analisis simbolico

## Objetivo

La ampliacion multicorpus incorpora tres fuentes complementarias al corpus inicial de partituras pedagogicas y clasicas:

- `SymbTr`, para abrir el proyecto a musica modal de tradicion makam.
- `PDMX`, para escalar el analisis a un volumen mucho mayor de MusicXML de dominio publico.
- `JAZZMUS`, para introducir lead sheets y repertorio jazzistico con vocabulario armonico mas denso.

La meta no es mezclar indiscriminadamente todos los repertorios, sino construir una capa de ingestión que permita comparar, filtrar y analizar cada coleccion con trazabilidad de procedencia.

## Criterio de integracion

Se mantiene una interfaz comun basada en `music21`, pero cada fuente recibe un adaptador propio. Esto evita perder metadatos musicales relevantes.

### SymbTr

`SymbTr` ya incluye `MusicXML` listo para analisis. Sus nombres de archivo codifican sistematicamente `makam`, `forma`, `usul`, titulo y compositor. El adaptador:

- busca los archivos en `SymbTr/MusicXML`,
- extrae `makam`, `usul` y `symbtr_form`,
- marca `style_system = makam`,
- conserva la posibilidad de analizar el material con el mismo pipeline HMM/HDP-HMM.

Esto no convierte la musica makam en armonia tonal; simplemente permite reutilizar el andamiaje simbolico y registrar el sistema modal original en el catalogo.

### PDMX

El repositorio `pnlong/PDMX` no contiene el corpus completo, sino el codigo y la documentacion del dataset. Por eso el proyecto distingue claramente entre:

- el repositorio clonado de referencia, y
- el dataset local ya descargado desde Zenodo.

Cuando existe un manifiesto CSV, el adaptador intenta:

- resolver rutas a `MusicXML/MXL`,
- recuperar metadatos como `title`, `composer`, `genre`,
- filtrar preferentemente el subconjunto `no_license_conflict`.

Si no existe manifiesto, el sistema cae a un descubrimiento recursivo de MusicXML, pero marca esa limitacion en `ingest_note`.

### JAZZMUS

`JAZZMUS` puede venir como `MusicXML` directo o como archivos `JSON` que contienen la codificacion `musicxml`. El proyecto incorpora una rutina de preparacion que:

- lee los JSON,
- extrae la cadena `musicxml`,
- exporta archivos `.musicxml`,
- deja esos archivos listos para el pipeline actual.

Esto evita depender de un flujo manual previo y facilita un uso reproducible dentro del propio proyecto.

## Nuevas columnas del catalogo

La integracion multicorpus agrega columnas comunes para distinguir el origen del material:

- `source_name`
- `source_type`
- `genre_family`
- `style_system`
- `modal_system`
- `ingest_note`

En fuentes particulares pueden aparecer ademas:

- `makam`, `usul`, `symbtr_form` en `SymbTr`
- `license_conflict`, `pdmx_subset` en `PDMX`

## Justificacion musicologica y computacional

La decision de sumar estas tres fuentes responde a tres ejes distintos:

- `SymbTr` aporta diversidad modal y permite poner a prueba la capa de contexto modal fuera del binomio mayor/menor.
- `PDMX` aporta escala y heterogeneidad dentro del universo de la notacion occidental en MusicXML.
- `JAZZMUS` aporta repertorio donde el vocabulario de septimas, novenas y acordes suspendidos resulta musicalmente central.

Desde el punto de vista computacional, la integracion se mantiene razonable porque el proyecto no cambia el parser base ni obliga a un rediseño completo del modelo; solo enriquece la capa de catalogacion y adquisicion.

## Uso recomendado

### SymbTr

Es la fuente mas lista para usar inmediatamente. Puede integrarse ya con:

```bash
python -m src.cli.multicorpus_analysis \
  --include-symbtr external/SymbTr \
  --model finite_hmm \
  --output-dir outputs/symbtr_batch
```

### JAZZMUS

Si tu carpeta contiene JSON con `musicxml`, la misma CLI prepara MusicXML automaticamente dentro del directorio de salida:

```bash
python -m src.cli.multicorpus_analysis \
  --include-jazzmus /ruta/a/jazzmus_dataset \
  --model both \
  --output-dir outputs/jazzmus_batch
```

### PDMX

Se recomienda usar una copia local del dataset de Zenodo y no solo el repositorio de codigo:

```bash
python -m src.cli.multicorpus_analysis \
  --include-pdmx /ruta/a/PDMX_dataset \
  --limit 200 \
  --model finite_hmm \
  --output-dir outputs/pdmx_batch
```

## Limitaciones y cautelas

- `SymbTr` codifica un sistema musical distinto; las etiquetas armonicas del baseline deben leerse con cautela y como aproximacion descriptiva, no como analisis nativo de teoria makam.
- `PDMX` puede ser muy grande; conviene empezar con `--limit`.
- `JAZZMUS` puede contener multiples versiones del mismo standard; es recomendable resumir resultados por titulo base o filtrar variantes si el experimento lo requiere.
- La mezcla directa de todos los corpus en una sola estadistica agregada puede ocultar diferencias estilisticas fuertes. Por eso el catalogo conserva siempre la procedencia.
