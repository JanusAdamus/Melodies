# Ampliacion Armonica del Proyecto

## Proposito del cambio

La ampliacion armonica del proyecto responde a una limitacion clara del prototipo inicial: un espacio de estados ocultos restringido a acordes mayores y menores simples puede ser util como demostracion elemental de un Hidden Markov Model, pero resulta insuficiente para describir de forma verosimil la variedad de sonoridades que aparecen en la musica tonal y modal comun. En repertorios simbolicos reales es habitual encontrar triadas no mayores o menores, acordes con septima, sonoridades con novena, acordes suspendidos, acordes con notas anadidas y colecciones de alturas cuyo significado depende del contexto modal local. 

El objetivo de esta ampliacion no fue construir un catalogo ilimitado de acordes, sino definir un vocabulario suficientemente rico para representar armonia comun de manera musicalmente defendible, interpretable en una tesis y computacionalmente razonable dentro de un pipeline basado en HMM finito y HDP-HMM truncado.

## Problema del modelo armonico reducido

En la version original del baseline, cada estado oculto representaba uno de los `24` acordes construidos a partir de las doce clases de altura en dos calidades: mayor o menor. Esa formulacion presenta al menos cuatro problemas.

Primero, ignora sonoridades muy frecuentes como los acordes de septima dominante, los mayores con septima mayor, los menores con septima o los semidisminuidos. Segundo, tiende a colapsar contextos armonicos distintos sobre un mismo estado, por ejemplo un `G7`, un `Gsus4` y un `G9`, que comparten parte del material pero no la misma funcion ni la misma sonoridad. Tercero, obliga a interpretar muchos pasajes modales dentro del eje tonal mayor/menor, lo que empobrece la lectura musical. Cuarto, reduce la capacidad de la capa interpretativa posterior, ya que si el espacio base es demasiado tosco, la trayectoria inferida tambien lo sera.

Por estas razones, la ampliacion armonica se concibio como una mejora estructural del proyecto y no solo como un cambio cosmetico de etiquetas.

## Fundamentos teoricos basicos

### La idea de raiz, calidad y extension

Desde un punto de vista teorico, un acorde puede entenderse como una organizacion de alturas alrededor de una raiz o fundamental, una calidad interna y un grado de complejidad vertical. La raiz identifica el centro de gravedad del acorde. La calidad depende de la disposicion de terceras, cuartas o segundas que definen su identidad principal. La extension describe la presencia de notas adicionales, especialmente septimas, sextas, novenas o notas suspendidas. 

Esta descomposicion es especialmente util en un proyecto computacional porque permite construir un vocabulario controlado: en lugar de tratar cada posible coleccion de notas como una entidad independiente, se representan familias de acordes mediante plantillas intervalicas reutilizables sobre las doce fundamentales.

### Triadas

La triada sigue siendo la unidad armonica elemental mas importante en repertorios tonales y post-tonales moderados. Las cuatro clases incluidas son:

- mayor: `0, 4, 7`
- menor: `0, 3, 7`
- disminuida: `0, 3, 6`
- aumentada: `0, 4, 8`

La inclusion de las formas disminuida y aumentada es necesaria porque ambas aparecen como acordes estables o de paso en armonia tonal y modal, y ademas permiten una lectura mas precisa de materiales sensibles a la atraccion o la simetria.

### Acordes con septima

La septima es el primer nivel de extension verdaderamente indispensable para un analisis armonico mas serio. En este proyecto se incluyen:

- `maj7`
- `7`
- `m7`
- `m(maj7)`
- `ø7`
- `dim7`

Cada uno de estos tipos cumple un papel musical reconocible. El acorde `maj7` es central en contextos tonales estables y colores modales brillantes. El acorde `7` representa la sonoridad dominante clasica y numerosas funciones mixolidias. El acorde `m7` es muy comun en armonia tonal, modal y popular. El acorde `m(maj7)` es menos frecuente, pero sigue siendo una sonoridad historicamente reconocible y suficientemente estable como para justificar su presencia. Los acordes `ø7` y `dim7` son esenciales para describir funciones sensibles, material locrio o pasajes con tension funcional moderada.

### Sextas, suspendidos y notas anadidas

La armonia comun no se agota en la construccion por terceras estrictas. Los acordes `6`, `m6`, `add9`, `madd9`, `sus2`, `sus4`, `7sus4` y `6/9` permiten capturar dos fenomenos frecuentes.

Por un lado, ciertas sonoridades extienden triadas sin convertirse en acordes de novena plenamente funcionales. Por otro, los acordes suspendidos expresan una identidad distinta de la triada clasica, especialmente en repertorios populares, modales y de acompanamiento. El `7sus4` es razonable porque modela una dominante suspendida muy comun sin abrir la puerta a alteraciones mas especializadas que dispararian el tamano del vocabulario.

### Novenas

El sistema se detiene en la novena como extension maxima principal. Se incluyen:

- `maj9`
- `9`
- `m9`

La razon teorica es doble. Musicalmente, la novena es la extension que ya introduce un grado notable de refinamiento armonico y permite distinguir entre triada, septima y un nivel superior de color vertical. Computacionalmente, detenerse aqui evita una explosion innecesaria del espacio de estados. A partir de la undecima y la trecena, las combinaciones posibles crecen con rapidez y obligarian a decidir entre un catalogo excesivo o una agrupacion demasiado arbitraria.

## Incorporacion de la armonia modal

### Por que no basta con mayor y menor

En una gran cantidad de musica simbolica, especialmente repertorio melodico, popular, liturgico, folclorico o de escritura mixta, la organizacion de alturas no se deja explicar del todo por la oposicion tonal mayor/menor. Aparecen rasgos propios de los modos diatonicos: sexta mayor sobre un acorde menor, septima menor sobre un centro mayor, cuarta aumentada estable, segunda menor estructural o quinta disminuida como rasgo interno del contexto.

Si el modelo solo admitiera una lectura mayor/menor, tenderia a reinterpretar como error o ruido muchos comportamientos que en realidad son coherentes dentro de un marco modal.

### Modos implementados

La ampliacion incorpora los siete modos diatonicos principales:

- jonic o `ionian`
- dorico o `dorian`
- frigio o `phrygian`
- lidio o `lydian`
- mixolidio o `mixolydian`
- eolico o `aeolian`
- locrio o `locrian`

La presencia de `locrian` se mantiene porque, aunque es menos estable como centro prolongado, resulta teoricamente justificable para describir acordes disminuidos, semidisminuidos y ciertas colecciones con quinta disminuida.

### Separacion entre acorde y modo

Una decision importante del proyecto es no fundir de manera ciega el acorde y el modo en una sola etiqueta de estado. En lugar de crear un espacio de estados del tipo `raiz x calidad x extension x modo`, se adopta una representacion estructurada en dos niveles:

- el estado armonico del baseline representa la sonoridad principal del acorde;
- el modo se infiere como contexto local compatible con la raiz y el material observado.

Esta decision es esencial para mantener el equilibrio entre riqueza musical y viabilidad computacional. Si se hubiese multiplicado todo el vocabulario de acordes por los siete modos, el baseline habria pasado inmediatamente a un catalogo muy grande, con una matriz de transicion mas costosa y menos interpretable. Al tratar el modo como capa contextual, el proyecto gana sensibilidad modal sin perder control sobre el tamano del modelo.

## Vocabulario armonico implementado

El archivo [`src/models/harmony.py`](../../src/models/harmony.py) define un vocabulario armonico controlado. El conjunto de plantillas abstractas es el siguiente:

- triadas: `maj`, `min`, `dim`, `aug`
- septimas: `maj7`, `7`, `m7`, `m(maj7)`, `ø7`, `dim7`
- sextas y anadidos: `6`, `m6`, `add9`, `madd9`, `6/9`
- suspendidos: `sus2`, `sus4`, `7sus4`
- novenas: `maj9`, `9`, `m9`

Estas `21` plantillas se trasladan a las doce fundamentales cromaticas, produciendo un espacio de `252` estados armonicos para el baseline finito. Ese numero es grande respecto del modelo inicial, pero sigue siendo perfectamente manejable para un HMM clasico con observaciones discretas de `pitch class`.

## Justificacion musical del vocabulario elegido

La seleccion busca representar un repertorio amplio de musica tonal comun, musica modal diatonica y musica popular simbolica sin entrar en armonias demasiado especializadas. La inclusion de triadas, septimas y novenas permite capturar la mayor parte de las sonoridades estructurales que aparecen en progresiones corrientes. Los acordes suspendidos y con novena anadida permiten describir colores muy comunes que de otro modo se confundirian con triadas simples. La presencia de acordes disminuidos, semidisminuidos y disminuidos con septima permite modelar sensibilidad funcional y material locrio o pre-dominante.

Quedaron fuera, de manera deliberada, extensiones por encima de novena, acordes alterados complejos, combinaciones avanzadas de jazz y catalogos de voicings. La razon no es solo economica en terminos de computo; tambien es epistemica. El proyecto busca una representacion armonica interpretable y estable. Un vocabulario excesivo complicaria la inferencia, aumentaria la dispersión de conteos y produciria etiquetas mas dificiles de justificar en una tesis enfocada en musica simbolica general.

## Justificacion computacional

### Control del tamano del espacio de estados

La pregunta central no es si seria posible agregar mas acordes, sino si conviene hacerlo. En un HMM finito, el costo de la inferencia depende directamente del numero de estados y del tamano de la matriz de transicion. Un espacio de `252` estados sigue siendo viable para Viterbi y para el calculo de matrices de emision discretas sobre doce clases de altura. En cambio, si se hubieran incluido decenas de alteraciones, inversiones y extensiones superiores, el espacio crecería rapidamente y perderia robustez frente a secuencias cortas.

### Reutilizacion de plantillas

La implementacion no codifica cada acorde a mano como un caso aislado. Utiliza plantillas intervalicas reutilizables sobre las doce fundamentales. Esta estrategia es importante porque:

- simplifica la definicion del vocabulario,
- facilita el mantenimiento del codigo,
- permite agregar o quitar familias de acordes con cambios acotados,
- hace mas transparente la relacion entre teoria musical y representacion computacional.

### Separacion estructurada de componentes

Otro punto fuerte es la separacion entre componentes conceptuales:

- raiz o fundamental,
- calidad,
- extension,
- contexto modal.

Aunque el baseline final sigue usando un estado unico por acorde para mantener compatibilidad con Viterbi, internamente el vocabulario se construye de forma estructurada. Esto deja el sistema mejor preparado para extensiones futuras, por ejemplo una representacion realmente factorizada o una cadena modal acoplada.

## Efectos sobre el modelo HMM finito

### Emisiones

La matriz de emision del baseline ya no se construye solo a partir de triadas mayor/menor. Ahora cada estado armonico emite probabilidades sobre las doce clases de altura segun tres zonas:

- tonos propios del acorde,
- tonos compatibles con modos diatonicos asociados,
- tonos externos de baja probabilidad.

Esto produce un perfil de emision mas musical. Un `C:maj9` no se comporta como una simple triada de `C`; tampoco un `D:sus4` o un `G:9` se reducen a sus equivalentes mayores o menores.

### Transiciones

La matriz de transicion del baseline sigue siendo heuristica, pero ya no depende solo de la auto-permanencia. Se favorecen:

- permanencia en el mismo acorde,
- transiciones entre acordes con la misma raiz,
- relaciones de cuarta y quinta entre fundamentales,
- compatibilidad modal compartida entre estados.

Esta eleccion es importante porque el aumento del vocabulario exige algun principio de organizacion interna. Sin un sesgo musical minimo, un espacio mas grande se volveria demasiado uniforme y menos expresivo.

### Etiquetas latentes y salida interpretable

El sistema puede ahora producir etiquetas mas expresivas, por ejemplo `C:maj7`, `A:min`, `G:9`, `D:sus4` o `E:ø7`, y al mismo tiempo reportar contextos modales locales como `D:dorian` o `G:mixolydian`. Esto mejora de manera notable la lectura de resultados en tablas, graficos y material de tesis.

## Relacion con el HDP-HMM truncado

El HDP-HMM del proyecto sigue aprendiendo contextos latentes no prefijados, por lo que no se reemplaza por un catalogo armonico fijo. Sin embargo, la ampliacion armonica si afecta su interpretacion posterior. Las emisiones aprendidas por cada estado `z_k` ahora se comparan con el vocabulario armonico extendido y con los modos diatonicos para proponer:

- candidatos de acorde extendido,
- contextos modales probables,
- etiquetas tentativas musicalmente mas informativas.

Esto es valioso porque preserva la honestidad del enfoque no parametrico: el HDP-HMM no nace con un diccionario de acordes obligatorios, pero sus estados pueden interpretarse a posteriori usando un marco armonico mas rico y mejor fundado.

## Ventajas para el proyecto y para la tesis

La ampliacion armonica fortalece el proyecto en varios niveles. Musicalmente, evita un reduccionismo excesivo y reconoce que la armonia comun incluye mas que triadas mayores y menores. Metodologicamente, vuelve mas defendible el baseline, porque lo acerca a un repertorio realista sin renunciar a la claridad del HMM clasico. Computacionalmente, mantiene el sistema dentro de una escala manejable. Interpretativamente, produce etiquetas mas cercanas al lenguaje analitico habitual. Y desde la perspectiva de tesis, deja mejor delimitadas tres capas diferentes:

- una teoria armonica explicitamente definida,
- una implementacion computacional controlada,
- una interpretacion posterior de resultados latentes.

## Ejemplos de codificacion

La ampliacion permite representar ejemplos como los siguientes:

- `C:maj`
- `A:min`
- `B:dim`
- `G:7`
- `D:m7`
- `F:maj7`
- `E:ø7`
- `C:maj9`
- `A:m9`
- `G:9`
- `D:sus4`
- `E:add9`

En la capa modal, el sistema tambien puede reportar contextos como:

- `D:dorian`
- `A:aeolian`
- `G:mixolydian`
- `C:ionian`

Estas etiquetas no implican que el analisis sea perfecto ni univoco, pero si proporcionan un vocabulario interpretable y coherente con la teoria musical comun.

## Alcance actual y extensiones futuras

La implementacion actual se detiene en una frontera deliberada. No modela aun menor armonica, menor melodica, intercambio modal explicito, modulaciones como variable latente separada, inversiones de acordes, acordes alterados avanzados, onceavas, trecenas ni voicings especificos. Esa restriccion no es una carencia accidental, sino una estrategia de diseno para consolidar primero una base armonica clara, robusta y defendible.

La arquitectura ya queda preparada para crecer. La existencia de plantillas de acordes y contextos modales separados facilita la incorporacion posterior de:

- modos adicionales,
- familias escala-acorde mas especializadas,
- representaciones factorizadas del estado oculto,
- o una capa modal probabilistica mas explicita.

## Conclusión

La ampliacion armonica del proyecto constituye una mejora sustantiva y no solo incremental. Reemplaza un inventario armonico minimo por un vocabulario estructurado de triadas, septimas, novenas, suspendidos y acordes con notas anadidas, complementado por una capa modal diatonica. Esta decision amplía la capacidad descriptiva del sistema, fortalece la interpretabilidad de las trayectorias ocultas y mantiene el modelo dentro de limites computacionales razonables. En consecuencia, el proyecto queda mejor alineado tanto con la realidad musical del repertorio simbolico como con las exigencias metodologicas de un trabajo academico serio.
