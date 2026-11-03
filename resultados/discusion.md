# Discusion de resultados - Laboratorio 4 (Quadtree vs fuerza bruta)

> Generado automaticamente por `analisis.py` a partir de `crudo.csv` / `resumen.csv` (perfil `completo`, R = 10 repeticiones, Q = 5.000 consultas, radio = 500 m). Los numeros son los de la maquina descrita en `entorno.json`.

## 1. ¿Desde que tamano de datos el Quadtree es mas rapido que la fuerza bruta?

- **Busqueda en radio de 500 m vs fuerza bruta (lista):** el Quadtree pasa a ser mas rapido **desde N = 50** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 20 y N = 50, las curvas se cruzan en **N ≈ 27**. Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = 50.
- **Busqueda en radio de 500 m vs fuerza bruta (numpy):** el Quadtree **ya fue mas rapido en el menor tamano medido** (N = 5); el cruce esta en ese valor o por debajo y habria que medir tamanos menores.
- **Vecino mas cercano vs fuerza bruta (lista):** el Quadtree pasa a ser mas rapido **desde N = 100** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 50 y N = 100, las curvas se cruzan en **N ≈ 79**. Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = 100.
- **Vecino mas cercano vs fuerza bruta (numpy):** el Quadtree pasa a ser mas rapido **desde N = 1.000** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 500 y N = 1.000, las curvas se cruzan en **N ≈ 657**. Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = 1.000.

**Respuesta (linea base del enunciado, fuerza bruta con listas):** busqueda en radio -> desde N = 50 (cruce ~ 27); vecino mas cercano -> desde N = 100 (cruce ~ 79).

La fuerza bruta vectorizada con NumPy es una linea base mucho mas exigente: hace el mismo recorrido completo pero en C. Por eso el cruce se desplaza hacia tamanos mayores (radio: el Quadtree ya gana en N = 5 (cruce <= ese valor); vecino: desde N = 1.000 (cruce ~ 657)). **El punto de cruce no es una propiedad del Quadtree solo: depende de que tan buena sea la fuerza bruta con la que se compara.**

## 2. Tiempos con N = 10.000 puntos (el caso del enunciado)

| Consulta | Quadtree (µs) | Fuerza bruta lista (µs) | Fuerza bruta NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |
|---|---|---|---|---|---|
| Radio 500 m | 54,2 | 1.781,6 | 185,9 | 32,9x | 3,4x |
| Vecino mas cercano | 30,8 | 1.817,4 | 186,8 | 59,1x | 6,1x |

Trabajo por consulta (determinista): el Quadtree calcula en promedio 40 distancias en la busqueda por radio y 9 en el vecino mas cercano, frente a las 10.000 de la fuerza bruta.

## 3. Como escala cada metodo (pendiente log-log, N >= 100)

Si el tiempo crece como N^k, la pendiente es k. La fuerza bruta deberia dar k cercano a 1; el Quadtree con vecino mas cercano, cercano a 0.

| Consulta | Metodo | Pendiente | R² |
|---|---|---|---|
| radio | Quadtree | 0,52 | 0,949 |
| radio | Fuerza bruta (lista) | 1,02 | 1,000 |
| radio | Fuerza bruta (NumPy) | 0,76 | 0,967 |
| vecino | Quadtree | 0,16 | 0,982 |
| vecino | Fuerza bruta (lista) | 1,02 | 1,000 |
| vecino | Fuerza bruta (NumPy) | 0,78 | 0,969 |

En la busqueda por radio el Quadtree no es plano: con la ciudad fija, al aumentar N sube la densidad y el numero de puntos encontrados k crece (de 0,0 con N = 5 a 200,2 con N = 100.000). El costo del Quadtree es O(log N + k), y la fuerza bruta es O(N) siempre; la ventaja relativa es mayor cuando k es pequeno frente a N.

## 4. Costo de construir el arbol

Con N = 10.000, construir el Quadtree tarda 44,7 ms (una sola vez, porque los datos son estaticos). Frente a la fuerza bruta de lista, ese costo se recupera con ~26 consultas de busqueda por radio o ~25 consultas de vecino mas cercano.
Si solo se va a hacer una consulta, la fuerza bruta es mas barata; el arbol compensa cuando los datos estaticos se consultan muchas veces.

## 5. Efecto del radio (N = 10.000)

| Radio (m) | Puntos encontrados | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |
|---|---|---|---|---|---|---|
| 50 | 0,2 | 20,8 | 1.770,2 | 188,0 | 85,0x | 9,0x |
| 100 | 0,8 | 23,7 | 1.754,6 | 183,6 | 74,1x | 7,8x |
| 250 | 4,8 | 30,9 | 1.774,4 | 188,0 | 57,5x | 6,1x |
| 500 | 19,7 | 53,6 | 1.822,8 | 189,9 | 34,0x | 3,5x |
| 1.000 | 78,1 | 120,0 | 1.795,6 | 192,3 | 15,0x | 1,6x |
| 2.000 | 307,1 | 345,5 | 1.894,1 | 190,5 | 5,5x | 0,6x |
| 5.000 | 1.733,5 | 1.683,2 | 2.677,5 | 193,3 | 1,6x | 0,1x |
| 10.000 | 5.376,9 | 5.046,1 | 5.129,5 | 191,3 | 1,0x | 0,0x |

Cuanto mayor es el radio, mas puntos devuelve la consulta y menos se puede podar: el Quadtree se acerca a la fuerza bruta (en el limite, un radio que cubre toda la ciudad devuelve los N puntos y ningun metodo puede evitar recorrerlos).

## 6. Efecto de la dimension K (vecino mas cercano, N = 10.000)

| K | Hijos posibles por nodo | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Distancias calculadas por el Quadtree |
|---|---|---|---|---|---|---|
| 2 | 4 | 33,1 | 1.809,3 | 189,9 | 54,7x | 8 de 10.000 |
| 3 | 8 | 58,8 | 1.828,0 | 215,2 | 31,1x | 14 de 10.000 |
| 4 | 16 | 123,4 | 1.945,2 | 206,0 | 15,8x | 25 de 10.000 |
| 6 | 64 | 491,4 | 2.099,6 | 244,4 | 4,3x | 82 de 10.000 |
| 8 | 256 | 1.559,0 | 2.279,2 | 252,5 | 1,5x | 82 de 10.000 |

La tabla muestra cuanta ventaja conserva el arbol en cada K; la ultima columna (distancias calculadas, una metrica determinista) indica cuanto trabajo evita. La explicacion habitual, que este experimento no mide directamente, es que al subir K cada nodo se divide en 2^K regiones y, con un numero fijo de puntos, la mayoria de las cajas quedan casi vacias, mientras que la distancia al vecino mas cercano se parece cada vez mas a la distancia a cualquier otro punto, de modo que la poda descarta menos cajas ("maldicion de la dimensionalidad").

## 7. Limitaciones

- Python es interpretado: los tiempos absolutos y, sobre todo, el punto de cruce dependen de las constantes de la implementacion. Una implementacion en C++ tendria un cruce distinto, con las mismas tendencias.
- Los datos son sinteticos (barrios gaussianos mas un fondo uniforme); en una ciudad real la distribucion cambia la altura del arbol y el trabajo por consulta.
- Las consultas son ubicaciones uniformes sobre la ciudad, incluidas zonas sin entregas. Consultas concentradas en barrios densos devolverian mas puntos.
- Una sola maquina y un solo sistema operativo; la variabilidad se controla con R repeticiones, no se elimina.
- El punto de cruce se estima sobre la malla de N medida: su resolucion es la de esa malla (la interpolacion log-log es solo una estimacion entre dos tamanos medidos).
- La fuerza bruta de lista usa menos consultas por medicion cuando N es grande (presupuesto de distancias), por lo que sus mediciones grandes tienen mas ruido.

