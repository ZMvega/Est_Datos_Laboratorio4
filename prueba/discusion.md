# Discusion de resultados - Laboratorio 4 (Quadtree vs fuerza bruta)

> Generado automaticamente por `analisis.py` a partir de `crudo.csv` / `resumen.csv` (perfil `rapido`, R = 3 repeticiones, Q = 400 consultas, radio = 500 m). Los numeros son los de la maquina descrita en `entorno.json`.

## 1. ¿Desde que tamano de datos el Quadtree es mas rapido que la fuerza bruta?

- **Busqueda en radio de 500 m vs fuerza bruta (lista):** el Quadtree pasa a ser mas rapido **desde N = 100** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 20 y N = 100, las curvas se cruzan en **N ≈ 37**. Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = 500.
- **Busqueda en radio de 500 m vs fuerza bruta (numpy):** el Quadtree **ya fue mas rapido en el menor tamano medido** (N = 5); el cruce esta en ese valor o por debajo y habria que medir tamanos menores.
- **Vecino mas cercano vs fuerza bruta (lista):** el Quadtree pasa a ser mas rapido **desde N = 100** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 20 y N = 100, las curvas se cruzan en **N ≈ 94**. Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = 500.
- **Vecino mas cercano vs fuerza bruta (numpy):** el Quadtree pasa a ser mas rapido **desde N = 2.000** (primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores). Interpolando en escala log-log entre N = 500 y N = 2.000, las curvas se cruzan en **N ≈ 880**. Con los intervalos de confianza del 95 %, la ventaja no llega a ser estadisticamente clara en el rango medido.

**Respuesta (linea base del enunciado, fuerza bruta con listas):** busqueda en radio -> desde N = 100 (cruce ~ 37); vecino mas cercano -> desde N = 100 (cruce ~ 94).

La fuerza bruta vectorizada con NumPy es una linea base mucho mas exigente: hace el mismo recorrido completo pero en C. Por eso el cruce se desplaza hacia tamanos mayores (radio: el Quadtree ya gana en N = 5 (cruce <= ese valor); vecino: desde N = 2.000 (cruce ~ 880)). **El punto de cruce no es una propiedad del Quadtree solo: depende de que tan buena sea la fuerza bruta con la que se compara.**

## 2. Tiempos con N = 2.000 puntos

| Consulta | Quadtree (µs) | Fuerza bruta lista (µs) | Fuerza bruta NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |
|---|---|---|---|---|---|
| Radio 500 m | 26,5 | 351,1 | 45,5 | 13,2x | 1,7x |
| Vecino mas cercano | 31,5 | 340,3 | 43,3 | 10,8x | 1,4x |

Trabajo por consulta (determinista): el Quadtree calcula en promedio 15 distancias en la busqueda por radio y 9 en el vecino mas cercano, frente a las 2.000 de la fuerza bruta.

## 3. Como escala cada metodo (pendiente log-log, N >= 100)

Si el tiempo crece como N^k, la pendiente es k. La fuerza bruta deberia dar k cercano a 1; el Quadtree con vecino mas cercano, cercano a 0.

| Consulta | Metodo | Pendiente | R² |
|---|---|---|---|
| radio | Quadtree | 0,30 | 0,998 |
| radio | Fuerza bruta (lista) | 1,02 | 1,000 |
| radio | Fuerza bruta (NumPy) | 0,44 | 0,981 |
| vecino | Quadtree | 0,21 | 0,976 |
| vecino | Fuerza bruta (lista) | 1,00 | 1,000 |
| vecino | Fuerza bruta (NumPy) | 0,49 | 0,967 |

En la busqueda por radio el Quadtree no es plano: con la ciudad fija, al aumentar N sube la densidad y el numero de puntos encontrados k crece (de 0,0 con N = 5 a 3,9 con N = 2.000). El costo del Quadtree es O(log N + k), y la fuerza bruta es O(N) siempre; la ventaja relativa es mayor cuando k es pequeno frente a N.

## 4. Costo de construir el arbol

Con N = 2.000, construir el Quadtree tarda 8,9 ms (una sola vez, porque los datos son estaticos). Frente a la fuerza bruta de lista, ese costo se recupera con ~28 consultas de busqueda por radio o ~29 consultas de vecino mas cercano.
Si solo se va a hacer una consulta, la fuerza bruta es mas barata; el arbol compensa cuando los datos estaticos se consultan muchas veces.

## 5. Efecto del radio (N = 2.000)

| Radio (m) | Puntos encontrados | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |
|---|---|---|---|---|---|---|
| 100 | 0,2 | 16,6 | 335,9 | 44,8 | 20,3x | 2,7x |
| 500 | 4,5 | 25,5 | 339,3 | 45,6 | 13,3x | 1,8x |
| 2.000 | 68,2 | 95,7 | 358,3 | 53,2 | 3,7x | 0,6x |

Cuanto mayor es el radio, mas puntos devuelve la consulta y menos se puede podar: el Quadtree se acerca a la fuerza bruta (en el limite, un radio que cubre toda la ciudad devuelve los N puntos y ningun metodo puede evitar recorrerlos).

## 6. Efecto de la dimension K (vecino mas cercano, N = 2.000)

| K | Hijos posibles por nodo | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Distancias calculadas por el Quadtree |
|---|---|---|---|---|---|---|
| 2 | 4 | 28,0 | 341,2 | 53,0 | 12,2x | 9 de 2.000 |
| 3 | 8 | 51,0 | 381,6 | 48,6 | 7,5x | 15 de 2.000 |
| 5 | 32 | 196,8 | 391,7 | 57,0 | 2,0x | 37 de 2.000 |

La tabla muestra cuanta ventaja conserva el arbol en cada K; la ultima columna (distancias calculadas, una metrica determinista) indica cuanto trabajo evita. La explicacion habitual, que este experimento no mide directamente, es que al subir K cada nodo se divide en 2^K regiones y, con un numero fijo de puntos, la mayoria de las cajas quedan casi vacias, mientras que la distancia al vecino mas cercano se parece cada vez mas a la distancia a cualquier otro punto, de modo que la poda descarta menos cajas ("maldicion de la dimensionalidad").

## 7. Limitaciones

- Python es interpretado: los tiempos absolutos y, sobre todo, el punto de cruce dependen de las constantes de la implementacion. Una implementacion en C++ tendria un cruce distinto, con las mismas tendencias.
- Los datos son sinteticos (barrios gaussianos mas un fondo uniforme); en una ciudad real la distribucion cambia la altura del arbol y el trabajo por consulta.
- Las consultas son ubicaciones uniformes sobre la ciudad, incluidas zonas sin entregas. Consultas concentradas en barrios densos devolverian mas puntos.
- Una sola maquina y un solo sistema operativo; la variabilidad se controla con R repeticiones, no se elimina.
- El punto de cruce se estima sobre la malla de N medida: su resolucion es la de esa malla (la interpolacion log-log es solo una estimacion entre dos tamanos medidos).
- La fuerza bruta de lista usa menos consultas por medicion cuando N es grande (presupuesto de distancias), por lo que sus mediciones grandes tienen mas ruido.

