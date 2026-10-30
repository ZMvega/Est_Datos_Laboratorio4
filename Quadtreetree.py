import heapq
import math
import random
from operator import itemgetter

# distancia euclidiana, para cualquier K (implementada en C)
distancia = math.dist
_TOL = 1e-12
_ORDEN = itemgetter(1, 0)

#radio medio de la Tierra (metros)
RADIO_TIERRA_M = 6_371_008.8

#Quadtree de K dimensiones
class _Nodo:
    __slots__ = ("lo", "hi", "centro", "indices", "hijos")

    def __init__(self, lo, hi):
        self.lo = lo            # esquina minima de la caja
        self.hi = hi            # esquina maxima de la caja
        self.centro = None      # punto medio (solo en nodos internos)
        self.indices = []       # indices de puntos (solo en hojas)
        self.hijos = None       # {mascara: _Nodo} (solo en nodos internos)


class Quadtree:

    def __init__(self, puntos, capacidad=8, profundidad_max=20, limites=None, dimension=None):
        if capacidad < 1:
            raise ValueError("capacidad debe ser >= 1")
        if profundidad_max < 1:
            raise ValueError("profundidad_max debe ser >= 1")
        self.puntos = [tuple(float(c) for c in p) for p in puntos]
        self.capacidad = capacidad
        self.profundidad_max = profundidad_max

        if self.puntos:
            self.k = len(self.puntos[0])
        elif limites is not None:
            self.k = len(limites[0])
        elif dimension is not None:
            self.k = dimension
        else:
            raise ValueError("sin puntos ni limites hay que indicar la dimension")
        if self.k < 1:
            raise ValueError("la dimension debe ser >= 1")
        if any(len(p) != self.k for p in self.puntos):
            raise ValueError("todos los puntos deben tener la misma dimension")

        if limites is not None:
            lo, hi = tuple(map(float, limites[0])), tuple(map(float, limites[1]))
            if len(lo) != self.k or len(hi) != self.k:
                raise ValueError("los limites deben tener K coordenadas")
        elif self.puntos:
            lo = tuple(min(p[d] for p in self.puntos) for d in range(self.k))
            hi = tuple(max(p[d] for p in self.puntos) for d in range(self.k))
        else:
            raise ValueError("sin puntos hay que indicar los limites")

        self.raiz = _Nodo(lo, hi)
        self.raiz.indices = list(range(len(self.puntos)))
        self._dividir(self.raiz, 0)

    #utilidades
    def __len__(self):
        return len(self.puntos)

    def _validar_punto(self, p):
        q = tuple(float(c) for c in p)
        if len(q) != self.k:
            raise ValueError(f"se esperaban {self.k} coordenadas y llegaron {len(q)}")
        return q

    def _mascara(self, p, centro):
        m = 0
        for d in range(self.k):
            if p[d] >= centro[d]:
                m |= 1 << d
        return m

    def _caja_hijo(self, nodo, mascara):
        lo, hi, c = nodo.lo, nodo.hi, nodo.centro
        lo_h = tuple(c[d] if (mascara >> d) & 1 else lo[d] for d in range(self.k))
        hi_h = tuple(hi[d] if (mascara >> d) & 1 else c[d] for d in range(self.k))
        return lo_h, hi_h

    #construccion
    def _dividir(self, nodo, profundidad):
        """Si la hoja se desborda, la convierte en nodo interno con hijos en las 2^K regiones."""
        indices = nodo.indices
        if len(indices) <= self.capacidad or profundidad >= self.profundidad_max:
            return
        k, puntos = self.k, self.puntos
        centro = tuple((nodo.lo[d] + nodo.hi[d]) / 2.0 for d in range(k))
        grupos = {}
        for i in indices:
            p = puntos[i]
            m = 0
            for d in range(k):
                if p[d] >= centro[d]:
                    m |= 1 << d
            grupos.setdefault(m, []).append(i)

        nodo.centro = centro
        nodo.hijos = {}
        nodo.indices = None
        for m, grupo in grupos.items():
            lo_h, hi_h = self._caja_hijo(nodo, m)
            hijo = _Nodo(lo_h, hi_h)
            hijo.indices = grupo
            nodo.hijos[m] = hijo
            self._dividir(hijo, profundidad + 1)

    def insertar(self, punto):
        """Agrega un punto (debe caer dentro de los limites de la raiz). Devuelve su indice."""
        p = self._validar_punto(punto)
        lo, hi = self.raiz.lo, self.raiz.hi
        if any(p[d] < lo[d] or p[d] > hi[d] for d in range(self.k)):
            raise ValueError("el punto esta fuera de los limites del arbol")
        indice = len(self.puntos)
        self.puntos.append(p)

        nodo, profundidad = self.raiz, 0
        while nodo.hijos is not None:
            m = self._mascara(p, nodo.centro)
            hijo = nodo.hijos.get(m)
            if hijo is None:
                lo_h, hi_h = self._caja_hijo(nodo, m)
                hijo = _Nodo(lo_h, hi_h)
                nodo.hijos[m] = hijo
            nodo = hijo
            profundidad += 1
        nodo.indices.append(indice)
        self._dividir(nodo, profundidad)
        return indice

    #busqueda en radio
    def buscar_en_radio(self, centro, radio, estadisticas=None):

        if radio < 0:
            raise ValueError("el radio no puede ser negativo")
        c = self._validar_punto(centro)
        k, puntos = self.k, self.puntos
        r_poda2 = (radio * (1 + _TOL) + _TOL) ** 2
        r_dentro = radio * (1 - _TOL) - _TOL
        r_dentro2 = r_dentro * r_dentro if r_dentro > 0 else -1.0
        traza = None if estadisticas is None else estadisticas.get("traza")

        resultado = []
        nodos = calculos = 0
        pila = [self.raiz]
        while pila:
            nodo = pila.pop()
            nodos += 1
            lo, hi = nodo.lo, nodo.hi

            # distancia minima del centro a la caja
            dmin2 = 0.0
            for d in range(k):
                x = c[d]
                if x < lo[d]:
                    t = lo[d] - x
                    dmin2 += t * t
                elif x > hi[d]:
                    t = x - hi[d]
                    dmin2 += t * t
            if dmin2 > r_poda2:
                if traza is not None:
                    traza.append((lo, hi, "podada"))
                continue
            if traza is not None:
                traza.append((lo, hi, "visitada"))

            if nodo.hijos is None:                      # hoja: revisar sus puntos
                for i in nodo.indices:
                    calculos += 1
                    dist = distancia(puntos[i], c)
                    if dist <= radio:
                        resultado.append((i, dist))
                continue

            if r_dentro2 > 0:                           # caja completa dentro del circulo
                dmax2 = 0.0
                for d in range(k):
                    a = abs(c[d] - lo[d])
                    b = abs(hi[d] - c[d])
                    t = a if a > b else b
                    dmax2 += t * t
                if dmax2 <= r_dentro2:
                    if traza is not None:
                        traza.append((lo, hi, "dentro"))
                    todos = []
                    sub = [nodo]
                    while sub:
                        n = sub.pop()
                        if n.hijos is None:
                            todos.extend(n.indices)
                        else:
                            sub.extend(n.hijos.values())
                    for i in todos:
                        resultado.append((i, distancia(puntos[i], c)))
                    calculos += len(todos)
                    continue

            pila.extend(nodo.hijos.values())

        resultado.sort(key=_ORDEN)
        if estadisticas is not None:
            estadisticas["nodos"] = estadisticas.get("nodos", 0) + nodos
            estadisticas["distancias"] = estadisticas.get("distancias", 0) + calculos
        return resultado

    #vecinos mas cercanos
    def _dmin(self, nodo, q):
        s = 0.0
        lo, hi = nodo.lo, nodo.hi
        for d in range(self.k):
            x = q[d]
            if x < lo[d]:
                t = lo[d] - x
                s += t * t
            elif x > hi[d]:
                t = x - hi[d]
                s += t * t
        return math.sqrt(s)

    def vecinos_mas_cercanos(self, consulta, k=1, estadisticas=None):
        """Los k puntos mas cercanos a `consulta`: [(indice, distancia)] ordenado por (distancia, indice)."""
        if k < 1:
            raise ValueError("k debe ser >= 1")
        q = self._validar_punto(consulta)
        n = len(self.puntos)
        if n == 0:
            return []
        k = min(k, n)
        puntos = self.puntos

        mejores = []                                    # max-heap de (-distancia, -indice)
        cola = [(self._dmin(self.raiz, q), 0, self.raiz)]
        contador = 1
        nodos = calculos = 0
        while cola:
            dmin, _, nodo = heapq.heappop(cola)
            if len(mejores) == k and dmin > -mejores[0][0] * (1 + _TOL) + _TOL:
                break                                   # ninguna caja restante puede mejorar
            nodos += 1
            if nodo.hijos is None:
                for i in nodo.indices:
                    calculos += 1
                    item = (-distancia(puntos[i], q), -i)
                    if len(mejores) < k:
                        heapq.heappush(mejores, item)
                    elif item > mejores[0]:
                        heapq.heapreplace(mejores, item)
            else:
                for hijo in nodo.hijos.values():
                    dm = self._dmin(hijo, q)
                    if len(mejores) == k and dm > -mejores[0][0] * (1 + _TOL) + _TOL:
                        continue
                    heapq.heappush(cola, (dm, contador, hijo))
                    contador += 1

        if estadisticas is not None:
            estadisticas["nodos"] = estadisticas.get("nodos", 0) + nodos
            estadisticas["distancias"] = estadisticas.get("distancias", 0) + calculos
        return sorted(((-ni, -nd) for nd, ni in mejores), key=_ORDEN)

    def vecino_mas_cercano(self, consulta, estadisticas=None):
        r = self.vecinos_mas_cercanos(consulta, 1, estadisticas)
        return r[0] if r else None

    #inspeccion
    def celdas(self):
        """Genera (lo, hi, profundidad, es_hoja, n_puntos) de cada nodo (para graficar la particion)."""
        pila = [(self.raiz, 0)]
        while pila:
            nodo, prof = pila.pop()
            if nodo.hijos is None:
                yield nodo.lo, nodo.hi, prof, True, len(nodo.indices)
            else:
                yield nodo.lo, nodo.hi, prof, False, 0
                pila.extend((h, prof + 1) for h in nodo.hijos.values())

    def altura(self):
        """Numero de niveles (una sola hoja = 1)."""
        return max((prof for _, _, prof, _, _ in self.celdas()), default=0) + 1

    def num_nodos(self):
        return sum(1 for _ in self.celdas())




# Fuerza bruta: lista + recorrido completo
class FuerzaBruta:
    """Misma interfaz que Quadtree, sin estructura: cada consulta recorre los N puntos."""
    def __init__(self, puntos):
        self.puntos = [tuple(float(c) for c in p) for p in puntos]
        self.k = len(self.puntos[0]) if self.puntos else None

    def __len__(self):
        return len(self.puntos)

    def insertar(self, punto):
        self.puntos.append(tuple(float(c) for c in punto))
        return len(self.puntos) - 1

    def buscar_en_radio(self, centro, radio, estadisticas=None):
        if radio < 0:
            raise ValueError("el radio no puede ser negativo")
        c = tuple(float(x) for x in centro)
        resultado = []
        for i, p in enumerate(self.puntos):
            dist = distancia(p, c)
            if dist <= radio:
                resultado.append((i, dist))
        resultado.sort(key=_ORDEN)
        if estadisticas is not None:
            estadisticas["nodos"] = estadisticas.get("nodos", 0) + 1
            estadisticas["distancias"] = estadisticas.get("distancias", 0) + len(self.puntos)
        return resultado

    def vecinos_mas_cercanos(self, consulta, k=1, estadisticas=None):
        if k < 1:
            raise ValueError("k debe ser >= 1")
        q = tuple(float(x) for x in consulta)
        n = len(self.puntos)
        if n == 0:
            return []
        k = min(k, n)
        if estadisticas is not None:
            estadisticas["nodos"] = estadisticas.get("nodos", 0) + 1
            estadisticas["distancias"] = estadisticas.get("distancias", 0) + n
        if k == 1:
            mejor_d, mejor_i = math.inf, -1
            for i, p in enumerate(self.puntos):
                d = distancia(p, q)
                if d < mejor_d:                         # '<' estricto: en empate gana el menor indice
                    mejor_d, mejor_i = d, i
            return [(mejor_i, mejor_d)]
        elegidos = heapq.nsmallest(k, ((distancia(p, q), i) for i, p in enumerate(self.puntos)))
        return [(i, d) for d, i in elegidos]

    def vecino_mas_cercano(self, consulta, estadisticas=None):
        r = self.vecinos_mas_cercanos(consulta, 1, estadisticas)
        return r[0] if r else None




# Datos: generadores reproducibles y conversion de unidades
def generar_puntos_ciudad(n, semilla=0, ancho=20_000.0, alto=20_000.0,
                          n_barrios=12, fraccion_uniforme=0.25):

    rng = random.Random(semilla)
    lado_min = min(ancho, alto)
    barrios = [(rng.uniform(0.1 * ancho, 0.9 * ancho), rng.uniform(0.1 * alto, 0.9 * alto),
                rng.uniform(0.03, 0.08) * lado_min) for _ in range(n_barrios)]
    pesos = [rng.uniform(0.5, 2.0) for _ in range(n_barrios)]
    elegidos = rng.choices(range(n_barrios), weights=pesos, k=n)

    puntos = []
    for b in elegidos:
        if rng.random() < fraccion_uniforme:
            puntos.append((rng.uniform(0, ancho), rng.uniform(0, alto)))
            continue
        cx, cy, sigma = barrios[b]
        while True:                                      # rechazo: mantener el punto dentro de la ciudad
            x, y = rng.gauss(cx, sigma), rng.gauss(cy, sigma)
            if 0 <= x <= ancho and 0 <= y <= alto:
                break
        puntos.append((x, y))
    return puntos


def generar_puntos_uniformes(n, k, semilla=0, lado=20_000.0):

    rng = random.Random(semilla)
    return [tuple(rng.uniform(0, lado) for _ in range(k)) for _ in range(n)]


def generar_consultas(m, lo, hi, semilla=0):

    rng = random.Random(semilla)
    return [tuple(rng.uniform(lo[d], hi[d]) for d in range(len(lo))) for _ in range(m)]


def latlon_a_metros(latitudes, longitudes, lat0=None, lon0=None):

    if lat0 is None:
        lat0 = sum(latitudes) / len(latitudes)
    if lon0 is None:
        lon0 = sum(longitudes) / len(longitudes)
    factor_x = RADIO_TIERRA_M * math.cos(math.radians(lat0))
    puntos = [(factor_x * math.radians(lon - lon0), RADIO_TIERRA_M * math.radians(lat - lat0))
              for lat, lon in zip(latitudes, longitudes)]
    return puntos, (lat0, lon0)
