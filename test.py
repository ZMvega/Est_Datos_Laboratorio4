import argparse
import math
import os
import random
import statistics
import sys
import unittest

from Quadtreetree import (FuerzaBruta, Quadtree, RADIO_TIERRA_M, generar_consultas,
                          generar_puntos_ciudad, generar_puntos_uniformes, latlon_a_metros)

CIUDAD = 20_000.0
RADIO_ENTREGA_M = 500.0


#PRUEBAS UNITARIAS
class PruebasQuadtree(unittest.TestCase):

    def comparar(self, puntos, consultas, radios, ks=(1, 5), **opciones):
        qt, fb = Quadtree(puntos, **opciones), FuerzaBruta(puntos)
        for c in consultas:
            for r in radios:
                self.assertEqual(qt.buscar_en_radio(c, r), fb.buscar_en_radio(c, r), msg=f"radio {r} en {c}")
            for k in ks:
                self.assertEqual(qt.vecinos_mas_cercanos(c, k), fb.vecinos_mas_cercanos(c, k), msg=f"k={k} en {c}")

    #casos borde
    def test_arbol_vacio(self):
        qt = Quadtree([], limites=((0, 0), (10, 10)))
        self.assertEqual(len(qt), 0)
        self.assertEqual(qt.buscar_en_radio((1, 1), 5), [])
        self.assertEqual(qt.vecinos_mas_cercanos((1, 1), 3), [])
        self.assertIsNone(qt.vecino_mas_cercano((1, 1)))

    def test_un_solo_punto(self):
        qt = Quadtree([(3.0, 4.0)])
        self.assertEqual(qt.buscar_en_radio((0, 0), 5), [(0, 5.0)])
        self.assertEqual(qt.buscar_en_radio((0, 0), 4.99), [])
        self.assertEqual(qt.vecino_mas_cercano((100, 100))[0], 0)

    def test_k_mayor_que_n_devuelve_todos(self):
        pts = generar_puntos_uniformes(7, 2, semilla=1)
        qt = Quadtree(pts)
        self.assertEqual(len(qt.vecinos_mas_cercanos((0, 0), 50)), 7)

    def test_radio_cero_y_radio_enorme(self):
        pts = generar_puntos_ciudad(500, semilla=2)
        qt = Quadtree(pts)
        self.assertEqual([i for i, _ in qt.buscar_en_radio(pts[10], 0)], [10])
        self.assertEqual(len(qt.buscar_en_radio((0, 0), 1e9)), 500)

    def test_puntos_justo_en_el_borde_del_radio(self):
        pts = [(500, 0), (0, 500), (-500, 0), (0, -500), (300, 400), (400, 300),
               (500.0001, 0), (499.9999, 0), (0, 0)]
        qt, fb = Quadtree(pts, capacidad=1), FuerzaBruta(pts)
        esperado = {0, 1, 2, 3, 4, 5, 7, 8}
        self.assertEqual({i for i, _ in qt.buscar_en_radio((0, 0), 500)}, esperado)
        self.assertEqual(qt.buscar_en_radio((0, 0), 500), fb.buscar_en_radio((0, 0), 500))

    def test_resultados_ordenados_por_distancia(self):
        pts = generar_puntos_ciudad(2000, semilla=3)
        qt = Quadtree(pts)
        for res in (qt.buscar_en_radio((10_000, 10_000), 2500), qt.vecinos_mas_cercanos((10_000, 10_000), 30)):
            distancias = [d for _, d in res]
            self.assertEqual(distancias, sorted(distancias))

    #2D vs fuerza bruta
    def test_radio_y_vecinos_en_la_ciudad_de_10000_puntos(self):
        pts = generar_puntos_ciudad(10_000, semilla=4)
        consultas = generar_consultas(150, (0, 0), (CIUDAD, CIUDAD), semilla=5)
        self.comparar(pts, consultas, radios=(RADIO_ENTREGA_M,), ks=(1, 3, 10))

    def test_distintos_radios(self):
        pts = generar_puntos_ciudad(3000, semilla=6)
        consultas = generar_consultas(60, (0, 0), (CIUDAD, CIUDAD), semilla=7)
        self.comparar(pts, consultas, radios=(0, 10, 100, 500, 2000, 8000, 50_000), ks=(1,))

    def test_consultas_fuera_de_la_ciudad(self):
        pts = generar_puntos_ciudad(2000, semilla=8)
        consultas = [(-5000, -5000), (30_000, 10_000), (10_000, 90_000), (-1, 10_000), (CIUDAD, CIUDAD)]
        self.comparar(pts, consultas, radios=(500, 12_000, 60_000), ks=(1, 4))

    def test_distintas_capacidades_y_profundidades(self):
        pts = generar_puntos_ciudad(1500, semilla=9)
        consultas = generar_consultas(40, (0, 0), (CIUDAD, CIUDAD), semilla=10)
        for capacidad in (1, 2, 8, 64):
            self.comparar(pts, consultas, radios=(300, 3000), capacidad=capacidad)
        self.comparar(pts, consultas, radios=(300, 3000), capacidad=2, profundidad_max=3)

    def test_limites_mas_grandes_que_los_datos(self):
        pts = generar_puntos_ciudad(800, semilla=11)
        consultas = generar_consultas(30, (0, 0), (CIUDAD, CIUDAD), semilla=12)
        self.comparar(pts, consultas, radios=(700,), limites=((-1000, -1000), (25_000, 25_000)))

    def test_puntos_repetidos_no_causan_division_infinita(self):
        rng = random.Random(13)
        pts = [(5.0, 5.0)] * 60 + [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(60)]
        qt = Quadtree(pts, capacidad=4, profundidad_max=12)
        self.assertLessEqual(qt.altura(), 13)
        self.comparar(pts, [(5, 5), (50, 50), (500, 500)], radios=(0, 10, 200), ks=(1, 10, 70), capacidad=4, profundidad_max=12)

    def test_dimension_con_todos_los_puntos_iguales_en_un_eje(self):
        pts = [(float(i), 7.0) for i in range(200)]          # todos con y = 7
        self.comparar(pts, [(10, 7), (100.5, 0), (-3, 50)], radios=(0, 5, 50, 1000))

    #K dimensiones
    def test_k_dimensiones(self):
        for k in (1, 2, 3, 4, 5):
            with self.subTest(K=k):
                pts = generar_puntos_uniformes(800, k, semilla=20 + k, lado=1000.0)
                consultas = generar_consultas(40, (0.0,) * k, (1000.0,) * k, semilla=30 + k)
                consultas += [(-200.0,) * k, (1300.0,) * k]
                self.comparar(pts, consultas, radios=(0, 100, 350, 2500), ks=(1, 6))

    def test_k_alto_con_pocos_hijos_guardados(self):
        pts = generar_puntos_uniformes(600, 8, semilla=41, lado=1000.0)
        qt = Quadtree(pts)
        self.assertLess(qt.num_nodos(), 600 * 4)
        consultas = generar_consultas(15, (0.0,) * 8, (1000.0,) * 8, semilla=42)
        self.comparar(pts, consultas, radios=(600,), ks=(1, 3))

    #insercion y validaciones
    def test_insertar_despues_de_construir(self):
        pts = generar_puntos_ciudad(1000, semilla=14)
        qt, fb = Quadtree(pts[:500], limites=((0, 0), (CIUDAD, CIUDAD))), FuerzaBruta(pts[:500])
        for p in pts[500:]:
            self.assertEqual(qt.insertar(p), fb.insertar(p))
        for c in generar_consultas(40, (0, 0), (CIUDAD, CIUDAD), semilla=15):
            self.assertEqual(qt.buscar_en_radio(c, 900), fb.buscar_en_radio(c, 900))
            self.assertEqual(qt.vecinos_mas_cercanos(c, 4), fb.vecinos_mas_cercanos(c, 4))

    def test_insertar_fuera_de_limites_falla(self):
        qt = Quadtree([(0, 0), (10, 10)])
        with self.assertRaises(ValueError):
            qt.insertar((11, 5))

    def test_validaciones(self):
        qt = Quadtree([(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            qt.buscar_en_radio((0, 0), -1)
        with self.assertRaises(ValueError):
            qt.buscar_en_radio((0, 0, 0), 5)
        with self.assertRaises(ValueError):
            qt.vecinos_mas_cercanos((0, 0), 0)
        with self.assertRaises(ValueError):
            Quadtree([(0, 0)], capacidad=0)
        with self.assertRaises(ValueError):
            Quadtree([(0, 0), (1, 1, 1)])
        with self.assertRaises(ValueError):
            Quadtree([])

    #estructura interna
    def punto_en_hoja_caja(self):
        pts = generar_puntos_ciudad(3000, semilla=16)
        qt = Quadtree(pts, capacidad=8)
        vistos = []
        pila = [(qt.raiz, 0)]
        while pila:
            nodo, prof = pila.pop()
            if nodo.hijos is None:
                if prof < qt.profundidad_max:
                    self.assertLessEqual(len(nodo.indices), qt.capacidad)
                for i in nodo.indices:
                    p = pts[i]
                    for d in range(2):
                        self.assertTrue(nodo.lo[d] <= p[d] <= nodo.hi[d])
                vistos.extend(nodo.indices)
            else:
                self.assertIsNone(nodo.indices)
                self.assertLessEqual(len(nodo.hijos), 4)
                pila.extend((h, prof + 1) for h in nodo.hijos.values())
        self.assertEqual(sorted(vistos), list(range(3000)))

    def test_poda_reduce_el_trabajo(self):
        n = 10_000
        pts = generar_puntos_ciudad(n, semilla=17)
        qt = Quadtree(pts)
        calculos = []
        for c in generar_consultas(100, (0, 0), (CIUDAD, CIUDAD), semilla=18):
            est = {}
            qt.buscar_en_radio(c, RADIO_ENTREGA_M, est)
            calculos.append(est["distancias"])
        self.assertLess(statistics.mean(calculos), 0.10 * n)

    #datos y unidades
    def test_generador_reproducible_y_dentro_de_la_ciudad(self):
        a = generar_puntos_ciudad(500, semilla=19)
        self.assertEqual(a, generar_puntos_ciudad(500, semilla=19))
        self.assertNotEqual(a, generar_puntos_ciudad(500, semilla=20))
        self.assertTrue(all(0 <= x <= CIUDAD and 0 <= y <= CIUDAD for x, y in a))

    def test_conversion_latlon_a_metros(self):
        lat0, lon0 = 6.2442, -75.5812
        # 0,001 grados de latitud ~ 111,2 m
        (p0, p1), _ = latlon_a_metros([lat0, lat0 + 0.001], [lon0, lon0], lat0, lon0)
        self.assertAlmostEqual(math.dist(p0, p1), 111.2, delta=0.3)
        # 0,001 grados de longitud ~ 111,2 * cos(lat) m (menos que en latitud)
        (q0, q1), _ = latlon_a_metros([lat0, lat0], [lon0, lon0 + 0.001], lat0, lon0)
        self.assertAlmostEqual(math.dist(q0, q1), 111.2 * math.cos(math.radians(lat0)), delta=0.3)
        # contra la formula de haversine, a ~1,5 km
        lat2, lon2 = lat0 + 0.010, lon0 + 0.008
        (a, b), _ = latlon_a_metros([lat0, lat2], [lon0, lon2], lat0, lon0)
        p1r, p2r = math.radians(lat0), math.radians(lat2)
        h = (math.sin((p2r - p1r) / 2) ** 2
             + math.cos(p1r) * math.cos(p2r) * math.sin(math.radians(lon2 - lon0) / 2) ** 2)
        haversine = 2 * RADIO_TIERRA_M * math.asin(math.sqrt(h))
        self.assertLess(abs(math.dist(a, b) - haversine) / haversine, 0.005)


#ANALISIS DE LAS PRUEBAS
def verificacion_masiva(qt, fb, n_consultas, radio, semilla):
    lo, hi = (0.0, 0.0), (CIUDAD, CIUDAD)
    consultas = generar_consultas(n_consultas, lo, hi, semilla)
    r = {"consultas": n_consultas, "radio": radio, "n": len(qt),
         "desacuerdos_radio": 0, "desacuerdos_vecino": 0,
         "encontrados": [], "distancias_radio": [], "distancias_vecino": [], "nodos_radio": []}
    for c in consultas:
        est = {}
        a = qt.buscar_en_radio(c, radio, est)
        if a != fb.buscar_en_radio(c, radio):
            r["desacuerdos_radio"] += 1
        r["encontrados"].append(len(a))
        r["distancias_radio"].append(est["distancias"])
        r["nodos_radio"].append(est["nodos"])

        est = {}
        if qt.vecino_mas_cercano(c, est) != fb.vecino_mas_cercano(c):
            r["desacuerdos_vecino"] += 1
        r["distancias_vecino"].append(est["distancias"])
    return r


def verificacion_dimensiones(dimensiones=(1, 2, 3, 4, 5, 6), n=2000, n_consultas=200, objetivo=20):
    filas = []
    lado = 1000.0
    for k in dimensiones:
        volumen_unitario = math.pi ** (k / 2) / math.gamma(k / 2 + 1)
        radio = lado * (objetivo / n / volumen_unitario) ** (1 / k)
        pts = generar_puntos_uniformes(n, k, semilla=100 + k, lado=lado)
        qt, fb = Quadtree(pts), FuerzaBruta(pts)
        malos_r = malos_v = 0
        encontrados = []
        for c in generar_consultas(n_consultas, (0.0,) * k, (lado,) * k, semilla=200 + k):
            a = qt.buscar_en_radio(c, radio)
            encontrados.append(len(a))
            malos_r += a != fb.buscar_en_radio(c, radio)
            malos_v += qt.vecinos_mas_cercanos(c, 5) != fb.vecinos_mas_cercanos(c, 5)
        filas.append((k, n, n_consultas, radio, statistics.mean(encontrados), malos_r, malos_v))
    return filas


def escribir_resumen(ruta, masiva, filas_k, resultado_unitarias, qt):
    dist_r, dist_v, n = masiva["distancias_radio"], masiva["distancias_vecino"], masiva["n"]
    lineas = [
        "RESUMEN DE LAS PRUEBAS - Laboratorio 4 (Quadtree)",
        f"Pruebas unitarias: {resultado_unitarias.testsRun} ejecutadas, "
        f"{len(resultado_unitarias.failures)} fallos, {len(resultado_unitarias.errors)} errores",
        "",
        f"Verificacion masiva (K = 2, N = {n:,} puntos, ciudad de {CIUDAD / 1000:.0f} km x {CIUDAD / 1000:.0f} km)",
        f"  Consultas aleatorias comparadas con fuerza bruta: {masiva['consultas']:,}",
        f"  Radio de busqueda: {masiva['radio']:.0f} m",
        f"  Discrepancias en busqueda por radio:    {masiva['desacuerdos_radio']}",
        f"  Discrepancias en vecino mas cercano:    {masiva['desacuerdos_vecino']}",
        f"  Puntos encontrados por consulta (radio): promedio {statistics.mean(masiva['encontrados']):.1f}, "
        f"mediana {statistics.median(masiva['encontrados']):.0f}, maximo {max(masiva['encontrados'])}",
        "",
        "Trabajo por consulta (distancias calculadas; la fuerza bruta calcula N = "
        f"{n:,} siempre) - metrica determinista, no depende de la maquina",
        f"  Busqueda por radio: promedio {statistics.mean(dist_r):.0f} ({statistics.mean(dist_r) / n * 100:.2f} % de N), "
        f"maximo {max(dist_r)} ({max(dist_r) / n * 100:.2f} % de N)",
        f"  Vecino mas cercano: promedio {statistics.mean(dist_v):.0f} ({statistics.mean(dist_v) / n * 100:.2f} % de N), "
        f"maximo {max(dist_v)} ({max(dist_v) / n * 100:.2f} % de N)",
        f"  Cajas examinadas en busqueda por radio: promedio {statistics.mean(masiva['nodos_radio']):.0f} "
        f"de {qt.num_nodos():,} nodos del arbol",
        "",
        f"Estructura del arbol: altura {qt.altura()} niveles, {qt.num_nodos():,} nodos, capacidad de hoja {qt.capacidad}",
        "",
        "Verificacion en K dimensiones (Quadtree vs fuerza bruta; el radio se ajusta para devolver ~20 puntos)",
        f"  {'K':>2} {'N':>6} {'consultas':>9} {'radio':>9} {'encontrados':>11} {'disc. radio':>11} {'disc. 5-vecinos':>15}",
    ]
    for k, n_k, q, radio, prom, mr, mv in filas_k:
        lineas.append(f"  {k:>2} {n_k:>6} {q:>9} {radio:>9.1f} {prom:>11.1f} {mr:>11} {mv:>15}")
    texto = "\n".join(lineas)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    return texto


#VISUALIZACIONES
def _importar_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    return plt, Circle, Rectangle


def _dibujar_celdas(ax, Rectangle, celdas, xlim, ylim, color="#9aa7b4", ancho=0.6):
    for lo, hi in celdas:
        if hi[0] < xlim[0] or lo[0] > xlim[1] or hi[1] < ylim[0] or lo[1] > ylim[1]:
            continue
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False,
                               edgecolor=color, linewidth=ancho, zorder=1))


def _hojas(qt):
    return [(lo, hi) for lo, hi, _, es_hoja, _ in qt.celdas() if es_hoja]


def elegir_consulta(fb, radio, minimo, maximo, semilla):
    for c in generar_consultas(2000, (0, 0), (CIUDAD, CIUDAD), semilla):
        if minimo <= len(fb.buscar_en_radio(c, radio)) <= maximo:
            return c
    raise RuntimeError("no se encontro una consulta adecuada para graficar")


def figura_radio(qt, fb, centro, radio, ruta):
    plt, Circle, Rectangle = _importar_matplotlib()
    res_qt, res_fb = qt.buscar_en_radio(centro, radio), fb.buscar_en_radio(centro, radio)
    idx_qt, idx_fb = {i for i, _ in res_qt}, {i for i, _ in res_fb}
    faltan, sobran = idx_fb - idx_qt, idx_qt - idx_fb
    xs, ys = zip(*qt.puntos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6))
    #vista general
    ax1.scatter(xs, ys, s=1.5, c="#b9b9b9", label=f"Puntos de entrega (N = {len(qt):,})")
    ax1.add_patch(Circle(centro, radio, fill=False, edgecolor="#d62728", linewidth=1.6))
    ax1.scatter([qt.puntos[i][0] for i in idx_qt], [qt.puntos[i][1] for i in idx_qt], s=10, c="#d62728",
                label=f"Encontrados a <= {radio:.0f} m: {len(idx_qt)}")
    ax1.scatter([centro[0]], [centro[1]], marker="*", s=220, c="black", edgecolors="white", zorder=5,
                label="Ubicacion consultada")
    medio = 1.5 * radio
    ax1.add_patch(Rectangle((centro[0] - medio, centro[1] - medio), 2 * medio, 2 * medio, fill=False,
                            edgecolor="#1f77b4", linewidth=1.2, linestyle="--", zorder=4))
    ax1.set_xlim(0, CIUDAD)
    ax1.set_ylim(0, CIUDAD)
    ax1.set_title("Vista general de la ciudad (el recuadro azul es el zoom)")

    #zoom con la particion del arbol
    xlim, ylim = (centro[0] - medio, centro[0] + medio), (centro[1] - medio, centro[1] + medio)
    _dibujar_celdas(ax2, Rectangle, _hojas(qt), xlim, ylim)
    dentro = [(i, p) for i, p in enumerate(qt.puntos) if xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]]
    ax2.scatter([p[0] for i, p in dentro if i not in idx_qt], [p[1] for i, p in dentro if i not in idx_qt],
                s=16, c="#8c8c8c", label="Punto de entrega fuera del radio", zorder=2)
    ax2.scatter([qt.puntos[i][0] for i in idx_qt], [qt.puntos[i][1] for i in idx_qt], s=34, c="#d62728",
                edgecolors="black", linewidths=0.4, label="Encontrado por el Quadtree", zorder=3)
    if faltan or sobran:
        ax2.scatter([qt.puntos[i][0] for i in faltan | sobran], [qt.puntos[i][1] for i in faltan | sobran],
                    s=120, facecolors="none", edgecolors="magenta", linewidths=2, label="DISCREPANCIA", zorder=6)
    ax2.add_patch(Circle(centro, radio, fill=False, edgecolor="#d62728", linewidth=1.8, linestyle="--",
                         label=f"Circulo de radio {radio:.0f} m", zorder=4))
    ax2.scatter([centro[0]], [centro[1]], marker="*", s=260, c="black", edgecolors="white", zorder=5,
                label="Ubicacion consultada")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    veredicto = "COINCIDEN" if not (faltan or sobran) else "NO COINCIDEN"
    ax2.set_title(f"Zoom: Quadtree {len(idx_qt)} puntos | fuerza bruta {len(idx_fb)} puntos -> {veredicto}")
    ax2.text(0.02, 0.02, "Lineas grises: celdas (hojas) del Quadtree", transform=ax2.transAxes,
             fontsize=8, color="#555555")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Verificacion visual de la busqueda por radio ({radio:.0f} m)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return len(idx_qt), len(idx_fb), faltan | sobran


def figura_vecinos(qt, fb, consulta, ruta, k=5):
    plt, Circle, Rectangle = _importar_matplotlib()
    vec_qt, vec_fb = qt.vecinos_mas_cercanos(consulta, k), fb.vecinos_mas_cercanos(consulta, k)
    d1, dk = vec_qt[0][1], vec_qt[-1][1]
    medio = 1.4 * dk
    xlim, ylim = (consulta[0] - medio, consulta[0] + medio), (consulta[1] - medio, consulta[1] + medio)
    hojas = _hojas(qt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6))

    for ax, kk, titulo in ((ax1, 1, "Vecino mas cercano (k = 1)"), (ax2, k, f"{k} vecinos mas cercanos (k = {k})")):
        elegidos = vec_qt[:kk]
        radio = elegidos[-1][1]
        _dibujar_celdas(ax, Rectangle, hojas, xlim, ylim)
        visibles = [(i, p) for i, p in enumerate(qt.puntos) if xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]]
        marcados = {i for i, _ in elegidos}
        ax.scatter([p[0] for i, p in visibles if i not in marcados], [p[1] for i, p in visibles if i not in marcados],
                   s=16, c="#8c8c8c", label="Otros puntos de entrega", zorder=2)
        ax.add_patch(Circle(consulta, radio, fill=False, edgecolor="#2ca02c", linewidth=1.8, linestyle="--",
                            label=f"Circulo hasta el vecino {kk} ({radio:.0f} m)", zorder=3))
        for orden, (i, d) in enumerate(elegidos, start=1):
            p = qt.puntos[i]
            ax.plot([consulta[0], p[0]], [consulta[1], p[1]], color="#2ca02c", linewidth=1, zorder=3)
            ax.scatter([p[0]], [p[1]], s=70, c="#2ca02c", edgecolors="black", linewidths=0.6, zorder=4,
                       label="Vecino(s) encontrado(s) por el Quadtree" if orden == 1 else None)
            ax.annotate(f"{orden}: {d:.0f} m", p, xytext=(6, 6), textcoords="offset points", fontsize=9, zorder=6)
        ax.scatter([consulta[0]], [consulta[1]], marker="*", s=260, c="black", edgecolors="white", zorder=5,
                   label="Ubicacion consultada")
        dentro = len(fb.buscar_en_radio(consulta, radio))
        ax.set_title(f"{titulo}\npuntos dentro del circulo: {dentro} (deben ser {kk})")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    coincide = vec_qt == vec_fb
    fig.suptitle("Verificacion visual de los vecinos mas cercanos - "
                 + ("Quadtree y fuerza bruta COINCIDEN" if coincide else "DISCREPANCIA con fuerza bruta"), fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return coincide


def figura_particion(qt, fb, centro, radio, ruta):
    plt, Circle, Rectangle = _importar_matplotlib()
    est = {"traza": []}
    res = qt.buscar_en_radio(centro, radio, est)
    hojas = _hojas(qt)
    conjunto_hojas = set(hojas)
    xs, ys = zip(*qt.puntos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6))
    ax1.scatter(xs, ys, s=1.2, c="#b9b9b9", zorder=2)
    _dibujar_celdas(ax1, Rectangle, hojas, (0, CIUDAD), (0, CIUDAD), ancho=0.3)
    ax1.set_xlim(0, CIUDAD)
    ax1.set_ylim(0, CIUDAD)
    ax1.set_title(f"Particion del Quadtree: {len(hojas):,} hojas, altura {qt.altura()}\n"
                  "(las zonas densas se dividen mas; las vacias, casi nada)")

    medio = 3.2 * radio
    xlim, ylim = (centro[0] - medio, centro[0] + medio), (centro[1] - medio, centro[1] + medio)
    _dibujar_celdas(ax2, Rectangle, hojas, xlim, ylim, ancho=0.4)
    visitadas = [(lo, hi) for lo, hi, estado in est["traza"] if estado == "visitada" and (lo, hi) in conjunto_hojas]
    podadas = [(lo, hi) for lo, hi, estado in est["traza"] if estado == "podada"]
    adentro = [(lo, hi) for lo, hi, estado in est["traza"] if estado == "dentro"]
    for lo, hi in visitadas:
        ax2.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], facecolor="#2ca02c", alpha=0.30,
                                edgecolor="#2ca02c", linewidth=1.0, zorder=1))
    for lo, hi in adentro:
        ax2.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], facecolor="#1f77b4", alpha=0.25,
                                edgecolor="#1f77b4", linewidth=1.0, zorder=1))
    for lo, hi in podadas:
        ax2.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, edgecolor="#d62728",
                                linewidth=1.0, linestyle=":", zorder=1))
    dentro = [(i, p) for i, p in enumerate(qt.puntos) if xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]]
    ids = {i for i, _ in res}
    ax2.scatter([p[0] for i, p in dentro if i not in ids], [p[1] for i, p in dentro if i not in ids], s=10, c="#8c8c8c", zorder=2)
    ax2.scatter([qt.puntos[i][0] for i in ids], [qt.puntos[i][1] for i in ids], s=26, c="#d62728", zorder=3)
    ax2.add_patch(Circle(centro, radio, fill=False, edgecolor="#d62728", linewidth=1.8, zorder=4))
    ax2.scatter([centro[0]], [centro[1]], marker="*", s=240, c="black", edgecolors="white", zorder=5)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_title(f"Una consulta de {radio:.0f} m: {est['nodos']} cajas examinadas y {est['distancias']} distancias calculadas\n"
                  f"(la fuerza bruta calcularia {len(qt):,})")
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor="#2ca02c", alpha=0.5,
               markeredgecolor="#2ca02c", label="Hoja visitada (se calcularon distancias)"),
        Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor="#1f77b4", alpha=0.5,
               markeredgecolor="#1f77b4", label="Caja totalmente dentro del circulo (puntos sin comparar)"),
        Line2D([], [], color="#d62728", linestyle=":", label="Caja descartada (podada) sin mirar sus puntos"),
        Line2D([], [], marker="o", linestyle="", color="#d62728", label="Punto encontrado dentro del radio"),
        Line2D([], [], marker="o", linestyle="", color="#8c8c8c", label="Otro punto de entrega"),
        Line2D([], [], marker="*", linestyle="", markersize=12, color="black", label="Ubicacion consultada"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle("Como poda el Quadtree: solo se miran las celdas que tocan el circulo", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return est["nodos"], est["distancias"]


def figura_masiva(masiva, ruta):
    plt, _, _ = _importar_matplotlib()
    n = masiva["n"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    ax1.hist(masiva["encontrados"], bins=range(0, max(masiva["encontrados"]) + 2, max(1, max(masiva["encontrados"]) // 40)),
             color="#d62728", alpha=0.85)
    ax1.set_xlabel(f"Puntos encontrados dentro de {masiva['radio']:.0f} m")
    ax1.set_ylabel("Numero de consultas")
    ax1.set_title(f"Resultado de {masiva['consultas']:,} consultas por radio\n"
                  f"(discrepancias con fuerza bruta: {masiva['desacuerdos_radio']})")
    ax2.hist(masiva["distancias_radio"], bins=40, color="#1f77b4", alpha=0.75, label="Quadtree: busqueda por radio")
    ax2.hist(masiva["distancias_vecino"], bins=40, color="#ff7f0e", alpha=0.65, label="Quadtree: vecino mas cercano")
    ax2.axvline(n, color="black", linestyle="--", label=f"Fuerza bruta: siempre N = {n:,}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Distancias calculadas por consulta (escala logaritmica)")
    ax2.set_ylabel("Numero de consultas")
    ax2.set_title("Trabajo por consulta: Quadtree vs fuerza bruta\n(metrica independiente de la maquina)")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


#programa principal
def main():
    parser = argparse.ArgumentParser(description="Pruebas, verificaciones visuales y analisis de pruebas del Quadtree")
    parser.add_argument("--salida", default="graficas", help="carpeta para graficas y resumen (por defecto graficas)")
    parser.add_argument("--semilla", type=int, default=2026, help="semilla de los datos de las verificaciones")
    parser.add_argument("--sin-visuales", action="store_true", help="solo pruebas unitarias y resumen de texto")
    args = parser.parse_args()

    print("1) PRUEBAS UNITARIAS")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PruebasQuadtree)
    resultado = unittest.TextTestRunner(verbosity=2).run(suite)
    if not resultado.wasSuccessful():
        print("\nHay pruebas que fallan: no se generan graficas con un codigo incorrecto.")
        sys.exit(1)

    os.makedirs(args.salida, exist_ok=True)
    print("\n")
    print("2) ANALISIS DE LAS PRUEBAS (10.000 puntos de entrega)")
    puntos = generar_puntos_ciudad(10_000, semilla=args.semilla, ancho=CIUDAD, alto=CIUDAD)
    qt, fb = Quadtree(puntos), FuerzaBruta(puntos)
    masiva = verificacion_masiva(qt, fb, n_consultas=2000, radio=RADIO_ENTREGA_M, semilla=args.semilla + 1)
    filas_k = verificacion_dimensiones()
    texto = escribir_resumen(os.path.join(args.salida, "resumen_pruebas.txt"), masiva, filas_k, resultado, qt)
    print(texto)
    if masiva["desacuerdos_radio"] or masiva["desacuerdos_vecino"] or any(f[5] or f[6] for f in filas_k):
        print("\nHAY DISCREPANCIAS entre el Quadtree y la fuerza bruta.")
        sys.exit(1)

    if args.sin_visuales:
        return
    try:
        _importar_matplotlib()
    except ImportError:
        print("\nmatplotlib no esta instalado: se omiten las graficas (pip install matplotlib).")
        return

    print("\n")
    print("3) VERIFICACION VISUAL")
    centro = elegir_consulta(fb, RADIO_ENTREGA_M, 15, 60, args.semilla + 2)
    n_qt, n_fb, dif = figura_radio(qt, fb, centro, RADIO_ENTREGA_M, os.path.join(args.salida, "verificacion_radio.png"))
    print(f"  verificacion_radio.png: Quadtree {n_qt} puntos | fuerza bruta {n_fb} puntos | discrepancias {len(dif)}")

    consulta_nn = elegir_consulta(fb, 700.0, 3, 12, args.semilla + 3)
    coincide = figura_vecinos(qt, fb, consulta_nn, os.path.join(args.salida, "verificacion_vecinos.png"))
    print(f"  verificacion_vecinos.png: coincide con fuerza bruta = {coincide}")

    nodos, calculos = figura_particion(qt, fb, centro, RADIO_ENTREGA_M, os.path.join(args.salida, "particion_quadtree.png"))
    print(f"  particion_quadtree.png: {nodos} cajas examinadas, {calculos} distancias calculadas (de {len(qt):,})")

    figura_masiva(masiva, os.path.join(args.salida, "verificacion_masiva.png"))
    print("  verificacion_masiva.png")
    print(f"\nGraficas y resumen guardados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
