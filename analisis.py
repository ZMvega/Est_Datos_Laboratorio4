import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import textwrap
import time
from datetime import datetime

import numpy as np

from Quadtreetree import (FuerzaBruta, Quadtree, distancia, generar_consultas,
                          generar_puntos_ciudad, generar_puntos_uniformes)

CIUDAD = 20_000.0
QT, LISTA, NUMPY = "Quadtree", "Fuerza bruta (lista)", "Fuerza bruta (NumPy)"
METODOS = (QT, LISTA, NUMPY)
ABREV = {QT: "quadtree", LISTA: "lista", NUMPY: "numpy"}
MIN_CONSULTAS_LISTA = 50

PERFILES = {
    "rapido": dict(tamanos=[5, 20, 100, 500, 2000], repeticiones=3, consultas=400, radio=500.0,
                   n_fijo=2000, radios=[100.0, 500.0, 2000.0], dimensiones=[2, 3, 5],
                   repeticiones_aux=2),
    "completo": dict(tamanos=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
                     repeticiones=10, consultas=5000, radio=500.0, n_fijo=10_000,
                     radios=[50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0],
                     dimensiones=[2, 3, 4, 6, 8], repeticiones_aux=5),
}

CAMPOS_CRUDO = ["experimento", "consulta", "N", "k_dim", "radio_m", "metodo", "repeticion", "semilla",
                "tam_operacion", "tiempo_s", "tiempo_us_por_op", "resultados_promedio",
                "distancias_promedio", "nodos_promedio"]
CAMPOS_RESUMEN = ["experimento", "consulta", "N", "k_dim", "radio_m", "metodo", "n_ejecuciones",
                  "tam_operacion", "media_us", "desv_std_us", "mediana_us", "min_us", "max_us", "ic95_us",
                  "n_atipicos", "media_sin_atipicos_us", "resultados_promedio", "distancias_promedio",
                  "nodos_promedio"]



#fuerza bruta vectorizada
class FuerzaBrutaNumpy:
    nombre = NUMPY

    def __init__(self, puntos):
        self.P = np.asarray(puntos, dtype=float)

    def buscar_en_radio(self, centro, radio):
        d = self.P - centro
        return np.flatnonzero(np.einsum("ij,ij->i", d, d) <= radio * radio)

    def vecino_mas_cercano(self, consulta):
        d = self.P - consulta
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))


def cronometrar(funcion, *args):
    gc.collect()
    gc.disable()
    try:
        inicio = time.perf_counter()
        resultado = funcion(*args)
        duracion = time.perf_counter() - inicio
    finally:
        gc.enable()
    return duracion, resultado


def _consultar_radio(estructura, consultas, radio):
    buscar = estructura.buscar_en_radio
    for c in consultas:
        buscar(c, radio)


def _consultar_vecino(estructura, consultas):
    buscar = estructura.vecino_mas_cercano
    for c in consultas:
        buscar(c)


def consultas_efectivas(metodo, n, q, presupuesto):
    if metodo == LISTA:
        return min(q, max(MIN_CONSULTAS_LISTA, presupuesto // max(n, 1)))
    return q


def verificar(estructuras, consultas, radio, n_ver=25):
    qt, lista, npb = estructuras[QT], estructuras[LISTA], estructuras[NUMPY]
    for c in consultas[:n_ver]:
        if qt.buscar_en_radio(c, radio) != lista.buscar_en_radio(c, radio):
            raise AssertionError("el Quadtree no coincide con la fuerza bruta en una busqueda por radio")
        if qt.vecino_mas_cercano(c) != lista.vecino_mas_cercano(c):
            raise AssertionError("el Quadtree no coincide con la fuerza bruta en el vecino mas cercano")
        if set(npb.buscar_en_radio(c, radio).tolist()) != {i for i, _ in lista.buscar_en_radio(c, radio)}:
            raise AssertionError("la fuerza bruta NumPy no coincide en una busqueda por radio")
        i_np, (i_ref, d_ref) = npb.vecino_mas_cercano(c), lista.vecino_mas_cercano(c)
        if i_np != i_ref and abs(distancia(lista.puntos[i_np], c) - d_ref) > 1e-9:
            raise AssertionError("la fuerza bruta NumPy no coincide en el vecino mas cercano")


def trabajo_del_arbol(qt, consultas, radio, n_muestra=300):
    m = min(len(consultas), n_muestra)
    r_enc, r_dist, r_nod, v_dist, v_nod = [], [], [], [], []
    for c in consultas[:m]:
        est = {}
        r_enc.append(len(qt.buscar_en_radio(c, radio, est)))
        r_dist.append(est["distancias"])
        r_nod.append(est["nodos"])
        est = {}
        qt.vecino_mas_cercano(c, est)
        v_dist.append(est["distancias"])
        v_nod.append(est["nodos"])
    mean = statistics.fmean
    return {"radio": (mean(r_enc), mean(r_dist), mean(r_nod)), "vecino": (1.0, mean(v_dist), mean(v_nod))}



#estadistica
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_critico_95(gl):
    return float("nan") if gl < 1 else (_T95[gl - 1] if gl <= 30 else 1.96)


def resumir(valores):
    n = len(valores)
    media = statistics.fmean(valores)
    desv = statistics.stdev(valores) if n > 1 else 0.0
    ic95 = t_critico_95(n - 1) * desv / math.sqrt(n) if n > 1 else float("nan")
    sin, atipicos = valores, 0
    if n >= 5:
        q1, _, q3 = statistics.quantiles(valores, n=4, method="inclusive")
        iqr = q3 - q1
        sin = [v for v in valores if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
        atipicos = n - len(sin)
    return {"n_ejecuciones": n, "media_us": media, "desv_std_us": desv, "mediana_us": statistics.median(valores),
            "min_us": min(valores), "max_us": max(valores), "ic95_us": ic95, "n_atipicos": atipicos,
            "media_sin_atipicos_us": statistics.fmean(sin)}



def modelo_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.lower().startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def memoria_total_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("MemTotal"):
                    return round(int(linea.split()[1]) / 1024 ** 2, 2)
    except OSError:
        pass
    return None


def describir_entorno(args, parametros):
    return {
        "fecha_inicio": datetime.now().isoformat(timespec="seconds"),
        "sistema_operativo": platform.platform(), "arquitectura": platform.machine(),
        "cpu": modelo_cpu(), "nucleos_logicos": os.cpu_count(), "ram_total_gb": memoria_total_gb(),
        "python": sys.version.replace("\n", " "), "numpy": np.__version__,
        "reloj": "time.perf_counter (tiempo de pared)",
        "recolector_basura_durante_medicion": "desactivado (gc.disable)",
        "ejecutado_en_colab": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ,
        "parametros": parametros, "argumentos_cli": vars(args),
    }


#experimentos
class Registro:
    """Escribe cada medicion en crudo.csv apenas se obtiene"""

    def __init__(self, archivo):
        self.archivo = archivo
        self.escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_CRUDO)
        self.escritor.writeheader()

    def fila(self, **campos):
        base = {c: "" for c in CAMPOS_CRUDO}
        base.update(campos)
        self.escritor.writerow(base)

    def vaciar(self):
        self.archivo.flush()


def _medir_metodos(reg, estructuras, consultas, radio, presupuesto, rng, **ident):
    metodos = list(estructuras)
    rng.shuffle(metodos)
    tiempos = {}
    for metodo in metodos:
        m = consultas_efectivas(metodo, ident["N"], len(consultas), presupuesto)
        cs = consultas[:m]
        t_r, _ = cronometrar(_consultar_radio, estructuras[metodo], cs, radio)
        t_v, _ = cronometrar(_consultar_vecino, estructuras[metodo], cs)
        tiempos[metodo] = (m, t_r, t_v)
    return tiempos


def experimento_escalamiento(reg, args, p):
    for n in p["tamanos"]:
        for rep in range(p["repeticiones"]):
            semilla = f"{args.semilla}-esc-{n}-{rep}"
            rng = random.Random(semilla + "-orden")
            puntos = generar_puntos_ciudad(n, semilla=semilla, ancho=CIUDAD, alto=CIUDAD)
            consultas = generar_consultas(p["consultas"], (0.0, 0.0), (CIUDAD, CIUDAD), semilla + "-q")
            t_build, qt = cronometrar(Quadtree, puntos)
            estructuras = {QT: qt, LISTA: FuerzaBruta(puntos), NUMPY: FuerzaBrutaNumpy(puntos)}
            verificar(estructuras, consultas, p["radio"])
            trabajo = trabajo_del_arbol(qt, consultas, p["radio"])
            reg.fila(experimento="escalamiento", consulta="construccion", N=n, metodo=QT, repeticion=rep,
                     semilla=semilla, tam_operacion=1, tiempo_s=t_build, tiempo_us_por_op=t_build * 1e6)
            tiempos = _medir_metodos(reg, estructuras, consultas, p["radio"], args.presupuesto, rng, N=n)
            linea = []
            for metodo, (m, t_r, t_v) in tiempos.items():
                for consulta, t in (("radio", t_r), ("vecino", t_v)):
                    enc, dist, nod = trabajo[consulta]
                    if metodo != QT:
                        dist, nod = float(n), 1.0
                    reg.fila(experimento="escalamiento", consulta=consulta, N=n, radio_m=p["radio"] if consulta == "radio" else "",
                             metodo=metodo, repeticion=rep, semilla=semilla, tam_operacion=m, tiempo_s=t,
                             tiempo_us_por_op=t / m * 1e6, resultados_promedio=enc, distancias_promedio=dist,
                             nodos_promedio=nod)
                linea.append(f"{ABREV[metodo]}: radio {t_r / m * 1e6:9.1f} us")
            reg.vaciar()
            print(f"[escalamiento N={n:>7,} rep {rep + 1:>2}/{p['repeticiones']}] construir {t_build * 1e3:8.1f} ms | " + " | ".join(linea))


def experimento_radio(reg, args, p):
    n = p["n_fijo"]
    for rep in range(p["repeticiones_aux"]):
        semilla = f"{args.semilla}-rad-{n}-{rep}"
        rng = random.Random(semilla + "-orden")
        puntos = generar_puntos_ciudad(n, semilla=semilla, ancho=CIUDAD, alto=CIUDAD)
        consultas = generar_consultas(min(p["consultas"], 2000), (0.0, 0.0), (CIUDAD, CIUDAD), semilla + "-q")
        qt = Quadtree(puntos)
        estructuras = {QT: qt, LISTA: FuerzaBruta(puntos), NUMPY: FuerzaBrutaNumpy(puntos)}
        for radio in p["radios"]:
            verificar(estructuras, consultas, radio, n_ver=10)
            trabajo = trabajo_del_arbol(qt, consultas, radio, n_muestra=100)
            metodos = list(estructuras)
            rng.shuffle(metodos)
            linea = []
            for metodo in metodos:
                m = consultas_efectivas(metodo, n, len(consultas), args.presupuesto)
                t, _ = cronometrar(_consultar_radio, estructuras[metodo], consultas[:m], radio)
                enc, dist, nod = trabajo["radio"]
                if metodo != QT:
                    dist, nod = float(n), 1.0
                reg.fila(experimento="barrido_radio", consulta="radio", N=n, radio_m=radio, metodo=metodo,
                         repeticion=rep, semilla=semilla, tam_operacion=m, tiempo_s=t, tiempo_us_por_op=t / m * 1e6,
                         resultados_promedio=enc, distancias_promedio=dist, nodos_promedio=nod)
                linea.append(f"{ABREV[metodo]}: {t / m * 1e6:9.1f} us")
            reg.vaciar()
            print(f"[radio {radio:>7.0f} m rep {rep + 1}/{p['repeticiones_aux']}] " + " | ".join(linea))


def experimento_dimension(reg, args, p):
    n = p["n_fijo"]
    for k in p["dimensiones"]:
        for rep in range(p["repeticiones_aux"]):
            semilla = f"{args.semilla}-dim-{k}-{rep}"
            rng = random.Random(semilla + "-orden")
            puntos = generar_puntos_uniformes(n, k, semilla=semilla, lado=CIUDAD)
            consultas = generar_consultas(min(p["consultas"], 1000), (0.0,) * k, (CIUDAD,) * k, semilla + "-q")
            t_build, qt = cronometrar(Quadtree, puntos)
            estructuras = {QT: qt, LISTA: FuerzaBruta(puntos), NUMPY: FuerzaBrutaNumpy(puntos)}
            for c in consultas[:15]:                         # verificacion (el vecino mas cercano es lo que se mide)
                if qt.vecino_mas_cercano(c) != estructuras[LISTA].vecino_mas_cercano(c):
                    raise AssertionError(f"el Quadtree no coincide con la fuerza bruta (K = {k})")
            trabajo = trabajo_del_arbol(qt, consultas, 1.0, n_muestra=100)["vecino"]
            reg.fila(experimento="dimension", consulta="construccion", N=n, k_dim=k, metodo=QT, repeticion=rep,
                     semilla=semilla, tam_operacion=1, tiempo_s=t_build, tiempo_us_por_op=t_build * 1e6)
            metodos = list(estructuras)
            rng.shuffle(metodos)
            linea = []
            for metodo in metodos:
                m = consultas_efectivas(metodo, n, len(consultas), args.presupuesto)
                t, _ = cronometrar(_consultar_vecino, estructuras[metodo], consultas[:m])
                dist, nod = (trabajo[1], trabajo[2]) if metodo == QT else (float(n), 1.0)
                reg.fila(experimento="dimension", consulta="vecino", N=n, k_dim=k, metodo=metodo, repeticion=rep,
                         semilla=semilla, tam_operacion=m, tiempo_s=t, tiempo_us_por_op=t / m * 1e6,
                         resultados_promedio=1, distancias_promedio=dist, nodos_promedio=nod)
                linea.append(f"{ABREV[metodo]}: {t / m * 1e6:9.1f} us")
            reg.vaciar()
            print(f"[dimension K={k} rep {rep + 1}/{p['repeticiones_aux']}] " + " | ".join(linea))


# Resumen estadistico
def _num(texto, tipo=float):
    return tipo(texto) if texto != "" else None


def resumir_csv(ruta_crudo, ruta_resumen):
    grupos = {}
    with open(ruta_crudo, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            clave = (r["experimento"], r["consulta"], int(r["N"]), _num(r["k_dim"], int), _num(r["radio_m"]), r["metodo"])
            g = grupos.setdefault(clave, {"t": [], "tam": int(r["tam_operacion"]), "res": [], "dist": [], "nod": []})
            g["t"].append(float(r["tiempo_us_por_op"]))
            for campo, lista in (("resultados_promedio", "res"), ("distancias_promedio", "dist"), ("nodos_promedio", "nod")):
                if r[campo] != "":
                    g[lista].append(float(r[campo]))

    orden_exp = {"escalamiento": 0, "barrido_radio": 1, "dimension": 2}
    orden_con = {"construccion": 0, "radio": 1, "vecino": 2}
    orden_met = {m: i for i, m in enumerate(METODOS)}
    claves = sorted(grupos, key=lambda k: (orden_exp[k[0]], orden_con[k[1]], k[3] or 0, k[4] or 0, k[2], orden_met[k[5]]))
    with open(ruta_resumen, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_RESUMEN)
        w.writeheader()
        for clave in claves:
            exp, con, n, k, radio, metodo = clave
            g = grupos[clave]
            fila = {"experimento": exp, "consulta": con, "N": n, "k_dim": "" if k is None else k,
                    "radio_m": "" if radio is None else radio, "metodo": metodo, "tam_operacion": g["tam"]}
            fila.update(resumir(g["t"]))
            for campo, lista in (("resultados_promedio", "res"), ("distancias_promedio", "dist"), ("nodos_promedio", "nod")):
                fila[campo] = statistics.fmean(g[lista]) if g[lista] else ""
            w.writerow(fila)
    return len(claves)


def cargar_resumen(ruta):
    filas = []
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["N"] = int(r["N"])
            r["k_dim"] = _num(r["k_dim"], int)
            r["radio_m"] = _num(r["radio_m"])
            for c in CAMPOS_RESUMEN[6:]:
                r[c] = _num(r[c]) if r[c] != "" else None
            filas.append(r)
    return filas


def serie(res, experimento, consulta, metodo, **filtro):
    s = [r for r in res if r["experimento"] == experimento and r["consulta"] == consulta and r["metodo"] == metodo
         and all(r[k] == v for k, v in filtro.items())]
    clave = {"escalamiento": "N", "barrido_radio": "radio_m", "dimension": "k_dim"}[experimento]
    return sorted(s, key=lambda r: r[clave])


def pendiente_loglog(x, y):
    lx, ly = np.log10(x), np.log10(y)
    p = np.polyfit(lx, ly, 1)
    ss = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - float(np.sum((ly - (p[0] * lx + p[1])) ** 2)) / ss if ss > 0 else float("nan")
    return float(p[0]), r2


def tabla_pendientes(res, ruta, n_min=100):
    esperado = {(QT, "radio"): "tiende a 1 para N grande: el costo es log N + k y k crece con N (ciudad fija)",
                (QT, "vecino"): "cercana a 0 (O(log N))",
                (LISTA, "radio"): "1 (O(N))", (LISTA, "vecino"): "1 (O(N))",
                (NUMPY, "radio"): "1 (O(N)); menos de 1 con N pequeno por el costo fijo de NumPy",
                (NUMPY, "vecino"): "1 (O(N)); menos de 1 con N pequeno por el costo fijo de NumPy"}
    filas = []
    for consulta in ("radio", "vecino"):
        for metodo in METODOS:
            s = serie(res, "escalamiento", consulta, metodo)
            for etiqueta, sub in (("todos los N", s), (f"N >= {n_min}", [r for r in s if r["N"] >= n_min])):
                if len(sub) >= 3:
                    pend, r2 = pendiente_loglog([r["N"] for r in sub], [r["media_us"] for r in sub])
                    filas.append({"consulta": consulta, "metodo": metodo, "rango": etiqueta, "n_puntos": len(sub),
                                  "pendiente": round(pend, 3), "r2": round(r2, 4),
                                  "esperada_aprox": esperado[(metodo, consulta)]})
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


#cruce
def calcular_cruce(ns, t_a, ic_a, t_b, ic_b):
    n = len(ns)
    mejor = [t_a[i] < t_b[i] for i in range(n)]
    claro = [t_a[i] + (ic_a[i] or 0) < t_b[i] - (ic_b[i] or 0) for i in range(n)]
    primero = next((i for i in range(n) if all(mejor[i:])), None)
    out = {"ns": ns, "estado": None, "n_primero": None, "n_interp": None, "n_previo": None, "n_claro": None}
    if primero is None:
        out["estado"] = "nunca"
        return out
    out["n_primero"] = ns[primero]
    out["n_claro"] = next((ns[i] for i in range(n) if all(claro[i:])), None)
    if primero == 0:
        out["estado"] = "desde_el_inicio"
        return out
    out["estado"] = "cruce"
    out["n_previo"] = ns[primero - 1]
    la, lb = math.log(t_a[primero - 1] / t_b[primero - 1]), math.log(t_a[primero] / t_b[primero])
    if la != lb:
        frac = (0 - la) / (lb - la)
        out["n_interp"] = math.exp(math.log(ns[primero - 1]) + frac * (math.log(ns[primero]) - math.log(ns[primero - 1])))
    return out


def cruce_para(res, consulta, base):
    a, b = serie(res, "escalamiento", consulta, QT), serie(res, "escalamiento", consulta, base)
    comunes = sorted({r["N"] for r in a} & {r["N"] for r in b})
    da, db = {r["N"]: r for r in a}, {r["N"]: r for r in b}
    return calcular_cruce(comunes, [da[n]["media_us"] for n in comunes], [da[n]["ic95_us"] for n in comunes],
                          [db[n]["media_us"] for n in comunes], [db[n]["ic95_us"] for n in comunes])


#graficas
COLOR = {QT: "#1f77b4", LISTA: "#d62728", NUMPY: "#ff9f1c"}
MARCA = {QT: "o", LISTA: "s", NUMPY: "^"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 9})
    return plt


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


def _ejes_n(ax, ns):
    ns = sorted({int(n) for n in ns})
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([_fmt(n) for n in ns], rotation=45 if len(ns) > 6 else 0)
    ax.minorticks_off()


def _pie(fig, info, extra=""):
    p = info.get("parametros", {})
    texto = (f"Cada punto: media de R = {p.get('repeticiones', '?')} repeticiones (datos y consultas distintos en cada una); "
             f"donde hay barras de error, son el IC 95 % de la media. Q = {p.get('consultas', '?')} consultas por medicion "
             f"(la fuerza bruta de lista usa menos cuando N es grande). {extra}")
    texto = textwrap.fill(texto, width=int(fig.get_figwidth() * 15))
    fig.text(0.5, 0.008, texto, ha="center", va="bottom", fontsize=7.5, color="#444444")


def _guardar(fig, carpeta, nombre, info, extra=""):
    _pie(fig, info, extra)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    ruta = os.path.join(carpeta, nombre)
    fig.savefig(ruta, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  ", ruta)


def _leyenda_con_cruce(ax):
    ax.legend()


def grafica_tiempo(res, info, carpeta, consulta, titulo, radio):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 6))
    for metodo in METODOS:
        s = serie(res, "escalamiento", consulta, metodo)
        if s:
            ax.errorbar([r["N"] for r in s], [r["media_us"] for r in s], yerr=[r["ic95_us"] or 0 for r in s],
                        color=COLOR[metodo], marker=MARCA[metodo], capsize=3, linewidth=1.8, label=metodo)
    for base, estilo in ((LISTA, "--"), (NUMPY, ":")):
        c = cruce_para(res, consulta, base)
        valor = c["n_interp"] or (c["n_primero"] if c["estado"] == "desde_el_inicio" else None)
        if valor:
            signo = "<=" if c["estado"] == "desde_el_inicio" else "~"
            ax.axvline(valor, color=COLOR[base], linestyle=estilo, alpha=0.8,
                       label=f"Cruce con {base.lower()}: N {signo} {_fmt(round(valor))}")
    ax.set_yscale("log")
    _ejes_n(ax, [r["N"] for r in res if r["experimento"] == "escalamiento"])
    ax.set_xlabel("Numero de puntos de entrega N (ciudad fija de 20 km x 20 km)")
    ax.set_ylabel("Tiempo por consulta (µs)")
    ax.set_title(titulo + (f" ({radio:.0f} m)" if radio else ""))
    ax.legend()
    _guardar(fig, carpeta, f"tiempo_{consulta}_vs_N.png", info)


def grafica_aceleracion(res, info, carpeta):
    plt = _plt()
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, consulta, titulo in zip(ejes, ("radio", "vecino"), ("Busqueda en radio", "Vecino mas cercano")):
        a = {r["N"]: r["media_us"] for r in serie(res, "escalamiento", consulta, QT)}
        for base in (LISTA, NUMPY):
            b = serie(res, "escalamiento", consulta, base)
            ns = [r["N"] for r in b if r["N"] in a]
            ax.plot(ns, [next(r["media_us"] for r in b if r["N"] == n) / a[n] for n in ns], color=COLOR[base],
                    marker=MARCA[base], linewidth=1.8, label=f"{base} ÷ Quadtree")
        ax.axhline(1, color="black", linestyle=":", label="Igual de rapidos (=1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _ejes_n(ax, list(a))
        ax.set_xlabel("Numero de puntos de entrega N")
        ax.set_title(titulo)
        ax.legend()
    ejes[0].set_ylabel("Aceleracion (veces; >1 = el Quadtree es mas rapido)")
    fig.suptitle("¿Desde que N gana el Quadtree? Tiempo de la fuerza bruta ÷ tiempo del Quadtree", fontsize=13)
    _guardar(fig, carpeta, "aceleracion_vs_N.png", info)


def grafica_trabajo(res, info, carpeta):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for consulta, color, etiqueta in (("radio", "#1f77b4", "Quadtree: busqueda en radio"),
                                      ("vecino", "#2ca02c", "Quadtree: vecino mas cercano")):
        s = serie(res, "escalamiento", consulta, QT)
        ax1.plot([r["N"] for r in s], [r["distancias_promedio"] for r in s], color=color, marker="o", linewidth=1.8, label=etiqueta)
    ns = sorted({r["N"] for r in serie(res, "escalamiento", "radio", QT)})
    ax1.plot(ns, ns, color=COLOR[LISTA], linestyle="--", label="Fuerza bruta: N distancias")
    ax1.set_yscale("log")
    _ejes_n(ax1, ns)
    ax1.set_xlabel("Numero de puntos de entrega N")
    ax1.set_ylabel("Distancias calculadas por consulta")
    ax1.set_title("Trabajo por consulta (independiente de la maquina)")
    ax1.legend()
    s = serie(res, "escalamiento", "radio", QT)
    ax2.plot([r["N"] for r in s], [r["resultados_promedio"] for r in s], color="#9467bd", marker="o", linewidth=1.8)
    ax2.set_yscale("log")
    _ejes_n(ax2, ns)
    ax2.set_xlabel("Numero de puntos de entrega N")
    ax2.set_ylabel("Puntos encontrados por consulta (k)")
    ax2.set_title("Tamano de la respuesta: con la ciudad fija, k crece con N")
    _guardar(fig, carpeta, "trabajo_por_consulta.png", info)


def grafica_construccion(res, info, carpeta):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    s = serie(res, "escalamiento", "construccion", QT)
    ns = [r["N"] for r in s]
    ax1.errorbar(ns, [r["media_us"] / 1e6 for r in s], yerr=[(r["ic95_us"] or 0) / 1e6 for r in s],
                 color=COLOR[QT], marker="o", capsize=3, linewidth=1.8, label="Construir el Quadtree")
    n0, t0 = ns[len(ns) // 2], s[len(ns) // 2]["media_us"] / 1e6
    ref = [t0 * (n * math.log(n)) / (n0 * math.log(n0)) for n in ns]
    ax1.plot(ns, ref, color="#7f8c8d", linestyle=":", label="Referencia N log N (escalada)")
    ax1.set_yscale("log")
    _ejes_n(ax1, ns)
    ax1.set_xlabel("Numero de puntos de entrega N")
    ax1.set_ylabel("Tiempo de construccion (s)")
    ax1.set_title("Costo de construir el arbol (una sola vez)")
    ax1.legend()

    build = {r["N"]: r["media_us"] for r in s}
    for consulta, titulo, estilo in (("radio", "radio", "-"), ("vecino", "vecino mas cercano", "--")):
        a = {r["N"]: r["media_us"] for r in serie(res, "escalamiento", consulta, QT)}
        for base in (LISTA, NUMPY):
            b = {r["N"]: r["media_us"] for r in serie(res, "escalamiento", consulta, base)}
            puntos = [(n, build[n] / (b[n] - a[n])) for n in sorted(build) if n in a and n in b and b[n] > a[n]]
            if puntos:
                ax2.plot(*zip(*puntos), color=COLOR[base], linestyle=estilo, marker=MARCA[base],
                         label=f"{titulo} vs {base.lower()}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    _ejes_n(ax2, ns)
    ax2.set_xlabel("Numero de puntos de entrega N")
    ax2.set_ylabel("Consultas necesarias para recuperar el costo de construir")
    ax2.set_title("¿Cuantas consultas hacen rentable construir el arbol?")
    ax2.legend(fontsize=8)
    _guardar(fig, carpeta, "construccion_y_amortizacion.png", info)


def grafica_radio(res, info, carpeta):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    a = {r["radio_m"]: r for r in serie(res, "barrido_radio", "radio", QT)}
    for metodo in METODOS:
        s = serie(res, "barrido_radio", "radio", metodo)
        ax1.errorbar([r["radio_m"] for r in s], [r["media_us"] for r in s], yerr=[r["ic95_us"] or 0 for r in s],
                     color=COLOR[metodo], marker=MARCA[metodo], capsize=3, linewidth=1.8, label=metodo)
        if metodo != QT:
            ax2.plot([r["radio_m"] for r in s], [r["media_us"] / a[r["radio_m"]]["media_us"] for r in s],
                     color=COLOR[metodo], marker=MARCA[metodo], linewidth=1.8, label=f"{metodo} ÷ Quadtree")
    ax2.axhline(1, color="black", linestyle=":", label="Igual de rapidos (=1)")
    n_fijo = next(iter(a.values()))["N"]
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Radio de busqueda (m)")
        ax.legend()
    ax1.set_ylabel("Tiempo por consulta (µs)")
    ax1.set_title(f"Tiempo vs radio (N = {_fmt(n_fijo)})")
    ax2.set_ylabel("Aceleracion (veces; >1 = el Quadtree es mas rapido)")
    ax2.set_title("¿Hasta que radio compensa el Quadtree?")
    _guardar(fig, carpeta, "barrido_radio.png", info, "Experimento con N fijo.")


def grafica_dimension(res, info, carpeta):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    a = {r["k_dim"]: r for r in serie(res, "dimension", "vecino", QT)}
    for metodo in METODOS:
        s = serie(res, "dimension", "vecino", metodo)
        ax1.errorbar([r["k_dim"] for r in s], [r["media_us"] for r in s], yerr=[r["ic95_us"] or 0 for r in s],
                     color=COLOR[metodo], marker=MARCA[metodo], capsize=3, linewidth=1.8, label=metodo)
        if metodo != QT:
            ax2.plot([r["k_dim"] for r in s], [r["media_us"] / a[r["k_dim"]]["media_us"] for r in s],
                     color=COLOR[metodo], marker=MARCA[metodo], linewidth=1.8, label=f"{metodo} ÷ Quadtree")
    ax2.axhline(1, color="black", linestyle=":", label="Igual de rapidos (=1)")
    n_fijo = next(iter(a.values()))["N"]
    ks = sorted(a)
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("Dimension K (cada nodo tiene hasta 2^K hijos)")
        ax.legend()
    ax1.set_ylabel("Tiempo por consulta (µs)")
    ax1.set_title(f"Vecino mas cercano vs dimension (N = {_fmt(n_fijo)})")
    ax2.set_ylabel("Aceleracion (veces; >1 = el Quadtree es mas rapido)")
    ax2.set_title("Aceleracion del Quadtree segun la dimension K")
    _guardar(fig, carpeta, "dimension_k.png", info, "Datos uniformes en un hipercubo; experimento con N fijo.")


def _es(x, dec=1):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def texto_cruce(c, consulta_txt, base):
    ns = c["ns"]
    if c["estado"] == "nunca":
        return (f"- **{consulta_txt} vs {base}:** el Quadtree **no fue mas rapido en ningun tamano medido** "
                f"(N de {_fmt(ns[0])} a {_fmt(ns[-1])}).")
    if c["estado"] == "desde_el_inicio":
        return (f"- **{consulta_txt} vs {base}:** el Quadtree **ya fue mas rapido en el menor tamano medido** "
                f"(N = {_fmt(ns[0])}); el cruce esta en ese valor o por debajo y habria que medir tamanos menores.")
    claro = (f"Con los intervalos de confianza del 95 % sin solaparse, la ventaja es estadisticamente clara desde N = {_fmt(c['n_claro'])}."
             if c["n_claro"] else "Con los intervalos de confianza del 95 %, la ventaja no llega a ser estadisticamente clara en el rango medido.")
    interp = f" Interpolando en escala log-log entre N = {_fmt(c['n_previo'])} y N = {_fmt(c['n_primero'])}, las curvas se cruzan en **N ≈ {_fmt(round(c['n_interp']))}**." if c["n_interp"] else ""
    return (f"- **{consulta_txt} vs {base}:** el Quadtree pasa a ser mas rapido **desde N = {_fmt(c['n_primero'])}** "
            f"(primer tamano medido donde su media es menor y sigue siendolo en todos los tamanos mayores).{interp} {claro}")


def generar_discusion(res, info):
    p = info["parametros"]
    radio = p["radio"]
    lineas = ["# Discusion de resultados - Laboratorio 4 (Quadtree vs fuerza bruta)", "",
              f"> Generado automaticamente por `analisis.py` a partir de `crudo.csv` / `resumen.csv` "
              f"(perfil `{p.get('perfil')}`, R = {p['repeticiones']} repeticiones, Q = {_fmt(p['consultas'])} consultas, "
              f"radio = {radio:.0f} m). Los numeros son los de la maquina descrita en `entorno.json`.", ""]

    #respuesta
    lineas += ["## 1. ¿Desde que tamano de datos el Quadtree es mas rapido que la fuerza bruta?", ""]
    cruces = {}
    for consulta, texto in (("radio", f"Busqueda en radio de {radio:.0f} m"), ("vecino", "Vecino mas cercano")):
        for base in (LISTA, NUMPY):
            cruces[(consulta, base)] = cruce_para(res, consulta, base)
            lineas.append(texto_cruce(cruces[(consulta, base)], texto, base.lower()))
    lineas += ["",
               f"**Respuesta (linea base del enunciado, fuerza bruta con listas):** busqueda en radio -> "
               f"{_resumen_cruce(cruces[('radio', LISTA)])}; vecino mas cercano -> {_resumen_cruce(cruces[('vecino', LISTA)])}.",
               "", "La fuerza bruta vectorizada con NumPy es una linea base mucho mas exigente: hace el mismo "
               "recorrido completo pero en C. Por eso el cruce se desplaza hacia tamanos mayores "
               f"(radio: {_resumen_cruce(cruces[('radio', NUMPY)])}; vecino: {_resumen_cruce(cruces[('vecino', NUMPY)])}). "
               "**El punto de cruce no es una propiedad del Quadtree solo: depende de que tan buena sea la fuerza bruta con la que se compara.**", ""]

    #tabla en el N de referencia
    ns = sorted({r["N"] for r in res if r["experimento"] == "escalamiento"})
    n_ref = min(ns, key=lambda n: abs(n - 10_000))
    lineas += [f"## 2. Tiempos con N = {_fmt(n_ref)} puntos" + (" (el caso del enunciado)" if n_ref == 10_000 else ""), "",
               "| Consulta | Quadtree (µs) | Fuerza bruta lista (µs) | Fuerza bruta NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |",
               "|---|---|---|---|---|---|"]
    for consulta, nombre in (("radio", f"Radio {radio:.0f} m"), ("vecino", "Vecino mas cercano")):
        v = {m: next(r for r in serie(res, "escalamiento", consulta, m) if r["N"] == n_ref)["media_us"] for m in METODOS}
        lineas.append(f"| {nombre} | {_es(v[QT], 1)} | {_es(v[LISTA], 1)} | {_es(v[NUMPY], 1)} | "
                      f"{_es(v[LISTA] / v[QT], 1)}x | {_es(v[NUMPY] / v[QT], 1)}x |")
    trabajo = {c: next(r for r in serie(res, "escalamiento", c, QT) if r["N"] == n_ref) for c in ("radio", "vecino")}
    lineas += ["", f"Trabajo por consulta (determinista): el Quadtree calcula en promedio {_es(trabajo['radio']['distancias_promedio'], 0)} "
                   f"distancias en la busqueda por radio y {_es(trabajo['vecino']['distancias_promedio'], 0)} en el vecino mas cercano, "
                   f"frente a las {_fmt(n_ref)} de la fuerza bruta.", ""]

    #escalamiento
    pend = {(r["consulta"], r["metodo"], r["rango"]): r for r in _leer_pendientes(info)}
    lineas += ["## 3. Como escala cada metodo (pendiente log-log, N >= 100)", "",
               "Si el tiempo crece como N^k, la pendiente es k. La fuerza bruta deberia dar k cercano a 1; el Quadtree con vecino mas cercano, cercano a 0.", "",
               "| Consulta | Metodo | Pendiente | R² |", "|---|---|---|---|"]
    for consulta in ("radio", "vecino"):
        for m in METODOS:
            r = pend.get((consulta, m, "N >= 100"))
            if r:
                lineas.append(f"| {consulta} | {m} | {_es(float(r['pendiente']), 2)} | {_es(float(r['r2']), 3)} |")
    s_rad = serie(res, "escalamiento", "radio", QT)
    k_min, k_max = s_rad[0]["resultados_promedio"], s_rad[-1]["resultados_promedio"]
    lineas += ["", f"En la busqueda por radio el Quadtree no es plano: con la ciudad fija, al aumentar N sube la densidad y el numero de "
                   f"puntos encontrados k crece (de {_es(k_min, 1)} con N = {_fmt(s_rad[0]['N'])} a {_es(k_max, 1)} con N = {_fmt(s_rad[-1]['N'])}). "
                   "El costo del Quadtree es O(log N + k), y la fuerza bruta es O(N) siempre; la ventaja relativa es mayor cuando k es pequeno frente a N.", ""]

    #construccion
    lineas += ["## 4. Costo de construir el arbol", ""]
    sb = {r["N"]: r["media_us"] for r in serie(res, "escalamiento", "construccion", QT)}
    if n_ref in sb:
        linea = f"Con N = {_fmt(n_ref)}, construir el Quadtree tarda {_es(sb[n_ref] / 1e3, 1)} ms (una sola vez, porque los datos son estaticos)."
        recupera = []
        for consulta, nombre in (("radio", "busqueda por radio"), ("vecino", "vecino mas cercano")):
            a = next(r for r in serie(res, "escalamiento", consulta, QT) if r["N"] == n_ref)["media_us"]
            b = next(r for r in serie(res, "escalamiento", consulta, LISTA) if r["N"] == n_ref)["media_us"]
            if b > a:
                recupera.append(f"~{_es(sb[n_ref] / (b - a), 0)} consultas de {nombre}")
        if recupera:
            linea += " Frente a la fuerza bruta de lista, ese costo se recupera con " + " o ".join(recupera) + "."
        lineas.append(linea)
    lineas += ["Si solo se va a hacer una consulta, la fuerza bruta es mas barata; el arbol compensa cuando los datos estaticos se consultan muchas veces.", ""]

    #radio
    s_q = {r["radio_m"]: r for r in serie(res, "barrido_radio", "radio", QT)}
    if s_q:
        n_f = next(iter(s_q.values()))["N"]
        lineas += [f"## 5. Efecto del radio (N = {_fmt(n_f)})", "",
                   "| Radio (m) | Puntos encontrados | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Aceleracion vs NumPy |",
                   "|---|---|---|---|---|---|---|"]
        for rad in sorted(s_q):
            v = {m: next(r for r in serie(res, "barrido_radio", "radio", m) if r["radio_m"] == rad)["media_us"] for m in METODOS}
            lineas.append(f"| {_es(rad, 0)} | {_es(s_q[rad]['resultados_promedio'], 1)} | {_es(v[QT], 1)} | {_es(v[LISTA], 1)} | "
                          f"{_es(v[NUMPY], 1)} | {_es(v[LISTA] / v[QT], 1)}x | {_es(v[NUMPY] / v[QT], 1)}x |")
        lineas.append("")
        lineas.append("Cuanto mayor es el radio, mas puntos devuelve la consulta y menos se puede podar: el Quadtree se acerca a la fuerza bruta "
                      "(en el limite, un radio que cubre toda la ciudad devuelve los N puntos y ningun metodo puede evitar recorrerlos).")
        lineas.append("")

    #dimension
    s_k = {r["k_dim"]: r for r in serie(res, "dimension", "vecino", QT)}
    if s_k:
        n_f = next(iter(s_k.values()))["N"]
        lineas += [f"## 6. Efecto de la dimension K (vecino mas cercano, N = {_fmt(n_f)})", "",
                   "| K | Hijos posibles por nodo | Quadtree (µs) | Lista (µs) | NumPy (µs) | Aceleracion vs lista | Distancias calculadas por el Quadtree |",
                   "|---|---|---|---|---|---|---|"]
        for k in sorted(s_k):
            v = {m: next(r for r in serie(res, "dimension", "vecino", m) if r["k_dim"] == k)["media_us"] for m in METODOS}
            lineas.append(f"| {k} | {2 ** k} | {_es(v[QT], 1)} | {_es(v[LISTA], 1)} | {_es(v[NUMPY], 1)} | "
                          f"{_es(v[LISTA] / v[QT], 1)}x | {_es(s_k[k]['distancias_promedio'], 0)} de {_fmt(n_f)} |")
        lineas += ["", "La tabla muestra cuanta ventaja conserva el arbol en cada K; la ultima columna (distancias calculadas, una metrica determinista) indica cuanto trabajo evita. "
                       "La explicacion habitual, que este experimento no mide directamente, es que al subir K cada nodo se divide en 2^K regiones y, con un numero fijo de puntos, "
                       "la mayoria de las cajas quedan casi vacias, mientras que la distancia al vecino mas cercano se parece cada vez mas a la distancia a cualquier otro punto, "
                       "de modo que la poda descarta menos cajas (\"maldicion de la dimensionalidad\").", ""]

    lineas += ["## 7. Limitaciones", "",
               "- Python es interpretado: los tiempos absolutos y, sobre todo, el punto de cruce dependen de las constantes de la implementacion. "
               "Una implementacion en C++ tendria un cruce distinto, con las mismas tendencias.",
               "- Los datos son sinteticos (barrios gaussianos mas un fondo uniforme); en una ciudad real la distribucion cambia la altura del arbol y el trabajo por consulta.",
               "- Las consultas son ubicaciones uniformes sobre la ciudad, incluidas zonas sin entregas. Consultas concentradas en barrios densos devolverian mas puntos.",
               "- Una sola maquina y un solo sistema operativo; la variabilidad se controla con R repeticiones, no se elimina.",
               "- El punto de cruce se estima sobre la malla de N medida: su resolucion es la de esa malla (la interpolacion log-log es solo una estimacion entre dos tamanos medidos).",
               "- La fuerza bruta de lista usa menos consultas por medicion cuando N es grande (presupuesto de distancias), por lo que sus mediciones grandes tienen mas ruido.", ""]
    return "\n".join(lineas)


def _resumen_cruce(c):
    if c["estado"] == "nunca":
        return "sin cruce en el rango medido"
    if c["estado"] == "desde_el_inicio":
        return f"el Quadtree ya gana en N = {_fmt(c['ns'][0])} (cruce <= ese valor)"
    return f"desde N = {_fmt(c['n_primero'])}" + (f" (cruce ~ {_fmt(round(c['n_interp']))})" if c["n_interp"] else "")


def _leer_pendientes(info):
    ruta = info.get("_ruta_pendientes")
    if ruta and os.path.exists(ruta):
        with open(ruta, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


#procesamiento-programa principal
def procesar(salida, graficas, hacer_graficas=True):
    ruta_crudo = os.path.join(salida, "crudo.csv")
    ruta_resumen = os.path.join(salida, "resumen.csv")
    with open(os.path.join(salida, "entorno.json"), encoding="utf-8") as f:
        info = json.load(f)
    n = resumir_csv(ruta_crudo, ruta_resumen)
    print(f"resumen.csv: {n} grupos")
    res = cargar_resumen(ruta_resumen)
    ruta_pend = os.path.join(salida, "pendientes.csv")
    tabla_pendientes(res, ruta_pend)
    info["_ruta_pendientes"] = ruta_pend

    if hacer_graficas:
        os.makedirs(graficas, exist_ok=True)
        radio = info["parametros"]["radio"]
        print("Graficas:")
        grafica_tiempo(res, info, graficas, "radio", "Tiempo por consulta: busqueda en radio", radio)
        grafica_tiempo(res, info, graficas, "vecino", "Tiempo por consulta: vecino mas cercano", None)
        grafica_aceleracion(res, info, graficas)
        grafica_trabajo(res, info, graficas)
        grafica_construccion(res, info, graficas)
        if any(r["experimento"] == "barrido_radio" for r in res):
            grafica_radio(res, info, graficas)
        if any(r["experimento"] == "dimension" for r in res):
            grafica_dimension(res, info, graficas)

    texto = generar_discusion(res, info)
    with open(os.path.join(salida, "discusion.md"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\n" + texto)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Analisis de rendimiento del Quadtree (Laboratorio 4)")
    ap.add_argument("--perfil", choices=PERFILES, default="completo")
    ap.add_argument("--tamanos", type=int, nargs="+", help="valores de N para el experimento de escalamiento")
    ap.add_argument("--repeticiones", type=int, help="R, repeticiones por punto")
    ap.add_argument("--consultas", type=int, help="Q, consultas por medicion")
    ap.add_argument("--radio", type=float, help="radio de busqueda en metros (500 segun el enunciado)")
    ap.add_argument("--n-fijo", type=int, help="N de los experimentos de radio y dimension")
    ap.add_argument("--presupuesto", type=int, default=10_000_000,
                    help="tope aproximado de distancias por medicion de la fuerza bruta de lista")
    ap.add_argument("--semilla", default="lab4")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--graficas", default="graficas")
    ap.add_argument("--sin-graficas", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true", help="no mide: regenera resumen, graficas y discusion desde crudo.csv")
    args = ap.parse_args()

    if not args.solo_resumen:
        p = dict(PERFILES[args.perfil])
        for cli, clave in (("tamanos", "tamanos"), ("repeticiones", "repeticiones"), ("consultas", "consultas"),
                           ("radio", "radio"), ("n_fijo", "n_fijo")):
            if getattr(args, cli) is not None:
                p[clave] = getattr(args, cli)
        p["perfil"] = args.perfil
        os.makedirs(args.salida, exist_ok=True)
        if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
            print("AVISO: se detecto Google Colab; para el estudio use un equipo propio.\n")
        entorno = describir_entorno(args, p)

        #piloto
        pts = generar_puntos_ciudad(500, semilla="calentamiento")
        cons = generar_consultas(100, (0.0, 0.0), (CIUDAD, CIUDAD), "calentamiento-q")
        est = {QT: Quadtree(pts), LISTA: FuerzaBruta(pts), NUMPY: FuerzaBrutaNumpy(pts)}
        for e in est.values():
            _consultar_radio(e, cons, 500.0)
            _consultar_vecino(e, cons)

        inicio = time.perf_counter()
        with open(os.path.join(args.salida, "crudo.csv"), "w", newline="", encoding="utf-8") as archivo:
            reg = Registro(archivo)
            experimento_escalamiento(reg, args, p)
            experimento_radio(reg, args, p)
            experimento_dimension(reg, args, p)
        entorno["duracion_total_s"] = round(time.perf_counter() - inicio, 1)
        with open(os.path.join(args.salida, "entorno.json"), "w", encoding="utf-8") as f:
            json.dump(entorno, f, indent=2, ensure_ascii=False)
        print(f"\nMediciones terminadas en {entorno['duracion_total_s']} s.\n")

    procesar(args.salida, args.graficas, hacer_graficas=not args.sin_graficas)


if __name__ == "__main__":
    main()
