# Planificación Multi-Agente con Partición Espacial, A*/D* Lite y Aprendizaje por Refuerzo

Optimización de rutas autónomas para recolección de residuos

Autores:
* José Manuel Sanchez Perez (A01178230)
* Sergio Rodríguez Pérez (A00838856)
* Grecia Klarissa Saucedo Sandoval (A00839374)
* Luis Eduardo Cantú Leyva (A00840016)

Tecnológico de Monterrey
TC2038 — Analysis and Design of Advanced Algorithms
Diciembre 2025

---

## 1. Descripción general del proyecto

Los métodos tradicionales de planeación de rutas para múltiples agentes dependen de un planificador centralizado que escala de manera deficiente al incrementar el número de agentes o la complejidad del entorno. Esto produce trayectorias redundantes, conflictos entre vehículos y altos tiempos de cómputo, especialmente en escenarios dinámicos.

Este proyecto implementa un simulador multi-agente donde una flota de camiones recolectores opera de forma autónoma en un mapa discretizado. Cada camión selecciona objetivos, calcula rutas, gestiona energía y capacidad, y coopera de manera implícita mediante mecanismos de partición espacial y reserva de contenedores.

El sistema integra tres enfoques principales:

1. Partición geométrica del espacio con diagramas de Voronoi.
2. Planeación de rutas con A* y D* Lite para replaneación incremental.
3. Mecanismos de coordinación basados en aprendizaje por refuerzo.

El repositorio contiene el código del simulador, experimentos de rendimiento, gráficas de desempeño y una versión del póster académico presentado.

---

## 2. Planteamiento del problema

La planeación multi-agente enfrenta tres retos fundamentales:

1. Escalabilidad: los algoritmos tradicionales incrementan su complejidad al aumentar el número de agentes.
2. Conflictos: múltiples agentes pueden elegir rutas redundantes o colisionar cuando comparten objetivos.
3. Adaptación: cambios en el entorno obligan a recalcular rutas de manera costosa.

Dado un conjunto de camiones, contenedores y depósitos en un mapa modelado como grafo dirigido y ponderado, se requiere una política que asigne contenedores y rutas a cada agente, minimizando:

* Distancia recorrida
* Solapamiento entre rutas
* Conflictos y colisiones
* Tiempo total de planificación

Además, la solución debe mantener la operación aun cuando el entorno cambia o los recursos de los agentes (energía, carga, disponibilidad) se agotan.

---

## 3. Pregunta de investigación

¿Puede una arquitectura híbrida basada en partición espacial (Voronoi), planeación informada (A*/D* Lite) y coordinación mediante aprendizaje por refuerzo mejorar la eficiencia, reducir conflictos y escalar mejor que un planificador multi-agente convencional sin partición ni RL?

---

## 4. Objetivos

### Objetivo general

Diseñar e implementar un planificador multi-agente basado en partición geométrica, A*/D* Lite y aprendizaje por refuerzo para optimizar rutas de recolección autónoma.

### Objetivos específicos

* Reducir los tiempos de planificación comparado con enfoques sin partición espacial.
* Disminuir el solapamiento de rutas y la tasa de conflictos.
* Incorporar D* Lite para re-planificación local ante cambios del entorno.
* Evaluar la escalabilidad del sistema respecto al tamaño del mapa y número de agentes.

---

## 5. Diseño del sistema

El sistema sigue una arquitectura modular compuesta por los siguientes componentes:

### 5.1 Modelado del entorno

El mapa se representa como un grafo dirigido cuyos nodos son celdas accesibles y cuyas aristas corresponden a movimientos válidos. Incluye:

* Obstáculos
* Camiones
* Contenedores
* Depósitos

### 5.2 Partición espacial mediante diagramas de Voronoi

Se generan regiones de Voronoi a partir de las posiciones base de los depósitos. Cada camión atiende prioritariamente contenedores dentro de su región, reduciendo competencia y mejorando la distribución de carga.

### 5.3 Planeación con A*

Se utiliza A* para calcular rutas iniciales hacia contenedores o depósitos.
La heurística empleada es la distancia Manhattan.

### 5.4 Replanificación dinámica con D* Lite

D* Lite permite actualizar rutas cuando cambian los costos o aparecen restricciones adicionales.
Reduce el costo computacional en comparación con recalcular rutas completas.

### 5.5 Máquina de estados del camión

El comportamiento del agente se estructura mediante los estados:

`idle → to_bin → collecting → to_depot → idle`

Las transiciones dependen de:

* Nivel del contenedor
* Capacidad disponible del camión
* Nivel de energía
* Reservas activas del contenedor

### 5.6 Protocolo de reserva de contenedores

Cada contenedor posee:

* `reserved_by`
* `reserved_until`

Esto evita que varios camiones compitan por el mismo objetivo, reduce conflictos y mejora la eficiencia global.

### 5.7 Coordinación mediante aprendizaje por refuerzo

El simulador registra métricas por episodio para evaluar eficiencia:

* Botes recolectados
* Distancia recorrida
* Eficiencia (bins/distance)
* Desempeño individual por camión

El RL actúa sobre decisiones de alto nivel, apoyándose en rutas razonables calculadas por A*/D*.

---

## 6. Complejidad algorítmica

### A*

* Tiempo: O(E log V)
* Espacio: O(V)

### D* Lite

* Aplica actualizaciones incrementales.
* En entornos con cambios locales, su costo es menor que recalcular rutas completas.

### Voronoi

* Construcción típica: O(n log n)
* Reduce búsqueda a subproblemas locales.

### Aprendizaje por refuerzo

* Costo dependiente del número de episodios y del tamaño del espacio de estados.
* Al operar sobre rutas ya razonables, el RL evita aprender navegación de bajo nivel.

---

## 7. Resultados experimentales

Los resultados experimentales se generaron en el simulador propuesto y se incluyen en la carpeta `performance_plots/`.

### 7.1 Gráfica de progreso del episodio

Muestra la relación entre botes recolectados y distancia acumulada a lo largo de 1000 pasos.

### 7.2 Eficiencia por camión

La eficiencia (botes/distancia) presenta diferencias significativas entre agentes, mostrando especialización de rutas según región Voronoi.

### 7.3 Desempeño global

Incluye total de botes recolectados y distancia recorrida por la flota.

### 7.4 Mapa de calor de posiciones

Ilustra las zonas del mapa más transitadas por los camiones.

### 7.5 Desempeño individual

Muestra carga residual, botes recolectados y distancia recorrida por agente.

### 7.6 Estadística agregada

En 1001 episodios registrados:

* Eficiencia promedio: 0.136 botes/distancia
* Distancia total por episodio: ~316 unidades
* Comportamiento estable del planificador con partición espacial

Los archivos incluidos son:

* `combined_performance.png`
* `bins_progress.png`
* `truck_efficiency.png`
* `global_performance.png`
* `movement_heatmap.png`
* `truck_individual_performance.png`
* `garbage_sim.gif`

---

## 8. Ejecución del proyecto

### Requisitos

```
numpy
scipy
matplotlib
pillow
agentpy
psutil
memory-profiler
```

### Ejecución básica

```
python garbage_collection_agentpy.py --demo
```

Genera:

* Simulación completa
* GIF en `performance_plots/garbage_sim.gif`
* Gráficas en `performance_plots/`

### Benchmarks

```
python garbage_collection_agentpy.py --benchmark
```

Produce los archivos `benchmark.csv` y `benchmark_all.csv`.

Los resultados se grafican de la siguiente manera

```
python plot_benchmarks.py
```

Produciendo los archivos `memory_usage_all.png`, `runtime_all.png` y `speedup_all.png`.

### Pruebas unitarias

```
python garbage_collection_agentpy.py --run-tests
```

Pruebas incluidas para:

* A*
* Voronoi KDTree
* D* Lite

---

## 9. Estructura del repositorio

```
Midterm2/
│-- README.md
│-- garbage_collection_agentpy.py
│-- plot_benchmarks.py
│-- performance_plots/
│   ├── bins_progress.png
│   ├── combined_performance.png
│   ├── global_performance.png
│   ├── truck_efficiency.png
│   ├── movement_heatmap.png
│   ├── truck_individual_performance.png
│   └── garbage_sim.gif
│-- benchmark_plots/
│   └── benchmark.csv
│   └── benchmark_all.csv
│   └── line_memory.csv
│   └── line_runtime.csv
│   └── line_speedup.csv
│-- poster/
│   └── posterMidterm2.pdf
│-- requirements.txt
```

---

## 10. Conclusiones

El enfoque híbrido propuesto demuestra que:

* La partición espacial asigna eficazmente responsabilidad por zonas, reduciendo competencia.
* A* proporciona rutas iniciales eficientes, mientras que D* Lite permite adaptación local sin recomputar caminos completos.
* El protocolo de reservas disminuye conflictos entre agentes.
* El aprendizaje por refuerzo permite ajustar decisiones de alto nivel basadas en métricas de desempeño.

Los resultados muestran eficiencia estable, baja tasa de conflictos y comportamiento escalable en diferentes configuraciones del simulador.

---

## 11. Trabajo futuro

* Integración de un módulo de coordinación explícita entre agentes.
* Políticas de RL jerárquico o compartido entre camiones.
* Evaluación en mapas urbanos reales o datos sintéticos más complejos.
* Paralelización en GPU de los cálculos de A* y D* Lite.

---

## 12. Referencias

Koenig, S., & Likhachev, M. (2002). D* Lite. Proceedings of AAAI.
Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). Spatial Tessellations: Concepts and Applications of Voronoi Diagrams.
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.