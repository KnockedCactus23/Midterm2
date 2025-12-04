import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

# Cargar CSV de benchmarks
bench = pd.read_csv("benchmark_plots/benchmark_all.csv")

# Calcular speedup vs A*
astar_avg = bench[bench["planner"] == "A*"]["plan_time_s"].mean()
bench["speedup_vs_astar"] = astar_avg / bench["plan_time_s"]

planner_colors = {"A*": "green", "D*": "blue", "Voronoi": "orange"}
planner_marker = {"A*": "o", "D*": "s", "Voronoi": "D"}

# Tiempo de ejecución para cada algoritmo
plt.figure(figsize=(7,5))
for planner in bench["planner"].unique():
    sub = bench[bench["planner"] == planner]
    plt.plot(
        sub["rep"], sub["plan_time_s"],
        label=f"{planner}",
        linewidth=2.5,
        marker=planner_marker[planner],
        color=planner_colors[planner]
    )

plt.title("Runtime per Planning Algorithm")
plt.xlabel("Benchmark Repetition")
plt.ylabel("Planning Time (s)")
plt.legend(title="Algorithm")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_plots/line_runtime.png", dpi=300)
print("Generated: benchmark_plots/line_runtime.png")

# Uso de memoria para cada algoritmo
plt.figure(figsize=(7,5))
for planner in bench["planner"].unique():
    sub = bench[bench["planner"] == planner]
    plt.plot(
        sub["rep"], sub["peak_mem_mb"],
        label=f"{planner}",
        linewidth=2.5,
        marker=planner_marker[planner],
        color=planner_colors[planner]
    )

plt.title("Peak Memory Usage per Algorithm")
plt.xlabel("Benchmark Repetition")
plt.ylabel("Memory (MB)")
plt.legend(title="Algorithm")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_plots/line_memory.png", dpi=300)
print("Generated: benchmark_plots/line_memory.png")

# Speedup relativo a A*
plt.figure(figsize=(7,5))

for planner in bench["planner"].unique():
    if planner == "A*":
        continue
    sub = bench[bench["planner"] == planner]
    plt.plot(
        sub["rep"], sub["speedup_vs_astar"],
        label=f"{planner} vs A*",
        linewidth=2.5,
        marker=planner_marker[planner],
        color=planner_colors[planner]
    )

plt.title("Speedup Relative to A*")
plt.xlabel("Benchmark Repetition")
plt.ylabel("Speedup Factor")
plt.legend(title="Algorithm")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_plots/line_speedup.png", dpi=300)
print("Generated: benchmark_plots/line_speedup.png")

print("\n All plots created successfully!")