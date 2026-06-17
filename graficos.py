"""
graficos.py
Gera gráficos de desempenho e distribuição de classes a partir dos CSVs gerados.

Gráficos gerados:
  1. grafico_speedup.png     — tempo, speedup e eficiência por número de workers
  2. grafico_classes.png     — distribuição das 3 classes detectadas

Uso:
    python graficos.py
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
TEMPO_SERIAL_TXT  = "tempo_serial.txt"
TEMPOS_CSV        = "tempos_paralelos.csv"
RESULTADOS_CSV    = "resultados_serial.csv"
SAIDA_SPEEDUP     = "grafico_speedup.png"
SAIDA_CLASSES     = "grafico_classes.png"

CORES_WORKERS = ["#6366f1", "#22c55e", "#f97316", "#ef4444"]
# ──────────────────────────────────────────────────────────────────────────────


def ler_tempo_serial():
    try:
        return float(Path(TEMPO_SERIAL_TXT).read_text().strip())
    except Exception:
        return None


def ler_tempos_paralelos():
    registros = []
    try:
        with open(TEMPOS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                registros.append({
                    "workers":        int(row["workers"]),
                    "tempo_s":        float(row["tempo_s"]),
                    "speedup":        float(row["speedup"]),
                    "eficiencia_pct": float(row["eficiencia_pct"]),
                })
    except Exception as e:
        print(f"[ERRO] Não foi possível ler {TEMPOS_CSV}: {e}")
    return registros


def ler_resultados():
    totais = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
    try:
        with open(RESULTADOS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in totais:
                    totais[k] += int(row.get(k, 0))
    except Exception as e:
        print(f"[ERRO] Não foi possível ler {RESULTADOS_CSV}: {e}")
    return totais


def grafico_speedup(t_serial, registros):
    """Gráfico com 3 painéis: tempo, speedup e eficiência."""
    workers    = [r["workers"]        for r in registros]
    tempos     = [r["tempo_s"]        for r in registros]
    speedups   = [r["speedup"]        for r in registros]
    eficiencias = [r["eficiencia_pct"] for r in registros]

    todos_workers = [1] + workers
    todos_tempos  = [t_serial] + tempos if t_serial else tempos

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Benchmark — Processamento Paralelo de Detecção de Máscaras",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Painel 1: Tempo de execução ──────────────────────────────────────────
    cores_tempo = ["#6366f1"] + CORES_WORKERS
    bars = ax1.bar(
        [f"{w}w" for w in todos_workers],
        todos_tempos,
        color=cores_tempo[:len(todos_workers)],
        edgecolor="white", linewidth=0.8,
    )
    ax1.set_title("Tempo de Execução", fontweight="bold")
    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Tempo (segundos)")
    for bar, v in zip(bars, todos_tempos):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + max(todos_tempos) * 0.01,
                 f"{v:.2f}s", ha="center", va="bottom", fontsize=9)

    # ── Painel 2: Speedup ────────────────────────────────────────────────────
    ax2.plot([1] + workers, [1.0] + speedups,
             marker="o", color="#6366f1", linewidth=2, markersize=7, label="Real")
    ax2.plot([1] + workers, [1] + workers,
             linestyle="--", color="#9ca3af", linewidth=1, label="Ideal")
    ax2.set_title("Speedup", fontweight="bold")
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Speedup (x)")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for x, y in zip(workers, speedups):
        ax2.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    # ── Painel 3: Eficiência ─────────────────────────────────────────────────
    bars3 = ax3.bar(
        [str(w) for w in workers],
        eficiencias,
        color=CORES_WORKERS[:len(workers)],
        edgecolor="white", linewidth=0.8,
    )
    ax3.axhline(y=100, linestyle="--", color="#9ca3af", linewidth=1, label="Ideal (100%)")
    ax3.set_title("Eficiência por Worker", fontweight="bold")
    ax3.set_xlabel("Workers")
    ax3.set_ylabel("Eficiência (%)")
    ax3.set_ylim(0, 115)
    ax3.legend(fontsize=8)
    for bar, v in zip(bars3, eficiencias):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(SAIDA_SPEEDUP, dpi=150, bbox_inches="tight")
    print(f"[GRÁFICO] Salvo: {SAIDA_SPEEDUP}")
    plt.close()


def grafico_classes(totais):
    """Gráfico de pizza + barras com distribuição das 3 classes."""
    total = sum(totais.values())
    if total == 0:
        print("[AVISO] Nenhuma detecção encontrada no CSV.")
        return

    LABELS = {
        "with_mask":             "Máscara correta",
        "without_mask":          "Sem máscara",
        "mask_weared_incorrect": "Máscara incorreta",
    }
    CORES = {
        "with_mask":             "#22c55e",
        "without_mask":          "#ef4444",
        "mask_weared_incorrect": "#eab308",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição de Detecções por Classe — 13.100 imagens",
                 fontsize=13, fontweight="bold")

    # Pizza
    labels_fmt = [
        f"{LABELS[k]}\n{v:,} ({v/total*100:.1f}%)"
        for k, v in totais.items()
    ]
    ax1.pie(
        totais.values(),
        labels=labels_fmt,
        colors=[CORES[k] for k in totais],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    ax1.set_title("Proporção por classe", fontweight="bold")

    # Barras
    nomes  = [LABELS[k] for k in totais]
    valores = list(totais.values())
    cores  = [CORES[k] for k in totais]

    bars = ax2.bar(nomes, valores, color=cores, edgecolor="white", linewidth=0.8)
    ax2.set_title("Total de detecções por classe", fontweight="bold")
    ax2.set_ylabel("Quantidade de rostos detectados")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, v in zip(bars, valores):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(valores) * 0.01,
                 f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(SAIDA_CLASSES, dpi=150, bbox_inches="tight")
    print(f"[GRÁFICO] Salvo: {SAIDA_CLASSES}")
    plt.close()


def main():
    t_serial  = ler_tempo_serial()
    registros = ler_tempos_paralelos()
    totais    = ler_resultados()

    if not registros:
        print("[ERRO] Execute paralelo.py primeiro para gerar tempos_paralelos.csv")
        return

    if not t_serial:
        print("[ERRO] Execute serial.py primeiro para gerar tempo_serial.txt")
        return

    print(f"[INFO] Tempo serial    : {t_serial:.4f}s")
    print(f"[INFO] Configs paralelo: {[r['workers'] for r in registros]} workers")
    print(f"[INFO] Total detecções : {sum(totais.values()):,}")
    print()

    grafico_speedup(t_serial, registros)
    grafico_classes(totais)

    print("\n✅ Gráficos gerados com sucesso!")


if __name__ == "__main__":
    main()