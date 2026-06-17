"""
paralelo.py
Etapa 3 — Processamento PARALELO da classificação de máscaras.

Lê o bboxes.json gerado pelo preparar.py e distribui a classificação
entre múltiplos processos. Testa automaticamente 2, 4, 8 e 12 workers.

O processamento é repetido REPETICOES vezes (mesmo valor do serial.py)
para garantir comparação justa no benchmark.

Uso:
    python paralelo.py

⚠️  Windows: o bloco `if __name__ == "__main__":` é obrigatório.
"""

import csv
import json
import time
import sys
import multiprocessing as mp
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
ARQUIVO_BBOXES          = "bboxes.json"
SAIDA_CSV               = "resultados_paralelo.csv"
SAIDA_TEMPOS            = "tempos_paralelos.csv"
SAIDA_TEMPO_SERIAL      = "tempo_serial.txt"
CONFIGURACOES_PROCESSOS = [2, 4, 8, 12]
TAMANHO_CHUNK           = 100
REPETICOES              = 500   # deve ser igual ao serial.py
# ──────────────────────────────────────────────────────────────────────────────


def classificar_deteccao(box) -> str:
    x1, y1, x2, y2 = box
    largura = x2 - x1
    altura  = y2 - y1
    if largura == 0:
        return "with_mask"
    proporcao = altura / largura
    if proporcao > 1.6:
        return "without_mask"
    elif proporcao < 0.9:
        return "mask_weared_incorrect"
    else:
        return "with_mask"


def processar_chunk(args: tuple) -> list[dict]:
    """
    Executado em cada processo filho.
    Recebe (itens, repeticoes) e classifica cada bbox REPETICOES vezes.
    """
    itens, repeticoes = args
    resultados = []

    for _ in range(repeticoes):
        resultados = []
        for nome, bboxes in itens:
            contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
            for box in bboxes:
                contagem[classificar_deteccao(box)] += 1
            contagem["total"]  = sum(contagem.values())
            contagem["imagem"] = nome
            resultados.append(contagem)

    return resultados


def dividir_chunks(itens: list, n: int) -> list[list]:
    """Divide itens em n partes o mais iguais possível."""
    tam = max(1, len(itens) // n)
    return [itens[i:i + tam] for i in range(0, len(itens), tam)]


def ler_tempo_serial() -> float | None:
    try:
        return float(Path(SAIDA_TEMPO_SERIAL).read_text().strip())
    except Exception:
        return None


def salvar_csv_resultados(resultados: list[dict]):
    campos = ["imagem", "with_mask", "without_mask", "mask_weared_incorrect", "total"]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"[CSV] Resultados salvos em: {SAIDA_CSV}")


def salvar_csv_tempos(registros: list[dict]):
    campos = ["workers", "imagens_por_worker", "tempo_s", "speedup", "eficiencia_pct"]
    with open(SAIDA_TEMPOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)
    print(f"[CSV] Tempos salvos em: {SAIDA_TEMPOS}")


def main():
    if not Path(ARQUIVO_BBOXES).exists():
        print(f"[ERRO] '{ARQUIVO_BBOXES}' não encontrado.")
        print("       Execute primeiro: python preparar.py")
        sys.exit(1)

    print(f"[PARALELO] Carregando bounding boxes de: {ARQUIVO_BBOXES}")
    with open(ARQUIVO_BBOXES, encoding="utf-8") as f:
        bboxes = json.load(f)

    itens    = list(bboxes.items())
    t_serial = ler_tempo_serial()

    if t_serial:
        print(f"[REF]  Tempo serial: {t_serial:.4f}s")
    else:
        print("[AVISO] tempo_serial.txt não encontrado. Execute serial.py primeiro.\n")

    print(f"[PARALELO] {len(itens)} imagens x {REPETICOES} repetições | testando: {CONFIGURACOES_PROCESSOS} workers\n")

    registros_tempo    = []
    ultimos_resultados = []

    for n_proc in CONFIGURACOES_PROCESSOS:
        n_proc_real   = min(n_proc, len(itens))
        imgs_por_proc = len(itens) // n_proc_real

        # Divide itens entre os processos — cada processo recebe seu lote + nº de repetições
        lotes = dividir_chunks(itens, n_proc_real)
        args  = [(lote, REPETICOES) for lote in lotes]

        print(f"─── {n_proc_real} workers (~{imgs_por_proc} imgs cada, {REPETICOES} repetições) ───")
        inicio = time.perf_counter()

        with mp.Pool(processes=n_proc_real) as pool:
            lotes_resultado = pool.map(processar_chunk, args)

        tempo = time.perf_counter() - inicio

        resultados         = [item for sub in lotes_resultado for item in sub]
        ultimos_resultados = resultados

        speedup    = round(t_serial / tempo, 4) if t_serial else 0.0
        eficiencia = round(speedup / n_proc_real * 100, 2) if t_serial else 0.0

        print(f"    Tempo      : {tempo:.4f}s")
        if t_serial:
            print(f"    Speedup    : {speedup:.2f}x")
            print(f"    Eficiência : {eficiencia:.1f}%")
        print()

        registros_tempo.append({
            "workers":            n_proc_real,
            "imagens_por_worker": imgs_por_proc,
            "tempo_s":            round(tempo, 6),
            "speedup":            speedup,
            "eficiencia_pct":     eficiencia,
        })

    salvar_csv_resultados(ultimos_resultados)
    salvar_csv_tempos(registros_tempo)

    print("\n" + "=" * 55)
    print(f"{'Workers':>10} {'Tempo (s)':>12} {'Speedup':>10} {'Eficiência':>12}")
    print("-" * 55)
    if t_serial:
        print(f"{'1 (serial)':>10} {t_serial:>12.4f} {'1.00x':>10} {'100.0%':>12}")
    for r in registros_tempo:
        print(f"{r['workers']:>10} {r['tempo_s']:>12.4f} {r['speedup']:>9.2f}x {r['eficiencia_pct']:>11.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
