"""
serial.py
Etapa 2 — Processamento SERIAL da classificação de máscaras.

Lê o bboxes.json gerado pelo preparar.py e classifica cada bounding box
em with_mask / without_mask / mask_weared_incorrect, uma imagem por vez.

Para gerar carga mensurável de CPU (necessária para benchmark),
o processamento é repetido REPETICOES vezes — técnica padrão em benchmarks
quando a tarefa individual é muito rápida para ser medida com precisão.

Uso:
    python serial.py
"""

import csv
import json
import time
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
ARQUIVO_BBOXES = "bboxes.json"
SAIDA_CSV      = "resultados_serial.csv"
SAIDA_TEMPO    = "tempo_serial.txt"
REPETICOES     = 500   # repete o processamento para gerar carga mensurável
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


def processar_imagem(nome: str, bboxes: list[list]) -> dict:
    contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
    for box in bboxes:
        contagem[classificar_deteccao(box)] += 1
    contagem["total"]  = sum(contagem.values())
    contagem["imagem"] = nome
    return contagem


def salvar_csv(resultados: list[dict]):
    campos = ["imagem", "with_mask", "without_mask", "mask_weared_incorrect", "total"]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"[CSV] Resultados salvos em: {SAIDA_CSV}")


def main():
    if not Path(ARQUIVO_BBOXES).exists():
        print(f"[ERRO] '{ARQUIVO_BBOXES}' não encontrado.")
        print("       Execute primeiro: python preparar.py")
        sys.exit(1)

    print(f"[SERIAL] Carregando bounding boxes de: {ARQUIVO_BBOXES}")
    with open(ARQUIVO_BBOXES, encoding="utf-8") as f:
        bboxes = json.load(f)

    itens  = list(bboxes.items())
    n_imgs = len(itens)
    total_operacoes = n_imgs * REPETICOES

    print(f"[SERIAL] {n_imgs} imagens x {REPETICOES} repetições = {total_operacoes:,} operações")
    print(f"[SERIAL] Classificando sequencialmente...")
    print("-" * 50)

    # Guarda só o último resultado (para o CSV)
    resultados = []
    inicio = time.perf_counter()

    for rep in range(REPETICOES):
        resultados = []
        for nome, boxes in itens:
            resultado = processar_imagem(nome, boxes)
            resultados.append(resultado)

        if (rep + 1) % 50 == 0 or (rep + 1) == REPETICOES:
            print(f"  Repetição: {rep+1}/{REPETICOES}", end="\r")

    tempo_total = time.perf_counter() - inicio
    print(f"\n[SERIAL] Concluído em {tempo_total:.4f}s")

    with open(SAIDA_TEMPO, "w") as f:
        f.write(f"{tempo_total:.6f}\n")
    print(f"[TEMPO]  Salvo em: {SAIDA_TEMPO}")

    salvar_csv(resultados)

    total_mask      = sum(r["with_mask"]             for r in resultados)
    total_no_mask   = sum(r["without_mask"]          for r in resultados)
    total_incorreta = sum(r["mask_weared_incorrect"] for r in resultados)
    total_rostos    = total_mask + total_no_mask + total_incorreta

    print("\n" + "=" * 50)
    print(f"  Imagens no dataset        : {n_imgs}")
    print(f"  Repetições                : {REPETICOES}")
    print(f"  Total de operações        : {total_operacoes:,}")
    print(f"  Total de rostos           : {total_rostos}")
    if total_rostos > 0:
        print(f"  with_mask                 : {total_mask}  ({total_mask/total_rostos*100:.1f}%)")
        print(f"  without_mask              : {total_no_mask}  ({total_no_mask/total_rostos*100:.1f}%)")
        print(f"  mask_weared_incorrect     : {total_incorreta}  ({total_incorreta/total_rostos*100:.1f}%)")
    print(f"  Workers                   : 1 (serial)")
    print(f"  Tempo total               : {tempo_total:.4f}s")
    print(f"  Throughput                : {total_operacoes/tempo_total:,.0f} ops/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
