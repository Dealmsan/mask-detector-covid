"""
serial.py
Processa todas as imagens do dataset sequencialmente, uma por uma,
rodando a inferência real do YOLOv8n em cada uma.

Uso:
    python serial.py
"""

import os
import csv
import time
import sys
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset-covid-mask"
MODELO_PATH = "yolov8n.pt"
CONFIANCA   = 0.3
BATCH_SIZE  = 16
IMG_SIZE    = 320
SAIDA_CSV   = "resultados_serial.csv"
SAIDA_TEMPO = "tempo_serial.txt"
# ──────────────────────────────────────────────────────────────────────────────


def coletar_imagens(base_dir: str) -> list[str]:
    imagens = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imagens.append(os.path.join(root, f))
    return sorted(imagens)


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


def processar_batch(modelo: YOLO, caminhos: list[str]) -> list[dict]:
    resultados_yolo = modelo(
        caminhos, conf=CONFIANCA, classes=[0],
        verbose=False, imgsz=IMG_SIZE, stream=True,
    )
    resultados = []
    for r, caminho in zip(resultados_yolo, caminhos):
        contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
        for box in r.boxes:
            coords = list(map(int, box.xyxy[0].tolist()))
            contagem[classificar_deteccao(coords)] += 1
        contagem["total"]  = sum(contagem.values())
        contagem["imagem"] = str(Path(caminho).relative_to(DATASET_DIR))
        resultados.append(contagem)
    return resultados


def salvar_csv(resultados: list[dict]):
    campos = ["imagem", "with_mask", "without_mask", "mask_weared_incorrect", "total"]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"[CSV] Resultados salvos em: {SAIDA_CSV}")


def main():
    print(f"[MODELO] Carregando: {MODELO_PATH}")
    modelo = YOLO(MODELO_PATH)

    imagens = coletar_imagens(DATASET_DIR)
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em: {DATASET_DIR}")
        sys.exit(1)

    print(f"[SERIAL] Processando {len(imagens)} imagens em batches de {BATCH_SIZE}...")
    print("-" * 50)

    resultados = []
    batches = [imagens[i:i + BATCH_SIZE] for i in range(0, len(imagens), BATCH_SIZE)]
    inicio = time.perf_counter()

    for i, batch in enumerate(batches, 1):
        resultados.extend(processar_batch(modelo, batch))
        imgs_processadas = min(i * BATCH_SIZE, len(imagens))
        if i % 10 == 0 or imgs_processadas == len(imagens):
            decorrido = time.perf_counter() - inicio
            print(f"  Progresso: {imgs_processadas}/{len(imagens)} imagens "
                  f"({decorrido:.0f}s)", end="\r")

    tempo_total = time.perf_counter() - inicio
    print(f"\n[SERIAL] Concluído em {tempo_total:.2f}s ({tempo_total/60:.1f} min)")

    with open(SAIDA_TEMPO, "w") as f:
        f.write(f"{tempo_total:.4f}\n")
    print(f"[TEMPO]  Salvo em: {SAIDA_TEMPO}")

    salvar_csv(resultados)

    total_mask      = sum(r["with_mask"]             for r in resultados)
    total_no_mask   = sum(r["without_mask"]          for r in resultados)
    total_incorreta = sum(r["mask_weared_incorrect"] for r in resultados)
    total_rostos    = total_mask + total_no_mask + total_incorreta

    print("\n" + "=" * 50)
    print(f"  Imagens processadas       : {len(imagens)}")
    print(f"  Total de rostos detectados: {total_rostos}")
    if total_rostos > 0:
        print(f"  with_mask                 : {total_mask}  ({total_mask/total_rostos*100:.1f}%)")
        print(f"  without_mask              : {total_no_mask}  ({total_no_mask/total_rostos*100:.1f}%)")
        print(f"  mask_weared_incorrect     : {total_incorreta}  ({total_incorreta/total_rostos*100:.1f}%)")
    print(f"  Workers                   : 1 (serial)")
    print(f"  Tempo total               : {tempo_total:.2f}s ({tempo_total/60:.1f} min)")
    print(f"  Throughput                : {len(imagens)/tempo_total:.1f} imgs/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
