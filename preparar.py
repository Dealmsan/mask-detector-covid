"""
preparar.py
Etapa 1 — roda o YOLOv8n em todas as imagens UMA ÚNICA VEZ e salva
os bounding boxes detectados em bboxes.json.

Após isso, serial.py e paralelo.py não precisam mais chamar o YOLO —
apenas leem o bboxes.json e fazem a classificação, que é onde o
paralelismo realmente faz diferença.

Uso:
    python preparar.py
"""

import os
import json
import time
import sys
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset-covid-mask"
MODELO_PATH  = "yolov8n.pt"
CONFIANCA    = 0.3
IMG_SIZE     = 320
BATCH_SIZE   = 16
SAIDA_BBOXES = "bboxes.json"
# ──────────────────────────────────────────────────────────────────────────────


def coletar_imagens(base_dir: str) -> list[str]:
    imagens = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imagens.append(os.path.join(root, f))
    return sorted(imagens)


def main():
    print(f"[MODELO] Carregando: {MODELO_PATH}")
    modelo = YOLO(MODELO_PATH)

    imagens = coletar_imagens(DATASET_DIR)
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em: {DATASET_DIR}")
        sys.exit(1)

    print(f"[PREPARAR] Detectando pessoas em {len(imagens)} imagens...")
    print(f"           Batch size: {BATCH_SIZE} | Resolução: {IMG_SIZE}px")
    print("-" * 50)

    bboxes = {}   # { "nome_imagem.jpg": [[x1,y1,x2,y2], ...] }
    batches = [imagens[i:i + BATCH_SIZE] for i in range(0, len(imagens), BATCH_SIZE)]
    inicio = time.perf_counter()

    for i, batch in enumerate(batches, 1):
        resultados = modelo(
            batch,
            conf=CONFIANCA,
            classes=[0],
            verbose=False,
            imgsz=IMG_SIZE,
            stream=True,
        )
        for r, caminho in zip(resultados, batch):
            nome = str(Path(caminho).relative_to(DATASET_DIR))
            bboxes[nome] = [
                list(map(int, box.xyxy[0].tolist()))
                for box in r.boxes
            ]

        imgs_feitas = min(i * BATCH_SIZE, len(imagens))
        if i % 10 == 0 or imgs_feitas == len(imagens):
            print(f"  Progresso: {imgs_feitas}/{len(imagens)} imagens", end="\r")

    tempo = time.perf_counter() - inicio

    with open(SAIDA_BBOXES, "w", encoding="utf-8") as f:
        json.dump(bboxes, f, ensure_ascii=False)

    total_deteccoes = sum(len(v) for v in bboxes.values())
    print(f"\n[PREPARAR] Concluído em {tempo:.2f}s")
    print(f"[JSON]     Salvo em: {SAIDA_BBOXES}")
    print(f"\n{'='*50}")
    print(f"  Imagens processadas : {len(imagens)}")
    print(f"  Total de detecções  : {total_deteccoes}")
    print(f"  Tempo YOLO          : {tempo:.2f}s  (não será contado no serial/paralelo)")
    print(f"{'='*50}")
    print(f"\n✅ Agora execute: python serial.py")


if __name__ == "__main__":
    main()
