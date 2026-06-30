"""
paralelo.py
Processa todas as imagens do dataset em paralelo, rodando a inferência
real do YOLOv8n distribuída entre múltiplos processos.

OTIMIZAÇÕES aplicadas em relação à versão anterior:

1. CHUNKS MENORES (8 imagens em vez de 64)
   Com chunks grandes, poucos chunks ficam disponíveis por worker
   (ex: 13100/64 = 205 chunks ÷ 12 workers = ~17 chunks cada), o que
   prejudica o balanceamento dinâmico do imap_unordered — um worker
   que pega chunks "mais pesados" (imagens com mais rostos) atrasa
   todo o conjunto. Com chunks de 8, há ~1637 chunks no total, dando
   margem larga para o Pool redistribuir trabalho dinamicamente.

2. THREADS POR PROCESSO CALCULADAS DINAMICAMENTE
   Em vez de fixar 1 thread (que deixa núcleos ociosos com poucos
   workers) ou usar o padrão do sistema (que causa contenção com
   muitos workers), divide os núcleos disponíveis pelo número de
   processos: threads_por_processo = cpu_count() // n_workers.

3. BATCH_SIZE MENOR (4 em vez de 16) DENTRO DO CHUNK
   Batches menores reduzem a latência entre uma imagem processada
   e a próxima ficar disponível para reportar progresso e para o
   worker pegar a próxima tarefa do Pool — melhora granularidade.

Uso:
    python paralelo.py

⚠️  Windows: o bloco `if __name__ == "__main__":` é obrigatório.
"""

import os
import csv
import time
import sys
import multiprocessing as mp
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR             = "dataset-covid-mask"
MODELO_PATH             = "yolov8n.pt"
CONFIANCA               = 0.3
IMG_SIZE                = 320
BATCH_SIZE              = 4     # batch pequeno — melhora granularidade
TAMANHO_CHUNK           = 8     # chunk pequeno — melhora balanceamento dinâmico
SAIDA_CSV               = "resultados_paralelo.csv"
SAIDA_TEMPOS            = "tempos_paralelos.csv"
SAIDA_TEMPO_SERIAL      = "tempo_serial.txt"
CONFIGURACOES_PROCESSOS = [2, 4, 6, 8]
# ──────────────────────────────────────────────────────────────────────────────

_modelo = None
_n_workers_global = 1   # usado para calcular threads por processo


def inicializar_processo(n_workers: int):
    """
    Executado UMA VEZ ao iniciar cada processo filho.
    Calcula quantas threads cada processo deve usar com base no
    número total de workers, distribuindo os núcleos físicos
    disponíveis de forma proporcional — evita tanto contenção
    (threads demais) quanto subutilização (threads de menos).
    """
    global _modelo

    n_cores = os.cpu_count() or 12
    threads_por_processo = max(1, n_cores // n_workers)

    os.environ["OMP_NUM_THREADS"] = str(threads_por_processo)
    os.environ["MKL_NUM_THREADS"] = str(threads_por_processo)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_por_processo)

    import torch
    torch.set_num_threads(threads_por_processo)

    from ultralytics import YOLO
    _modelo = YOLO(MODELO_PATH)


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


def processar_chunk(caminhos: list[str]) -> list[dict]:
    """
    Processa um chunk pequeno (8 imagens) usando o modelo já carregado
    pelo initializer. Divide internamente em sub-batches de BATCH_SIZE.
    """
    global _modelo
    resultados = []

    sub_batches = [caminhos[i:i + BATCH_SIZE] for i in range(0, len(caminhos), BATCH_SIZE)]

    for sub_batch in sub_batches:
        try:
            r_yolo = _modelo(
                sub_batch, conf=CONFIANCA, classes=[0],
                verbose=False, imgsz=IMG_SIZE, stream=True,
            )
            for r, caminho in zip(r_yolo, sub_batch):
                contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
                for box in r.boxes:
                    coords = list(map(int, box.xyxy[0].tolist()))
                    contagem[classificar_deteccao(coords)] += 1
                contagem["total"]  = sum(contagem.values())
                contagem["imagem"] = str(Path(caminho).relative_to(DATASET_DIR))
                resultados.append(contagem)
        except Exception:
            for caminho in sub_batch:
                resultados.append({
                    "imagem": str(Path(caminho).relative_to(DATASET_DIR)),
                    "with_mask": 0, "without_mask": 0,
                    "mask_weared_incorrect": 0, "total": 0,
                })

    return resultados


def dividir_chunks(lista: list, tamanho: int) -> list[list]:
    return [lista[i:i + tamanho] for i in range(0, len(lista), tamanho)]


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
    campos = ["workers", "threads_por_worker", "imagens_por_worker", "tempo_s", "speedup", "eficiencia_pct"]
    with open(SAIDA_TEMPOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)
    print(f"[CSV] Tempos salvos em: {SAIDA_TEMPOS}")


def main():
    imagens = coletar_imagens(DATASET_DIR)
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em: {DATASET_DIR}")
        sys.exit(1)

    t_serial = ler_tempo_serial()
    if t_serial:
        print(f"[REF]  Tempo serial: {t_serial:.2f}s ({t_serial/60:.1f} min)")
    else:
        print("[AVISO] tempo_serial.txt não encontrado.\n")

    n_cores = os.cpu_count() or 12
    print(f"[INFO] CPU com {n_cores} threads lógicas disponíveis")

    chunks = dividir_chunks(imagens, TAMANHO_CHUNK)
    print(f"[PARALELO] {len(imagens)} imagens | {len(chunks)} chunks de {TAMANHO_CHUNK} | testando: {CONFIGURACOES_PROCESSOS} workers\n")

    registros_tempo    = []
    ultimos_resultados = []

    for n_proc in CONFIGURACOES_PROCESSOS:
        n_proc_real   = min(n_proc, len(imagens))
        imgs_por_proc = len(imagens) // n_proc_real
        threads_por_processo = max(1, n_cores // n_proc_real)

        print(f"─── {n_proc_real} workers x {threads_por_processo} threads (~{imgs_por_proc} imgs cada) ───")
        inicio = time.perf_counter()

        with mp.Pool(
            processes=n_proc_real,
            initializer=inicializar_processo,
            initargs=(n_proc_real,)
        ) as pool:
            lotes_resultado = list(
                pool.imap_unordered(processar_chunk, chunks, chunksize=1)
            )

        tempo = time.perf_counter() - inicio

        resultados         = [item for sub in lotes_resultado for item in sub]
        ultimos_resultados = resultados

        speedup    = round(t_serial / tempo, 4) if t_serial else 0.0
        eficiencia = round(speedup / n_proc_real * 100, 2) if t_serial else 0.0

        print(f"    Tempo      : {tempo:.2f}s ({tempo/60:.1f} min)")
        if t_serial:
            print(f"    Speedup    : {speedup:.2f}x")
            print(f"    Eficiência : {eficiencia:.1f}%")
        print()

        registros_tempo.append({
            "workers":            n_proc_real,
            "threads_por_worker": threads_por_processo,
            "imagens_por_worker": imgs_por_proc,
            "tempo_s":            round(tempo, 4),
            "speedup":            speedup,
            "eficiencia_pct":     eficiencia,
        })

    salvar_csv_resultados(ultimos_resultados)
    salvar_csv_tempos(registros_tempo)

    print("\n" + "=" * 65)
    print(f"{'Workers':>8} {'Threads':>9} {'Tempo (s)':>12} {'Speedup':>10} {'Eficiência':>12}")
    print("-" * 65)
    if t_serial:
        print(f"{'1 (ser)':>8} {n_cores:>9} {t_serial:>12.2f} {'1.00x':>10} {'100.0%':>12}")
    for r in registros_tempo:
        print(f"{r['workers']:>8} {r['threads_por_worker']:>9} {r['tempo_s']:>12.2f} "
              f"{r['speedup']:>9.2f}x {r['eficiencia_pct']:>11.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
