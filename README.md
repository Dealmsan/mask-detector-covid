# 😷 Projeto Programação CD — Detecção de Máscara COVID-19

## 📋 Sobre o Dataset

Este projeto utiliza o **Face Mask Detection Dataset**, um conjunto de dados para classificação do uso correto de máscara facial, composto por imagens de pessoas em ambientes públicos capturadas durante o período da pandemia de COVID-19.

O dataset contém imagens anotadas no formato **YOLO**, com três categorias de uso de máscara, permitindo identificar não apenas a ausência da máscara, mas também o uso incorreto (abaixo do nariz, no queixo etc.).

## 📊 Informações do Dataset

| Informação | Detalhe |
|---|---|
| 🗂️ Subconjuntos | Full_Dataset, Training, Validation |
| 😷 Classes | 3 (mask, no_mask, mask_not_in_position) |
| 📐 Formato das anotações | YOLO (arquivos `.txt` com bounding boxes normalizados) |
| 📁 Formato das imagens | JPEG / PNG |
| 🔗 Fonte | tamanho total de 3.51GB/26.2k de arquivos.[Covid Face-Mask Monitoring Dataset](https://www.kaggle.com/datasets/jishan900/covid-facemask-monitoring-dataset) |

### Classes detectadas

| ID | Classe | Descrição |
|----|--------|-----------|
| 0 | `mask` | Máscara usada corretamente (cobrindo nariz e boca) |
| 1 | `no_mask` | Pessoa sem máscara |
| 2 | `mask_not_in_position` | Máscara usada de forma incorreta |

## 📁 Arquivos do Projeto

| Arquivo | Descrição |
|---|---|
| `treinar.py` | Treina o modelo YOLOv8 com o dataset (executar uma vez) |
| `serial.py` | Processa todas as imagens sequencialmente, uma por uma |
| `paralelo.py` | Processa as imagens em paralelo usando 2, 4, 8 e 12 processos |
| `DATASET-COVID-MASK/data.yaml` | Configuração do dataset para o YOLO |

## 🔍 Como os Scripts Funcionam

### Captura das Imagens

Ambos os scripts percorrem automaticamente todas as subpastas do dataset usando `os.walk()`, coletando o caminho de cada imagem `.jpg` ou `.png` encontrada:

```python
def coletar_imagens(base_dir):
    imagens = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imagens.append(os.path.join(root, f))
    return sorted(imagens)
```

Isso significa que, independentemente de como as subpastas estão organizadas (Training, Validation, Full_Dataset), todas as imagens são encontradas automaticamente.

---

### `treinar.py` — Treinamento do Modelo

Usa o modelo base `yolov8n.pt` (YOLOv8 nano) pré-treinado no dataset COCO e realiza **transfer learning** com o dataset de máscaras. O treinamento ajusta os pesos da rede para reconhecer as 3 classes do projeto.

Parâmetros principais:
- **epochs**: número de ciclos completos de treinamento (padrão: 30)
- **batch**: imagens processadas por iteração (padrão: 8; reduza para 4 em máquinas com pouca RAM)
- **patience**: early stopping — para automaticamente se o modelo não melhorar por N épocas seguidas
- **device**: `"cpu"` por padrão; troque por `0` se tiver GPU NVIDIA

O modelo treinado é salvo em `runs/detect/mask_detector/weights/best.pt`.

---

### `serial.py` — Processamento Sequencial

Processa as imagens uma por uma, em sequência. Para cada imagem:

1. Carrega o arquivo com `cv2.imread()` via inferência YOLO
2. Roda o modelo YOLOv8 treinado com `modelo(caminho, conf=0.4)`
3. Lê os bounding boxes retornados e identifica a classe de cada detecção
4. Contabiliza quantas detecções de cada classe existem na imagem
5. Armazena o resultado em memória

Ao final, salva o tempo total em `tempo_serial.txt` e todos os resultados em `resultados_serial.csv`.

---

### `paralelo.py` — Processamento Paralelo

Usa a mesma lógica de detecção do `serial.py`, mas distribui as imagens entre múltiplos processos usando `multiprocessing.Pool`. O script testa automaticamente 4 configurações:

| Processos | Imagens por processo (aprox.) |
|-----------|-------------------------------|
| 2 | metade do total |
| 4 | um quarto do total |
| 8 | um oitavo do total |
| 12 | ~1/12 do total |

Cada processo trabalha de forma **completamente independente** — carrega sua própria cópia do modelo YOLO em memória e processa seu lote sem se comunicar com os outros durante a execução. No final, os resultados são reunidos pelo processo principal.

Os tempos de cada configuração, junto com o speedup calculado, são salvos em `tempos_paralelos.csv`.

> ⚠️ **Windows:** O bloco `if __name__ == "__main__":` é obrigatório pois o Windows usa o método **spawn** para criar processos, diferente do Linux que usa **fork**. Sem esse bloco, cada processo filho tentaria executar o script inteiro novamente, causando um loop infinito.

## ⚙️ Como Executar

### 1. Instalar dependências

```bash
pip install ultralytics opencv-python pandas numpy matplotlib
```

### 2. Configurar o caminho do dataset

Abra `serial.py` e `paralelo.py` e confirme a variável no topo do arquivo:

```python
DATASET_DIR = "DATASET-COVID-MASK"   # pasta raiz do dataset baixado
```

### 3. Treinar o modelo

```bash
python treinar.py
```

> ⏱ O treinamento leva entre 10 e 30 minutos dependendo do hardware e do tamanho do dataset.

### 4. Executar

```bash
python serial.py
python paralelo.py
```

## 📤 Arquivos Gerados

| Arquivo | Conteúdo |
|---|---|
| `tempo_serial.txt` | Tempo de execução do processamento serial (em segundos) |
| `resultados_serial.csv` | Contagem de detecções por imagem (serial) |
| `resultados_paralelo.csv` | Contagem de detecções por imagem (paralelo, última config.) |
| `tempos_paralelos.csv` | Tempo, speedup e eficiência para cada configuração de processos |

### Exemplo de `resultados_serial.csv`

```
imagem,mask,no_mask,mask_not_in_position,total
img_001.jpg,3,1,0,4
img_002.jpg,0,2,1,3
img_003.jpg,5,0,0,5
```

### Exemplo de `tempos_paralelos.csv`

```
processos,imagens_por_processo,tempo_s,speedup,eficiencia_pct
2,500,38.21,2.31,1.16
4,250,21.05,4.19,1.05
8,125,13.87,6.36,0.79
12,83,11.42,7.72,0.64
```

## 📐 Métricas de Desempenho

| Métrica | Fórmula | Significado |
|---|---|---|
| Speedup | T_serial / T_paralelo | Quantas vezes ficou mais rápido |
| Eficiência | Speedup / N_processos | Aproveitamento de cada núcleo (ideal = 1.0) |
| Throughput | imagens / segundo | Capacidade de processamento |

## ⬇️ Como baixar o dataset

As imagens não estão incluídas neste repositório devido ao tamanho dos arquivos.

Faça o download pelo link abaixo e extraia mantendo a estrutura de pastas original:

🔗 **[Link do dataset — Hugging Face / Kaggle / Roboflow]**

A estrutura esperada após a extração:

```
DATASET-COVID-MASK/
├── Full_Dataset/
│   ├── images/
│   └── labels/
├── Training/
│   ├── images/
│   └── labels/
├── Validation/
│   ├── images/
│   └── labels/
├── classes.txt
└── data.yaml
```
**## Gráficos**

Tempo x Eficiência
<img width="904" height="240" alt="image" src="https://github.com/user-attachments/assets/41aedac3-4477-4cea-9c8e-5a29bd0188c3" />

Tempo

<img width="466" height="231" alt="image" src="https://github.com/user-attachments/assets/d190553a-f9ad-4948-bf4e-68d1bbfa3d25" />

Speed up

<img width="453" height="204" alt="image" src="https://github.com/user-attachments/assets/daef4848-6839-4510-99ae-d21f0dddaefe" />

