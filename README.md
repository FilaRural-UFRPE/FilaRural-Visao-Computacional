# FilaRural — Visão Computacional

Sistema de detecção de fila no RU usando **YOLOv8** (ONNX Runtime) a partir da câmera Intelbras.

## O que faz

Captura frames da câmera via RTSP e envia para a API de análise, que conta pessoas na fila e publica o status (vazia/pequena/média/grande) para o app dos estudantes.

## Como rodar no notebook (24/7)

### 1. Pré-requisitos

- Python **3.8+**
- Notebook na **mesma rede local** da câmera
- IP e senha da câmera Intelbras

### 2. Instalar

```bash
git clone <url-do-repositorio>
cd FilaRural-Visao-Computacional
pip install -r requirements.txt
```

### 3. Configurar (credenciais da câmera — não estão no GitHub)

Crie um arquivo `.env` na pasta do projeto:

```
RTSP_PASSWORD="senha_da_camera"
RTSP_IP="192.168.x.x"
```

### 4. Rodar

```bash
python capture_and_analyze.py
```

O script captura um frame **a cada 5 minutos** e envia para a API. Deixe o notebook ligado.

## Análise de uma imagem individual

```bash
python yolo_executable.py caminho/da/imagem.jpg
```

## Configurações já ativadas no código

| Configuração | Valor | Efeito |
|--------------|-------|--------|
| Rotação da câmera | `180°` | Corrige câmera instalada de cabeça para baixo |
| Threshold de confiança | `0.04` | Contagem calibrada (~37 pessoas em fila cheia) |
| Recorte de ROI | 64%–82% | Foca na faixa da calçada onde a fila se forma |

> Ajustes avançados disponíveis via variáveis de ambiente (`YOLO_CONF_THRESHOLD`, `CAMERA_ROTATE_DEGREES`, `ROI_*_PCT`, etc.). Veja os comentários em `yolo.py`.