# SmolLM2 Depth Up-Scaling

Kompletny eksperyment edukacyjny pokazujący:

1. pomiar bazowego `SmolLM2-135M` (30 warstw),
2. depth up-scaling do 40 warstw,
3. pomiar modelu bezpośrednio po przebudowie,
4. continued pretraining na WikiText-2,
5. ponowny pomiar loss i perplexity.

Skrypt zapisuje również przykładową generację tekstu na każdym etapie,
wykres treningu i końcowy model.

## Środowisko

Projekt jest ustawiony pod Windows 11, Python 3.12 i kartę NVIDIA z 4 GB VRAM.
Zalecane jest osobne środowisko wirtualne.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sprawdzenie CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Najpierw: smoke test

Smoke test przechodzi przez cały pipeline, ale wykonuje tylko 20 kroków:

```powershell
python train_dus.py --smoke-test
```

Wyniki znajdą się w:

```text
outputs/smollm2-dus-40l/
├── comparison.json
├── training_curve.png
├── checkpoints/
└── final_model/
```

## Pełny mały eksperyment

```powershell
python train_dus.py
```

Domyślne ustawienia:

- model: `HuggingFaceTB/SmolLM2-135M`;
- dane: `Salesforce/wikitext`, `wikitext-2-raw-v1`;
- głębokość: 30 → 40 warstw;
- długość sekwencji: 256 tokenów;
- batch: 1;
- gradient accumulation: 8;
- optymalizator: 8-bit AdamW;
- gradient checkpointing: włączony;
- 10 epok na ograniczonym podzbiorze danych.

wikitext-2-raw-v1	
Train:36718 Valid:3760  Test:4358

## Jak wykonany jest DUS

Model źródłowy ma warstwy `0..29`. Model docelowy jest składany z:

```text
pierwszy fragment:  0..19
drugi fragment:    10..29
wynik:             40 niezależnych warstw
```

Warstwy z obszaru `10..19` występują w nowym modelu dwukrotnie, ale są
głębokimi kopiami: podczas treningu ich wagi mogą zmieniać się niezależnie.

## Jak interpretować wyniki

Otwórz `comparison.json` i porównaj:

- `measurements.base` – model oryginalny;
- `measurements.after_dus` – model po zmianie architektury;
- `measurements.after_continued_pretraining` – model po treningu.

Najważniejsze wartości to `loss` i `perplexity`. Perplexity jest liczona jako:

```text
perplexity = exp(loss)
```

Mniejsza wartość jest lepsza. Typowym oczekiwaniem jest pogorszenie wyniku
bezpośrednio po DUS, a następnie jego poprawa podczas continued pretraining.
Krótki eksperyment nie gwarantuje jednak, że model 40-warstwowy pokona dobrze
wytrenowany model bazowy.

## Najważniejsze argumenty

```text
--target-layers 40
--block-size 256
--train-rows 5000
--eval-rows 500
--epochs 1
--max-steps -1
--learning-rate 5e-5
--gradient-accumulation 8
--eval-batches 50
```

Przykład treningu przez dokładnie 500 kroków:

```powershell
python train_dus.py --max-steps 500 --eval-steps 50 --save-steps 100
```

## Model bazowy SmolLM2 
Tokeny
  ↓
Embedding
  ↓
30 × LlamaDecoderLayer
  ↓
Końcowy RMSNorm
  ↓
LM Head
  ↓
Prawdopodobieństwa następnego tokenu


## Model bazowy SmolLM2 po Depth Up-Scaling
Tokeny
  ↓
Embedding
  ↓
40 × LlamaDecoderLayer
  ↓
Końcowy RMSNorm
  ↓
LM Head
  ↓
Prawdopodobieństwa następnego tokenu

Wszystkie parametry po DUS: 169,915,968
Dodane parametry: 35,400,960

## Wyniki po 1 epoce
```powershell
python train_dus.py --epochs 1 --train-rows 36718 --eval-rows 3760 --test-rows 4358
```
------------------------------------------------------------------------
base                             loss=3.2635 ppl=26.14
after_dus                        loss=3.5069 ppl=33.34
after_continued_pretraining      loss=2.7931 ppl=16.33

## Wyniki treningu
```powershell
python train_dus.py --epochs 5 --train-rows 36718 --eval-rows 3760 --test-rows 4358
```
Experiment complete
------------------------------------------------------------------------
base                             loss=3.2635 ppl=26.14
after_dus                        loss=3.5069 ppl=33.34
after_continued_pretraining      loss=2.7956 ppl=16.37

EarlyStopping na epoce 1.37.


## Benchmarki lambada_openai, hellaswag i piqa
Instalujemy lm_eval
```powershell
pip install "lm_eval[hf]"
```

Pełne benchmarki na rozszerzony model SmolLM2 40 warstw
```powershell
python -m lm_eval `
  --model hf `
  --model_args "pretrained=outputs/smollm2-dus-40l/final_model,dtype=float16" `
  --tasks lambada_openai,hellaswag,piqa `
  --device cuda:0 `
  --batch_size 1 `
  --output_path outputs/smollm2-dus-40l/benchmarks_full
```

Model bazowy SmolLM2
```powershell
python -m lm_eval `
  --model hf `
  --model_args "pretrained=HuggingFaceTB/SmolLM2-135M,dtype=float16" `
  --tasks lambada_openai,hellaswag,piqa `
  --device cuda:0 `
  --batch_size 1 `
  --output_path outputs/smollm2-dus-40l/benchmarks_base
```

Benchmark	              Model bazowy	  DUS + trening	        Zmiana
HellaSwag acc_norm	    43,12%	          41,25%	            −1,87 p.p.
LAMBADA acc	            42,83%	          37,75%	            −5,08 p.p.
LAMBADA perplexity	    19,26	            27,44	              +8,18 — gorzej
PIQA acc_norm	          68,44%	          65,18%	            −3,26 p.p.

## Dalsze prace
WikiText-103 – większy zbiór.
raw = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
)