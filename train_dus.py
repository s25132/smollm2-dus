"""Educational depth up-scaling + continued-pretraining experiment.

The default experiment transforms HuggingFaceTB/SmolLM2-135M from 30 to
40 decoder layers, then continues causal language-model pretraining on
WikiText-2 (wikitext-2-raw-v1 ~2,55 miliona tokenów ). It records loss/perplexity at three checkpoints:

1. original 30-layer model,
2. 40-layer model immediately after depth up-scaling,
3. 40-layer model after continued pretraining.

Designed for a 4 GB NVIDIA GPU. Start with --smoke-test.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from graphics_tools import plot_training, plot_dus_architecture
from HistoryCallback import HistoryCallback




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth up-scale SmolLM2 and run continued pretraining."
    )
    # Argumenty uruchomieniowe kontrolują konfigurację eksperymentu.
    # Skup się tu na doborze modelu, rozmiarze bloków tokenów oraz
    # ustawieniach treningowych (epoki, batch, learning-rate itp.).
    # Zmiany tych wartości wpływają bezpośrednio na czas i pamięć GPU.
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Hugging Face base causal language model.",
    )
    parser.add_argument("--output-dir", default="outputs/smollm2-dus-40l")
    parser.add_argument("--target-layers", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=256) # to jest rozmiar bloków tokenów
    parser.add_argument("--train-rows", type=int, default=5_000)
    parser.add_argument("--eval-rows", type=int, default=500)
    parser.add_argument("--test-rows", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="Maximum batches used for each diagnostic loss measurement.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny 20-step run that checks the complete pipeline.",
    )
    return parser.parse_args()


def configure_smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    # Szybka konfiguracja testowa: ograniczamy rozmiary i liczbę kroków,
    # aby uruchomić cały pipeline bez długiego treningu (przydatne do CI).
    args.block_size = min(args.block_size, 128)
    args.train_rows = min(args.train_rows, 250)
    args.eval_rows = min(args.eval_rows, 80)
    args.max_steps = 20
    args.eval_steps = 10
    args.save_steps = 20
    args.logging_steps = 1
    args.eval_batches = min(args.eval_batches, 10)


# Helper functions for model analysis and depth upscaling (DUS).
def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return decoder layers for Llama-like Hugging Face models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError(
        "Unsupported architecture: expected model.model.layers, as in "
        "LlamaForCausalLM/SmolLM2."
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def depth_upscale(model: nn.Module, target_layers: int) -> dict[str, Any]:
    """
        Increase depth using the two-slice construction used by DUS.

        (PL) Zwiększa głębokość modelu stosując konstrukcję "two-slice".

        Idea (przykład 30 -> 40 warstw):
            - pierwsza kopia (first slice) używa warstw źródłowych 0..19,
            - druga kopia (second slice) używa warstw źródłowych 10..29,
            - oba wycinki są połączone sekwencyjnie, dając 40 warstw.

        Ważne cechy i założenia:
            - nadmiarowe/overlapping warstwy są kopiowane przez `copy.deepcopy`,
                więc parametry nie są współdzielone między powstałymi blokami;
            - funkcja oczekuje struktury Llama-like: `model.model.layers` istnieje;
            - obsługiwany jest tylko wzrost do maksymalnie `2 * source_depth`.

        Funkcja zwraca słownik z mapowaniem indeksów źródła do modelu docelowego
        oraz informacją o "seam" (połączeniu), przydatny do diagnostyki.
    """
    source_layers = decoder_layers(model)
    source_depth = len(source_layers)

    # Sprawdzenie: cel musi być większy niż źródło (to jest up-scaling).
    if target_layers <= source_depth:
        raise ValueError(
            f"target_layers must exceed source depth ({source_depth}), "
            f"got {target_layers}."
        )
    # Ograniczenie: ten algorytm używa tylko dwóch wycinków źródła, więc
    # nie obsłuży celów większych niż 2x głębokość źródła.
    if target_layers > 2 * source_depth:
        raise ValueError(
            f"This two-copy implementation supports at most "
            f"{2 * source_depth} target layers."
        )

    first_count = target_layers // 2
    second_count = target_layers - first_count
    # Każda z części (slice) musi mieścić się w oryginalnej liczbie warstw.
    if first_count > source_depth or second_count > source_depth:
        raise ValueError("Each DUS slice must fit inside the source model.")

    first_indices = list(range(first_count))
    second_indices = list(range(source_depth - second_count, source_depth))

    # Tworzymy głębokie kopie wybranych warstw. deepcopy gwarantuje, że
    # powstałe warstwy mają własne, niezależne tensory parametrów.
    # Kolejność w `new_layers` odpowiada kolejności warstw w modelu docelowym.
    new_layers = [copy.deepcopy(source_layers[i]) for i in first_indices]
    new_layers.extend(copy.deepcopy(source_layers[i]) for i in second_indices)

    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = target_layers

    # Some Transformers versions also keep the depth in generation-related
    # config objects. The architectural source of truth remains model.config.
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = target_layers

    return {
        "source_depth": source_depth,
        "target_depth": target_layers,
        "first_source_indices": first_indices,
        "second_source_indices": second_indices,
        "seam": {
            "left_source_layer": first_indices[-1],
            "right_source_layer": second_indices[0],
        },
    }


def prepare_dataset(
    tokenizer: Any,
    block_size: int,
    train_rows: int = 1000,
    eval_rows: int = 1000,
    test_rows: int = 1000,
) -> DatasetDict:
    # Ładujemy mały, gotowy korpus WikiText-2 (surowy tekst bez tokenów specjalnych).
    # Dla większych eksperymentów zastąp innym zbiorem lub lokalnymi danymi.
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    selected = DatasetDict(
        {
            "train": raw["train"].select(
                range(min(train_rows, len(raw["train"])))
            ),
            "validation": raw["validation"].select(
                range(min(eval_rows, len(raw["validation"])))
            ),
            "test": raw["test"].select(
                range(min(test_rows, len(raw["test"])))
            ),
        }
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        """
        Tokenizuje listę fragmentów tekstu zwróconych przez Dataset.map.

        Wejście: `batch` zawiera klucz `text` o liście stringów.
        Zwracamy słownik zgodny z interfejsem tokenizerów HF — listy
        `input_ids` i opcjonalnie `attention_mask` dla każdego przykładu.

        Ważne: ustawiamy `add_special_tokens=False` ponieważ później
        łączymy tokeny i dzielimy na bloki o stałej długości.
        """
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = selected.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing WikiText-2",
    )

    def group_tokens(batch: dict[str, list[list[int]]]) -> dict[str, Any]:
        """
        Grupuje spakowane tokeny w bloki o stałej długości `block_size`.

        Wejście: `batch` to słownik, gdzie każdy klucz ma listę list tokenów
        (np. `input_ids` jest listą list int). Funkcja:
          1. konkatenatuje wszystkie listy tokenów w jedną długą listę,
          2. obcina końcówkę tak, aby długość była wielokrotnością
             `block_size`,
          3. dzieli na kolejne bloki o rozmiarze `block_size`.

        Zwracany format jest zgodny z Dataset.map — każda wartość to lista
        bloków (lista list int) gotowych do użycia jako sekwencje treningowe.
        """
        concatenated = {key: sum(batch[key], []) for key in batch.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        return {
            key: [
                values[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

    grouped = tokenized.map(
        group_tokens,
        batched=True,
        desc=f"Grouping tokens into blocks of {block_size}",
    )

    if len(grouped["train"]) == 0 or len(grouped["validation"]) == 0:
        raise RuntimeError(
            "The selected dataset subset produced no token blocks. Increase "
            "--train-rows/--eval-rows or reduce --block-size."
        )
    return grouped


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: Dataset,
    collator: Any,
    device: torch.device,
    max_batches: int,
) -> float:
    """
    Evaluate average loss on a dataset subset.

    (PL) Oblicza średnią stratę (loss) modelu na dostarczonym `dataset`.

    Szczegóły implementacji:
      - Tworzymy `DataLoader` z `batch_size=1` ponieważ `collator` przygotowuje
        batch dla pełnego przebiegu ewaluacji (np. jeden blok tokenów na entry).
      - Przenosimy model na zadany `device` i ustawiamy w tryb ewaluacji
        (`model.eval()`), aby wyłączyć dropout i inne tryby treningowe.
      - Iterujemy po batchach aż do `max_batches` i zbieramy `output.loss`.
      - Wynik zwracany jest jako średnia arytmetyczna strat. Jeśli nie
        przetworzono żadnego batcha, zgłaszamy błąd.

    Uwaga operacyjna: `max_batches` pozwala ograniczyć koszt obliczeń
    podczas szybkich pomiarów (np. `--eval-batches`).
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )
    # Przenieś model na target device i ustaw w tryb ewaluacji.
    model.to(device)
    model.eval()
    losses: list[float] = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        # Przenieś tensory batcha na device przed wywołaniem modelu.
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))

    if not losses:
        raise RuntimeError("No validation batches were evaluated.")
    return float(np.mean(losses))



def release_from_gpu(model: nn.Module) -> None: # Przenosi model na CPU i oczyszcza pamięć GPU.
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 50 else float("inf")


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_percentage = (
        100 * trainable / total
        if total > 0
        else 0
    )

    return {
        "total": total,
        "trainable": trainable,
        "trainable_percentage": trainable_percentage,
    }


def main() -> None:
    """
        Main entry point for the DUS experiment.

        (PL) Główna procedura eksperymentu:
            1. Parsuje argumenty i ustawia ziarno RNG.
            2. Przygotowuje katalog wyjściowy i wybiera device (GPU/CPU).
            3. Wczytuje tokenizer i dataset, grupuje na bloki tokenów.
            4. Wczytuje model bazowy, mierzy loss przed DUS.
            5. Stosuje `depth_upscale`, mierzy loss po DUS.
            6. Włącza gradient checkpointing i kontynuuje pretraining
                 (Trainer) zbierając historię metryk.
            7. Zapisuje model końcowy, tokenizer oraz porównawcze metryki.

        Komentarze w kodzie opisują istotne kroki konfiguracji i treningu.
    """
    args = parse_args()
    configure_smoke_test(args)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    # Wybór device i FP16: na GPU ustawiamy mieszane precyzje.
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA unavailable; training will be slow.")

    # Wczytanie tokenizer-a. Jeśli nie ma zdefiniowanego `pad_token_id`,
    # ustawiamy `pad_token` na `eos_token` aby uniknąć błędów padowania.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(
        tokenizer=tokenizer,
        block_size=args.block_size,
        train_rows=args.train_rows,
        eval_rows=args.eval_rows,
        test_rows=args.test_rows,
    )
    print(
        f"Prepared blocks: train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}"
    )

    # Wczytujemy model w odpowiednim dtype. `low_cpu_mem_usage=True` pomaga
    # zmniejszyć użycie pamięci CPU podczas ładowania dużych modeli.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    print("\nMODEL WCZYTANY Z DYSKU")
    print("-" * 60)
    print(f"Ścieżka/nazwa modelu: {args.model_name}")
    print(f"Liczba warstw: {len(model.model.layers)}")
    print(f"Config num_hidden_layers: {model.config.num_hidden_layers}")

    base_parameters = parameter_count(model)

    print(f"Wszystkie parametry przed DUS: {base_parameters:,}")

    assert len(model.model.layers) == 30, (
        "Model wejściowy powinien mieć 30 warstw, "
        f"ale wczytano {len(model.model.layers)}. "
        "Prawdopodobnie wczytujesz model zapisany po DUS."
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    results: dict[str, Any] = {
        "configuration": vars(args),
        "device": str(device),
        "base_parameters": parameter_count(model),
        "measurements": {},
    }

    # 1) Pomiar baseline: oryginalny model przed DUS
    print("\n[1/3] Evaluating original model...")
    base_loss = evaluate_loss(
        model, dataset["test"], collator, device, args.eval_batches
    )
    results["measurements"]["base"] = {
        "loss": base_loss,
        "perplexity": safe_perplexity(base_loss),
    }

    print(f"Base loss={base_loss:.4f}, ppl={safe_perplexity(base_loss):.2f}")
    release_from_gpu(model)

    # 2) Zastosowanie depth up-scaling (DUS) i pomiar natychmiast po zmianie
    print("\n[2/3] Applying depth up-scaling...")
    dus_layout = depth_upscale(model, args.target_layers)
    results["dus_layout"] = dus_layout
    results["upscaled_parameters"] = parameter_count(model)


    plot_dus_architecture(
        layout=dus_layout,
        base_parameters=results["base_parameters"],
        upscaled_parameters=results["upscaled_parameters"],
        destination=output_dir / "dus_architecture.png",
    )

    print(
        f"Diagram zapisano w: "
        f"{output_dir / 'dus_architecture.png'}"
    )

    upscaled_parameter_info = count_parameters(model)

    print("\nMODEL PO DUS")
    print("-" * 60)
    print(f"Liczba warstw: {len(model.model.layers)}")
    print(f"Wszystkie parametry po DUS: {upscaled_parameter_info['total']:,}")

    added_parameters = upscaled_parameter_info['total'] - base_parameters
    increase_percentage = 100 * added_parameters / base_parameters

    print(f"Dodane parametry: {added_parameters:,}")
    print(f"Wzrost parametrów: {increase_percentage:.2f}%")

    assert len(model.model.layers) == 40
    assert upscaled_parameter_info['total'] > base_parameters


    dus_loss = evaluate_loss(
        model, dataset["test"], collator, device, args.eval_batches
    )

    results["measurements"]["after_dus"] = {
        "loss": dus_loss,
        "perplexity": safe_perplexity(dus_loss),
    }
    print(f"After-DUS loss={dus_loss:.4f}, ppl={safe_perplexity(dus_loss):.2f}")
    release_from_gpu(model)

    # 3) Kontynuacja pretrainingu po rozszerzeniu głębokości.
    # Włączamy `gradient_checkpointing` aby zmniejszyć zużycie pamięci GPU
    # kosztem większego czasu obliczeń.
    print("\n[3/3] Starting continued pretraining...")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    history_callback = HistoryCallback()
    optimizer_name = "adamw_bnb_8bit" if use_fp16 else "adamw_torch"
    # Konfiguracja treningu (kluczowe opcje):
    # - `output_dir`: gdzie zapisywane będą checkpointy i model końcowy.
    # - `num_train_epochs` / `max_steps`: kontrolują liczbę iteracji.
    # - `per_device_train_batch_size` i `gradient_accumulation_steps`: używane
    #   do skalowania efektywnego batcha przy ograniczonej pamięci GPU.
    # - `fp16`: włącza mieszane precyzje gdy jest GPU z obsługą FP16.
    # - `optim`: wybór optymalizatora (tu zmieniamy dla mniejszych pamięci).
    # - `logging_strategy`/`eval_strategy`/`save_strategy`: decydują o tym,
    #   kiedy logować, ewaluować i zapisywać checkpointy.
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        fp16=use_fp16,
        gradient_checkpointing=True,
        optim=optimizer_name,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        # Wybór najlepszego modelu
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[
            history_callback,
            EarlyStoppingCallback(
                early_stopping_patience=5
            ),
        ],
    )
    # Uruchamiamy proces trenowania. `train_result` zawiera metryki i
    # informacje o trwaniu/ostatnim kroku. `history_callback.entries`
    # zbiera logi krokowe używane następnie do wykresów.
    train_result = trainer.train()

    model.config.use_cache = True
    model.gradient_checkpointing_disable() # Wyłączamy gradient checkpointing po treningu  ponieważ nie jest już potrzebny i może spowalniać generowanie.
    final_loss = evaluate_loss(
        model, dataset["test"], collator, device, args.eval_batches
    )
 
    results["measurements"]["after_continued_pretraining"] = {
        "loss": final_loss,
        "perplexity": safe_perplexity(final_loss),
    }

    results["train_metrics"] = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in train_result.metrics.items()
    }
    results["history"] = history_callback.entries

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    plot_training(history_callback.entries, output_dir / "training_curve.png")

    print("\nExperiment complete")
    print("-" * 72)
    for name, measurement in results["measurements"].items():
        print(
            f"{name:32s} loss={measurement['loss']:.4f} " # loss to jest miarą błędu modelu mniejsza wartość oznacza lepsze dopasowanie
            f"ppl={measurement['perplexity']:.2f}" # perplexity to miara trudności przewidywania modelu niższa wartość oznacza lepsze dopasowanie
        )
    print(f"\nArtifacts: {output_dir}")


if __name__ == "__main__":
    main()

