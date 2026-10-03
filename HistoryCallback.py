from transformers import TrainerCallback

class HistoryCallback(TrainerCallback):
    """
    Callback used with Hugging Face `Trainer` to collect logged metrics.

    `entries` accumulates dictionaries with a `step` key and any numeric
    metric reported by the trainer (e.g., `loss`, `eval_loss`, `learning_rate`).
    Zebrane wpisy służą do późniejszego zapisu wykresów i porównań.
    """

    def __init__(self) -> None:
        # Lista wpisów: każdy element to dict zawierający przynajmniej
        # {'step': float} plus dowolne inne numeryczne metryki.
        self.entries: list[dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        # Wywoływane przez HF Trainer, gdy logi są dostępne.
        # Filtrujemy tylko wartości liczbowe (int/float) i konwertujemy
        # je na float dla spójności JSON/plotowania.
        if logs:
            entry = {"step": float(state.global_step)}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    entry[key] = float(value)
            self.entries.append(entry)
