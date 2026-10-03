from pathlib import Path
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt

def plot_training(history: list[dict[str, float]], destination: Path) -> None:
    train_points = [item for item in history if "loss" in item]
    eval_points = [item for item in history if "eval_loss" in item]

    plt.figure(figsize=(9, 5))
    if train_points:
        plt.plot(
            [item["step"] for item in train_points],
            [item["loss"] for item in train_points],
            label="train loss",
            alpha=0.85,
        )
    if eval_points:
        plt.plot(
            [item["step"] for item in eval_points],
            [item["eval_loss"] for item in eval_points],
            marker="o",
            label="validation loss",
        )
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Continued pretraining after depth up-scaling")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()




def plot_dus_architecture(
    layout: dict,
    base_parameters: int,
    upscaled_parameters: int,
    destination: Path,
) -> None:
    """
    Rysuje architekturę modelu przed i po Depth Up-Scaling.

    Kolory:
    - niebieski: warstwy z pierwszej kopii modelu,
    - pomarańczowy: warstwy z drugiej kopii,
    - czerwona linia: nowy szew pomiędzy kopiami.
    """

    source_depth = layout["source_depth"]
    first_indices = layout["first_source_indices"]
    second_indices = layout["second_source_indices"]

    target_indices = first_indices + second_indices
    seam_position = len(first_indices)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 6.5),
        gridspec_kw={"height_ratios": [1, 1.35]},
    )

    fig.suptitle(
        "Depth Up-Scaling: architektura przed i po zwiększeniu głębokości",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    def configure_axis(axis, number_of_layers, title):
        axis.set_xlim(-0.8, number_of_layers + 0.8)
        axis.set_ylim(-0.35, 1.35)
        axis.set_title(
            title,
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        axis.axis("off")

    # ==========================================================
    # Model przed Depth Up-Scaling
    # ==========================================================

    before_axis = axes[0]

    configure_axis(
        before_axis,
        source_depth,
        (
            f"Przed DUS — {source_depth} warstw "
            f"({base_parameters / 1_000_000:.1f} mln parametrów)"
        ),
    )

    for layer_index in range(source_depth):
        rectangle = Rectangle(
            (layer_index, 0.15),
            width=0.82,
            height=0.72,
            facecolor="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
        )

        before_axis.add_patch(rectangle)

        before_axis.text(
            layer_index + 0.41,
            0.51,
            str(layer_index),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )

    before_axis.text(
        -0.65,
        0.51,
        "Wejście",
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=10,
    )

    before_axis.text(
        source_depth + 0.05,
        0.51,
        "Wyjście",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=10,
    )

    # ==========================================================
    # Model po Depth Up-Scaling
    # ==========================================================

    after_axis = axes[1]

    configure_axis(
        after_axis,
        len(target_indices),
        (
            f"Po DUS — {len(target_indices)} warstw "
            f"({upscaled_parameters / 1_000_000:.1f} mln parametrów)"
        ),
    )

    for new_position, source_layer_index in enumerate(target_indices):

        comes_from_first_copy = new_position < seam_position

        if comes_from_first_copy:
            color = "#4C78A8"
        else:
            color = "#F28E2B"

        rectangle = Rectangle(
            (new_position, 0.15),
            width=0.82,
            height=0.72,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
        )

        after_axis.add_patch(rectangle)

        # Pokazujemy indeks warstwy w modelu źródłowym
        after_axis.text(
            new_position + 0.41,
            0.51,
            str(source_layer_index),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=6.5,
            fontweight="bold",
        )

    # ==========================================================
    # Zaznaczenie nowego szwu
    # ==========================================================

    seam_x = seam_position - 0.09

    after_axis.axvline(
        seam_x,
        ymin=0.22,
        ymax=0.76,
        color="#C62828",
        linewidth=3,
        linestyle="--",
    )

    left_source_layer = first_indices[-1]
    right_source_layer = second_indices[0]

    after_axis.text(
        seam_x,
        1.08,
        (
            f"Nowy szew: warstwa źródłowa "
            f"{left_source_layer} → {right_source_layer}"
        ),
        horizontalalignment="center",
        verticalalignment="center",
        color="#A31515",
        fontsize=10,
        fontweight="bold",
    )

    arrow = FancyArrowPatch(
        (seam_x, 0.98),
        (seam_x, 0.82),
        arrowstyle="-|>",
        mutation_scale=13,
        color="#C62828",
        linewidth=1.5,
    )

    after_axis.add_patch(arrow)

    # ==========================================================
    # Podpisy
    # ==========================================================

    after_axis.text(
        -0.65,
        0.51,
        "Wejście",
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=10,
    )

    after_axis.text(
        len(target_indices) + 0.05,
        0.51,
        "Wyjście",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=10,
    )

    after_axis.text(
        len(first_indices) / 2,
        -0.12,
        (
            f"Pierwsza kopia: warstwy źródłowe "
            f"{first_indices[0]}–{first_indices[-1]}"
        ),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
        color="#2F5F8F",
    )

    after_axis.text(
        seam_position + len(second_indices) / 2,
        -0.12,
        (
            f"Druga kopia: warstwy źródłowe "
            f"{second_indices[0]}–{second_indices[-1]}"
        ),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
        color="#B85D00",
    )

    fig.text(
        0.5,
        0.015,
        (
            "Liczby w blokach oznaczają indeksy warstw modelu źródłowego. "
            "Skopiowane warstwy mają niezależne parametry podczas treningu."
        ),
        horizontalalignment="center",
        fontsize=10,
        color="#444444",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    plt.savefig(
        destination,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)