"""
This is a CLI EDC plot example: python example.py
See the CLI help for more, as defined below.
"""

# Standard imports:
import argparse
from pathlib import Path
import json

# External imports:
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

# Local imports:
from edc import EdcErrorType
from edc import EdcSample
from edc import EdcSamplePair
from edc import EdcOutput
from edc import compute_edc
from edc import compute_edc_pauc
from edc import compute_edc_area_under_theoretical_best

comparison_type_to_error_type = {
    "mated": EdcErrorType.FNMR,
    "nonmated": EdcErrorType.FMR,
}


def main(args: argparse.Namespace):
    # Load the input data:
    with open(args.data, "r", encoding="utf-8") as file:
        data = json.load(file)

    comparison_type = data["similarity_scores"]["type"].lower().replace("-", "")
    assert comparison_type in comparison_type_to_error_type, "The comparisons must either be mated or non-mated."

    similarity_scores = data["similarity_scores"]["scores_by_pair_ids"]
    quality_scores_per_algorithm = data["quality_scores_by_sample_id_per_algorithm"]

    # Compute EDC curves:
    edc_outputs = {}
    for quality_assessment_algorithm, quality_scores in tqdm(quality_scores_per_algorithm.items(), desc="EDC curves"):
        # Apply the example min-max normalization if set (deactivated by default):
        if args.min_max_normalize > 0:
            quality_scores = _min_max_normalize(quality_scores, args.min_max_normalize)
        # Prepare sample and sample pair structures for compute_edc:
        samples = {
            sample_id: EdcSample(quality_score=quality_score) for sample_id, quality_score in quality_scores.items()
        }
        sample_pairs = []
        for pair_id, similarity_score in similarity_scores.items():
            sample_id1, sample_id2 = pair_id.split("-")
            sample_pairs.append(
                EdcSamplePair(
                    samples=(
                        samples[sample_id1],
                        samples[sample_id2],
                    ),
                    similarity_score=similarity_score,
                )
            )

        # Run compute_edc:
        error_type = comparison_type_to_error_type[comparison_type]
        edc_output = compute_edc(
            error_type=error_type,
            sample_pairs=sample_pairs,
            starting_error=args.starting_error,
        )
        edc_outputs[quality_assessment_algorithm] = edc_output

    # Print true starting error vs. target starting error:
    print(f"- Target starting error (--starting-error CLI parameter): {args.starting_error}")
    some_edc_output = next(iter(edc_outputs.values()))
    true_starting_error = some_edc_output["error_fractions"][0]  # [0] for the 0% discard fraction.
    print(f"- True starting error at 0% discard fraction: {true_starting_error}")

    # The true starting error is equivalent for all QA algorithms, since it is at the 0% discard fraction:
    for edc_output in edc_outputs.values():
        assert true_starting_error == edc_output["error_fractions"][0]  # [0] for the 0% discard fraction.

    # Compute pAUC values for the EDC curves:
    pauc_values = {}
    for quality_assessment_algorithm, edc_output in edc_outputs.items():
        pauc_value = compute_edc_pauc(edc_output, args.pauc_discard_limit)
        pauc_value -= compute_edc_area_under_theoretical_best(edc_output, args.pauc_discard_limit)
        pauc_values[quality_assessment_algorithm] = pauc_value

    # Create the EDC plot:
    _create_edc_plot(
        error_type=error_type,
        edc_outputs=edc_outputs,
        starting_error=true_starting_error,
        pauc_values=pauc_values,
        pauc_discard_limit=args.pauc_discard_limit,
        shade_pauc=args.shade_pauc,
    )
    plt.show()


def _min_max_normalize(quality_scores: dict, bin_count: int) -> dict:
    value_min = min(quality_scores.values())
    value_max = max(quality_scores.values())
    value_range = value_max - value_min
    return {
        sample_id: round(bin_count * ((quality_score - value_min) / value_range))
        for sample_id, quality_score in quality_scores.items()
    }


def _create_edc_plot(
    error_type: EdcErrorType,
    edc_outputs: dict,
    starting_error: float,
    pauc_values: dict,
    pauc_discard_limit: float,
    shade_pauc: bool = True,
):
    """Create a matplotlib plot and plot the EDC curves, including the pAUC for the best curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the constant starting error as a horizontal line:
    ax.axhline(y=starting_error, xmin=0, xmax=1, color="gray", linestyle="--", alpha=0.7)

    # Plot the 'theoretical best' line:
    ax.plot([0, starting_error], [starting_error, 0], color="gray", linestyle="--", alpha=0.7)

    # Plot the shaded pAUC for the best curve:
    if shade_pauc:
        best_edc_output = _get_best_edc_output(edc_outputs, pauc_values)
        _plot_shaded_pauc(
            ax=ax,
            edc_output=best_edc_output,
            pauc_discard_limit=pauc_discard_limit,
            starting_error=starting_error,
        )

    # Plot EDC curves, with labels showing the algorithm names, pAUC values, and relative rankings:
    relative_rankings = _compute_relative_rankings(pauc_values)
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, (quality_assessment_algorithm, edc_output) in enumerate(reversed(edc_outputs.items())):
        discard_fractions = edc_output["discard_fractions"]
        error_fractions = edc_output["error_fractions"]
        label = (
            f"{quality_assessment_algorithm}"
            f" | pAUC: {pauc_values[quality_assessment_algorithm]:.4f}"
            f" | Ranking: {relative_rankings[quality_assessment_algorithm]:.2f}"
        )
        color = colors[i % len(colors)]
        ax.step(discard_fractions, error_fractions, where="post", label=label, color=color)

    ax.set_xlabel("Fraction of discarded comparisons")
    ax.set_ylabel(error_type.value)
    ax.legend()
    ax.set_title("EDC Curves")
    plt.tight_layout()


def _compute_relative_rankings(pauc_values: dict) -> dict:
    """Min-max normalize the pAUC values as 'relative rankings' (0 being the best algorithm, 1 being the worst)."""
    relative_rankings = {}
    pauc_value_min = min(pauc_values.values())
    pauc_value_max = max(pauc_values.values())
    pauc_value_range = pauc_value_max - pauc_value_min
    for quality_assessment_algorithm, pauc_value in pauc_values.items():
        relative_rankings[quality_assessment_algorithm] = (pauc_value - pauc_value_min) / pauc_value_range
    return relative_rankings


def _get_best_edc_output(edc_outputs: dict, pauc_values: dict) -> EdcOutput:
    """Get the best EdcOutput according to the pAUC values."""
    best_edc_output = None
    best_pauc_value = None
    for quality_assessment_algorithm, pauc_value in pauc_values.items():
        if (best_pauc_value is None) or (pauc_value < best_pauc_value):
            best_pauc_value = pauc_value
            best_edc_output = edc_outputs[quality_assessment_algorithm]
    return best_edc_output


def _plot_shaded_pauc(
    ax: plt.Axes,
    edc_output: EdcOutput,
    pauc_discard_limit: float,
    starting_error: float,
):
    """Plot the pAUC in the given matplotlib axis."""
    pauc_curve = {
        "x": edc_output["discard_fractions"],
        "y": edc_output["error_fractions"],
    }
    pauc_curve = _cut_curve(pauc_curve, x_limit=pauc_discard_limit)
    x = pauc_curve["x"]
    y = pauc_curve["y"]

    if pauc_discard_limit <= starting_error:
        curve_x_min = [0, pauc_discard_limit]
        curve_y_min = [starting_error, starting_error - pauc_discard_limit]
    else:
        curve_x_min = [0, starting_error, pauc_discard_limit]
        curve_y_min = [starting_error, 0, 0]

    # Fill between the EDC curve and the theoretical best
    x = np.concatenate([x, curve_x_min[::-1]])
    y = np.concatenate([y, curve_y_min[::-1]])
    ax.fill(x, y, color="lightgray", alpha=0.5)


def _cut_curve(curve: dict, x_limit: float):
    """Utility function to cut a curve for plotting purposes."""
    new_curve = {
        "x": [],
        "y": [],
    }
    index = -1
    while True:
        index += 1
        if index >= len(curve["x"]):
            break
        x_value = curve["x"][index]
        if x_value <= x_limit:
            new_curve["x"].append(x_value)
            new_curve["y"].append(curve["y"][index])
            if x_value == x_limit:
                break
        else:
            if len(new_curve["x"]) > 0:
                new_curve["x"].append(x_limit)
                new_curve["y"].append(new_curve["y"][-1])
            break
    return new_curve


if __name__ == "__main__":
    # Parse CLI arguments:
    parser = argparse.ArgumentParser(
        prog="EDC example",
        description="This example computes EDC curves with pAUC values, and shows the plot in the default browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=Path(__file__).parent / "example_data.json",
        help="Similarity scores (either mated or non-mated) and quality scores as a JSON file.",
    )
    parser.add_argument(
        "-se",
        "--starting-error",
        type=float,
        default=0.05,
        help="The target starting error at the 0%% discard fraction.",
    )
    parser.add_argument(
        "-pauc",
        "--pauc-discard-limit",
        type=float,
        default=0.20,
        help="The upper discard limit used to compute the pAUC value of the EDC curves.",
    )
    parser.add_argument(
        "--shade-pauc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shade the pAUC for the best curve.",
    )
    parser.add_argument(
        "-norm",
        "--min-max-normalize",
        type=int,
        default=0,
        help="If a value above 0 is given, e.g. 100,"
        " all quality scores will be normalized to the integer range [0, specified value]"
        " by using min-max normalization."
        " Note that the minimum and maximum values are derived from the same data that is then normalized,"
        " and this is only meant as an example for quality score normalization.",
    )
    args = parser.parse_args()

    main(args)
