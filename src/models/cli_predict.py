# \!/usr/bin/env python
"""
Command line interface for making heart disease predictions.
"""
import argparse
import json
import logging
import sys

from src.models.predict_model import HeartDiseasePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make heart disease predictions from patient data."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSON file with patient data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file for predictions (defaults to stdout)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--interpretation",
        "-int",
        action="store_true",
        help="Include interpretation in results",
    )
    parser.add_argument(
        "--ensemble-only",
        "-e",
        action="store_true",
        help="Return only ensemble predictions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main():
    """Run the prediction CLI."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Load patient data
    try:
        with open(args.input, "r") as f:
            logger.info(f"Loading patient data from {args.input}")
            patient_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        sys.exit(1)

    # Initialize predictor
    predictor = HeartDiseasePredictor(model_dir=args.model_dir)

    # Make predictions
    results = predictor.predict(
        patient_data,
        return_probabilities=True,
        return_interpretation=args.interpretation,
    )

    # Filter results if ensemble-only flag is set
    if args.ensemble_only and "ensemble_predictions" in results:
        filtered_results = {
            "predictions": results["ensemble_predictions"].tolist(),
            "probabilities": results["ensemble_probabilities"].tolist(),
        }
        if args.interpretation and "interpretation" in results:
            filtered_results["interpretation"] = results["interpretation"]
        results = filtered_results

    # Convert numpy arrays to lists
    for key in results:
        if key.endswith("predictions") or key.endswith("probabilities"):
            results[key] = results[key].tolist()

    # Output results
    results_json = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            logger.info(f"Writing predictions to {args.output}")
            f.write(results_json)
    else:
        print(results_json)


if __name__ == "__main__":
    main()
