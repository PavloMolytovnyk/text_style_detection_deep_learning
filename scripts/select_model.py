"""Select an existing neural checkpoint for the next service startup."""

import argparse
import json

from textstyle.predict import Predictor
from textstyle.settings import DEFAULT_ARTIFACTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=["cnn", "bilstm", "transformer"])
    args = parser.parse_args()
    predictor = Predictor(DEFAULT_ARTIFACTS / args.architecture / "model")
    (DEFAULT_ARTIFACTS / "active_model.json").write_text(
        json.dumps(
            {
                "model_dir": f"{args.architecture}/model",
                "reason": "Explicit local selection",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected {predictor.metadata['architecture']}. Restart the API/UI to apply.")


if __name__ == "__main__":
    main()
