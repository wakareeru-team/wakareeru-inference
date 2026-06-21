import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LANGUAGES = ("ja", "en", "zh")
L10N_METADATA_FILE_NAME = "l10n_metadata.json"


@dataclass(frozen=True)
class LocalizedLabel:
    label: dict[str, str]
    operators: dict[str, list[str]]


@dataclass(frozen=True)
class LocalizationIndex:
    by_label_id: dict[int, LocalizedLabel]


def load_localization_index(model_dir: str | Path) -> LocalizationIndex:
    path = Path(model_dir) / L10N_METADATA_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Localization metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Localization metadata must be a list: {path}")

    by_label_id = {}
    for row in payload:
        label_id, localized_label = parse_localized_label(row=row, path=path)
        if label_id in by_label_id:
            raise ValueError(f"Duplicate localization label id {label_id}: {path}")
        by_label_id[label_id] = localized_label
    return LocalizationIndex(by_label_id=by_label_id)


def validate_localization_index(
    localization: LocalizationIndex,
    *,
    labels: list[dict[str, Any]],
) -> None:
    expected = {int(row["label_id"]): str(row["label"]) for row in labels}
    if set(localization.by_label_id) != set(expected):
        missing = sorted(set(expected) - set(localization.by_label_id))
        unexpected = sorted(set(localization.by_label_id) - set(expected))
        raise ValueError(
            "Localization label ids do not match classifier labels: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for label_id, canonical_label in expected.items():
        localized_label = localization.by_label_id[label_id].label["ja"]
        if localized_label != canonical_label:
            raise ValueError(
                "Localization label mismatch for label id "
                f"{label_id}: classifier={canonical_label!r}, metadata={localized_label!r}"
            )


def parse_localized_label(*, row: Any, path: Path) -> tuple[int, LocalizedLabel]:
    if not isinstance(row, dict) or "id" not in row:
        raise ValueError(f"Invalid localization metadata row: {path}")
    label_id = int(row["id"])
    label = parse_language_map(row.get("label"), field="label", label_id=label_id, path=path)
    operator_map = row.get("operator")
    if not isinstance(operator_map, dict):
        raise ValueError(f"Invalid operator metadata for label id {label_id}: {path}")
    operator_values = {
        language: parse_operator_values(
            operator_map.get(language),
            language=language,
            label_id=label_id,
            path=path,
        )
        for language in LANGUAGES
    }
    lengths = {len(values) for values in operator_values.values()}
    if len(lengths) != 1:
        raise ValueError(f"Operator translations are not aligned for label id {label_id}: {path}")
    return label_id, LocalizedLabel(label=label, operators=operator_values)


def parse_language_map(
    value: Any,
    *,
    field: str,
    label_id: int,
    path: Path,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {field} metadata for label id {label_id}: {path}")
    result = {}
    for language in LANGUAGES:
        text = value.get(language)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Invalid {field}.{language} metadata for label id {label_id}: {path}"
            )
        result[language] = text.strip()
    return result


def parse_operator_values(
    value: Any,
    *,
    language: str,
    label_id: int,
    path: Path,
) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(
            f"Invalid operator.{language} for label id {label_id}; "
            f"expected a string or string list: {path}"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Invalid operator.{language} value for label id {label_id}: {path}")
    return [item.strip() for item in value]


def localize_prediction(
    prediction: dict[str, Any],
    *,
    localization: LocalizationIndex,
) -> dict[str, Any]:
    label_id = int(prediction["label_id"])
    localized = localization.by_label_id.get(label_id)
    if localized is None:
        raise ValueError(f"Localization metadata missing label id {label_id}")
    canonical_label = str(prediction["label"])
    if localized.label["ja"] != canonical_label:
        raise ValueError(
            "Localization label mismatch for label id "
            f"{label_id}: prediction={canonical_label!r}, metadata={localized.label['ja']!r}"
        )
    return {
        "label_id": label_id,
        "label": dict(localized.label),
        "operator": {
            language: list(localized.operators[language])
            for language in LANGUAGES
        },
        "probability": float(prediction["probability"]),
    }
