import json

import pytest

from wakareeru_inference.localization import (
    load_localization_index,
    validate_localization_index,
)


def test_load_localization_lists_operators_by_identity(tmp_path) -> None:
    payload = [
        {
            "id": 0,
            "label": {"ja": "101系", "en": "101 series", "zh": "101系"},
            "operator": {
                "ja": ["JR東日本", "国鉄"],
                "en": ["JR East", "Japanese National Railways"],
                "zh": ["JR东日本", "日本国有铁道"],
            },
            "wiki_title_ja": "国鉄101系電車",
        }
    ]
    (tmp_path / "l10n_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    localization = load_localization_index(tmp_path)

    assert localization.by_label_id[0].operators == {
        "ja": ["JR東日本", "国鉄"],
        "en": ["JR East", "Japanese National Railways"],
        "zh": ["JR东日本", "日本国有铁道"],
    }
    assert localization.by_label_id[0].wiki_title_ja == "国鉄101系電車"


def test_load_localization_normalizes_single_operator_to_list(tmp_path) -> None:
    payload = [
        {
            "id": 0,
            "label": {"ja": "101系", "en": "101 series", "zh": "101系"},
            "operator": {
                "ja": "国鉄",
                "en": "Japanese National Railways",
                "zh": "日本国有铁道",
            },
            "wiki_title_ja": "国鉄101系電車",
        }
    ]
    (tmp_path / "l10n_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    localization = load_localization_index(tmp_path)

    assert localization.by_label_id[0].operators == {
        "ja": ["国鉄"],
        "en": ["Japanese National Railways"],
        "zh": ["日本国有铁道"],
    }


def test_validate_localization_rejects_classifier_label_mismatch(tmp_path) -> None:
    payload = [
        {
            "id": 0,
            "label": {"ja": "101系", "en": "101 series", "zh": "101系"},
            "operator": {"ja": [], "en": [], "zh": []},
            "wiki_title_ja": "国鉄101系電車",
        }
    ]
    (tmp_path / "l10n_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    localization = load_localization_index(tmp_path)

    with pytest.raises(ValueError, match="Localization label mismatch"):
        validate_localization_index(
            localization,
            labels=[{"label_id": 0, "label": "103系"}],
        )


def test_load_localization_rejects_missing_wiki_title_ja(tmp_path) -> None:
    payload = [
        {
            "id": 0,
            "label": {"ja": "101系", "en": "101 series", "zh": "101系"},
            "operator": {"ja": ["国鉄"], "en": ["JNR"], "zh": ["日本国有铁道"]},
        }
    ]
    (tmp_path / "l10n_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid wiki_title_ja metadata"):
        load_localization_index(tmp_path)
