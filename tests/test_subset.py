"""Tests for class remapping. Run with: python -m pytest tests"""
import pytest

from assistive_vision.data.subset import build_id_map, remap_label_lines

SOURCE = ["bench", "pole", "warning_column", "spherical_roadblock", "waste_container"]


def test_id_map_follows_new_class_order():
    classes = {"bollard": ["warning_column"], "bin": ["waste_container"], "pole": ["pole"]}
    assert build_id_map(SOURCE, classes) == {2: 0, 4: 1, 1: 2}


def test_merging_two_sources_into_one_class():
    classes = {"bollard": ["warning_column", "spherical_roadblock"]}
    assert build_id_map(SOURCE, classes) == {2: 0, 3: 0}


def test_unknown_source_class_raises():
    with pytest.raises(ValueError):
        build_id_map(SOURCE, {"bollard": ["bolard_typo"]})


def test_remap_drops_unkept_and_keeps_coordinates():
    lines = ["0 0.5 0.5 0.1 0.1", "2 0.1 0.2 0.3 0.4", "", "4 0.9 0.9 0.05 0.05"]
    id_map = {2: 0, 4: 1}
    assert remap_label_lines(lines, id_map) == ["0 0.1 0.2 0.3 0.4", "1 0.9 0.9 0.05 0.05"]
