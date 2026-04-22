"""Tests for cli.py. Parser structure only — never triggers model loading."""

from __future__ import annotations

from infinity_forge.cli import build_parser


def test_multi_model_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--multi-model"])
    assert args.multi_model is True


def test_multi_model_flag_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.multi_model is False


def test_multi_model_flag_coexists_with_other_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--multi-model",
        "--iterations", "7",
        "--sanity-check",
    ])
    assert args.multi_model is True
    assert args.iterations == 7
    assert args.sanity_check is True
