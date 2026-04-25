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


def test_compose_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--compose"])
    assert args.compose is True


def test_compose_flag_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.compose is False


def test_compose_log_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args(["--compose"])
    assert args.compose_log is None


def test_compose_log_override():
    from pathlib import Path

    parser = build_parser()
    args = parser.parse_args(["--compose", "--compose-log", "/tmp/x.jsonl"])
    assert args.compose_log == Path("/tmp/x.jsonl")


def test_compose_l3_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--compose-l3"])
    assert args.compose_l3 is True


def test_compose_l3_flag_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.compose_l3 is False


def test_compose_l3_log_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args(["--compose-l3"])
    assert args.compose_l3_log is None


def test_compose_l3_log_override():
    from pathlib import Path

    parser = build_parser()
    args = parser.parse_args(
        ["--compose-l3", "--compose-l3-log", "/tmp/x.l3.jsonl"]
    )
    assert args.compose_l3_log == Path("/tmp/x.l3.jsonl")
