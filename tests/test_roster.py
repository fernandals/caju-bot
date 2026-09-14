"""
Tests for utils/helpers.py roster loading + enrollment check - this is
the entire access-control gate for the student app (login screen has no
real password, see README). Worth locking down before a real class uses
this, since a bug here means either locking out enrolled students or
letting anyone in.
"""

from pathlib import Path

from utils.helpers import is_enrolled, load_roster


def write_roster(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "roster.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_plain_ids(tmp_path):
    path = write_roster(tmp_path, "202312345\n202398765\n")
    roster = load_roster(path)

    assert roster == {"202312345", "202398765"}


def test_ignores_blank_lines_and_comments(tmp_path):
    path = write_roster(
        tmp_path,
        "# turma IMD1151 2026.2\n"
        "202312345\n"
        "\n"
        "   \n"
        "# comentario no meio\n"
        "202398765\n",
    )
    roster = load_roster(path)

    assert roster == {"202312345", "202398765"}


def test_missing_file_returns_empty_set(tmp_path):
    # No roster file present must fail closed (nobody enrolled), not crash
    # the app or silently let everyone in.
    roster = load_roster(tmp_path / "does_not_exist.txt")

    assert roster == set()


def test_is_enrolled_matches_exact_id():
    roster = {"202312345"}
    assert is_enrolled("202312345", roster) is True


def test_is_enrolled_rejects_unknown_id():
    roster = {"202312345"}
    assert is_enrolled("999999999", roster) is False


def test_is_enrolled_strips_whitespace_from_input():
    # A student pasting their ID with a trailing space/newline from the
    # login field should not be locked out.
    roster = {"202312345"}
    assert is_enrolled("202312345 \n", roster) is True


def test_is_enrolled_against_empty_roster_denies_everyone():
    assert is_enrolled("202312345", set()) is False


def test_roster_lines_are_not_trimmed_of_internal_whitespace_oddly():
    # A roster line is stripped as a whole line, but comparison is exact -
    # an ID with a typo'd internal space would not silently match.
    roster = {"2023 12345"}
    assert is_enrolled("202312345", roster) is False
