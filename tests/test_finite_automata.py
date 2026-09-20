import pytest

from project.finite_automata import regex_to_dfa


@pytest.mark.parametrize(
    "regex",
    ["a", "a b c", "a*", "(a|b)* c", "(a b) | (a c)", "a* a* b", "$"],
)
def test_regex_to_dfa_is_minimal_dfa(regex):
    dfa = regex_to_dfa(regex)

    assert dfa.is_deterministic()
    # повторная минимизация не уменьшает число состояний
    assert len(dfa.minimize().states) == len(dfa.states)


@pytest.mark.parametrize(
    "regex, word",
    [
        ("a", ["a"]),
        ("a b c", ["a", "b", "c"]),
        ("a*", []),
        ("a*", ["a", "a", "a"]),
        ("(a|b)* c", ["c"]),
        ("(a|b)* c", ["a", "b", "a", "c"]),
        ("(a b) | (a c)", ["a", "c"]),
        ("$", []),
    ],
)
def test_regex_to_dfa_accepts(regex, word):
    assert regex_to_dfa(regex).accepts(word)


@pytest.mark.parametrize(
    "regex, word",
    [
        ("a", []),
        ("a", ["b"]),
        ("a b c", ["a", "c"]),
        ("a*", ["b"]),
        ("(a|b)* c", ["a", "b"]),
        ("(a b) | (a c)", ["a", "b", "c"]),
        ("$", ["a"]),
    ],
)
def test_regex_to_dfa_rejects(regex, word):
    assert not regex_to_dfa(regex).accepts(word)


def test_regex_to_dfa_empty_regex_gives_empty_language():
    assert regex_to_dfa("").is_empty()


def test_regex_to_dfa_symbol_is_whole_token():
    # символы разделяются пробелами, поэтому "abc" --- один символ
    dfa = regex_to_dfa("abc | d")

    assert dfa.accepts(["abc"])
    assert not dfa.accepts(["a", "b", "c"])
