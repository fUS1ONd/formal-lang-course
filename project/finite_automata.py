from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Строит минимальный ДКА по регулярному выражению в синтаксисе pyformlang."""
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()
