"""utils/formatting.render_math: \\( \\) / \\[ \\] math -> $ / $$, every other "$" literal."""

from utils.formatting import render_math


def test_inline_math_becomes_dollars():
    assert render_math(r"custa \(O(n \log n)\) no pior caso") == r"custa $O(n \log n)$ no pior caso"


def test_display_math_becomes_double_dollars():
    assert render_math(r"\[T(n) = 2T(n/2) + n\]") == "$$T(n) = 2T(n/2) + n$$"


def test_currency_is_escaped():
    assert render_math("custa R$ 10 ou US$ 2") == r"custa R\$ 10 ou US\$ 2"


def test_currency_next_to_math():
    assert render_math(r"R$ 5 por \(n\) itens") == r"R\$ 5 por $n$ itens"


def test_stray_dollar_math_is_shown_literally():
    assert render_math("$n$") == r"\$n\$"


def test_already_escaped_dollar_is_kept():
    assert render_math(r"R\$ 10") == r"R\$ 10"


def test_code_is_untouched():
    text = "use `echo $HOME`:\n\n```sh\nprice=$1\n```\n"
    assert render_math(text) == text
