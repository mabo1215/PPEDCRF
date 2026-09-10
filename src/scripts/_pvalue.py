r"""One place where a $p$-value is turned into printable LaTeX.

Five table generators used to carry five private formatters, which is how one
generated table came to print $0.798$ beside $1.00$ and $0.20$ while another
printed three decimals for the same magnitude. Every generator now imports
from here, so the display rule is changed in one file or not at all.

The reporting rule, ``fmt_p``:

    p >= 0.01       two decimals                      0.18
    0.001 <= p      three significant figures in      $6.12{\times}10^{-3}$
       < 0.01       scientific notation
    p < 0.001       the bound, not a rounded zero     $<$0.001

The bound is the honest form below a thousandth: a query-cluster bootstrap and
a Wilcoxon signed-rank over a few hundred queries do not resolve a p-value
further, and ``0.000`` claims a precision the test does not have.

``fmt_p_exponent`` is a second, deliberately different form, kept for the two
direction tables only. Those tables reach $10^{-68}$, the manuscript's prose
quotes their exponents verbatim (``$p=1\times10^{-15}$``), and the supplement's
hand-written tables print the same style, so collapsing them to ``$<$0.001``
would replace one text-versus-table contradiction with a worse one. Anything
that is not one of those two tables should use ``fmt_p``.
"""
from __future__ import annotations


def fmt_p(p: float) -> str:
    """The reporting rule above, as a LaTeX cell."""
    p = float(p)
    if p != p:
        raise ValueError("p-value is NaN")
    if p < 0:
        raise ValueError("p-value is negative: %r" % p)
    if p >= 0.01:
        return "%.2f" % p
    if p < 0.001:
        return "$<$0.001"
    # [0.001, 0.01): the exponent is always -3, so three significant figures
    # is two decimals on the mantissa.
    mantissa = round(p * 1000.0, 2)
    if mantissa >= 10.0:
        # 0.00999... rounds up out of this band; print it in the band above
        # rather than as $10.00{\times}10^{-3}$.
        return "%.2f" % p
    return r"$%.2f{\times}10^{-3}$" % mantissa


def fmt_p_exponent(p: float) -> str:
    """One significant figure in scientific notation below $0.01$.

    The direction tables' form. See the module docstring for why it exists.
    """
    if p >= 0.01:
        return "%.2f" % p
    exp = 0
    while p < 1 and p != 0:
        p *= 10
        exp += 1
    return "$%.0f{\\times}10^{-%d}$" % (p, exp)
