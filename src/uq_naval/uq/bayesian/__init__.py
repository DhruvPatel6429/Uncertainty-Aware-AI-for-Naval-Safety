"""Bayesian Deep Learning methods.

Implements three complementary approximations to the Bayesian posterior
over neural-network weights:

- :class:`MCDropoutBayesian`  — Gal & Ghahramani (2016) — approximate VI via
  dropout at inference time.
- :class:`LaplaceBayesian`    — post-hoc Laplace approximation around the MLE
  solution (Kristiadi et al., 2020).
- :class:`BayesByBackprop`    — Blundell et al. (2015) — variational posterior
  with reparameterisation trick.

References
----------
Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
Blundell, C. et al. (2015). Weight Uncertainty in Neural Networks.
Kendall, A. & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian DL.
Kristiadi, A. et al. (2020). Being Bayesian, Even Just a Bit.
"""
from .mc_dropout import MCDropoutBayesian
from .laplace import LaplaceBayesian
from .bayes_by_backprop import BayesByBackprop

__all__ = [
    "MCDropoutBayesian",
    "LaplaceBayesian",
    "BayesByBackprop",
]
