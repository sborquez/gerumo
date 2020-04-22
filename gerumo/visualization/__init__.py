"""
Visualization
=============

Generate plots and show information.

`plot` funtions generate a figure. These can be complex figures or independent
figure. `show` functions generate individual and simples plots in an axis. This
type of function can be used by `plot` to generate more elaborated figures.

"""

from .metrics import plot_model_training_history
from .data import show_input_sample, show_target_sample
