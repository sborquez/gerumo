"""
Visualization
=============

Generate plots and show information.

`plot` funtions generate a figure. These can be complex figures or independent
figure. `show` functions generate individual and simples plots in an axis. This
type of function can be used by `plot` to generate more elaborated figures.

"""

from .metrics import (\
    plot_model_training_history,
    plot_assembler_prediction,
    show_prediction_1d, show_prediction_2d, show_prediction_3d,
    plot_regression_evaluation,
    show_regression_identity, show_residual_error, show_residual_error_distribution
)

from .data import show_input_sample, show_target_sample
