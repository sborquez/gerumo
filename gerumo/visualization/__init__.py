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
    show_regression_identity, show_residual_error, show_residual_error_distribution,
    show_absolute_error_angular, show_angular_resolution,
    plot_angular_resolution_comparison, plot_error_and_angular_resolution,
    show_absolute_error_angular, show_energy_resolution,
    plot_energy_resolution_comparison, plot_error_and_energy_resolution
)

from .dataset import show_input_sample, show_target_sample
