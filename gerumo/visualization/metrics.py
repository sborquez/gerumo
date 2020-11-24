"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparison.
"""

from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import norm, multivariate_normal, rv_continuous, gaussian_kde
from scipy.stats._multivariate import multivariate_normal_frozen
import numpy as np
import pandas as pd
import ctaplot
from sklearn.metrics import r2_score


__all__ = [
    'plot_model_training_history',
    'plot_prediction', 
        'show_points_1d','show_points_2d',
        'show_pdf_2d',
        'show_pmf_1d', 'show_pmf_2d', 'show_pmf_3d', 
    'plot_model_validation_regressions',
    'plot_regression_evaluation', 
        'show_regression_identity', 
        'show_residual_error',
        'show_residual_error_distribution',
    'plot_energy_resolution_comparison', 
    'plot_error_and_energy_resolution', 
        'show_energy_resolution', 
        'show_absolute_error_energy', 
    'plot_angular_resolution_comparison', 
    'plot_error_and_angular_resolution',  
        'show_angular_resolution', 
        'show_absolute_error_angular'
]

"""
Training Metrics
================
"""

def plot_model_training_history(history, training_time, model_name, epochs, save_to=None):
    """
    Display training loss and validation loss vs epochs.
    """
    fig = plt.figure(figsize=(12,6))
    epochs = len(history.history['loss']) #fix: early stop 
    epochs = [i for i in range(1, epochs+1)]
    plt.plot(epochs, history.history['loss'], "*--", label="Train")
    plt.plot(epochs, history.history['val_loss'], "*--", label="Validation")

    # Style
    plt.title(f'Model {model_name} Training Loss\n Training time {training_time} [min]')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(epochs, rotation=-90)
    plt.grid()
    
    # Show or Save
    if save_to is not None:
        fig.savefig(join(save_to, f'{model_name} - Training Loss.png'))
        plt.close(fig)
    else:
        plt.show()


def plot_model_validation_regressions(evaluation_results, targets, save_to=None):
    """
    Display regression metrics for a model's predictions.
    
    Example:
    ========
    ```
    results = {
      "pred_alt": [1, 2],
      "true_alt": [1, 2],
      "pred_az":  [1, 2],
      "true_az":  [1, 2],
    }
    targets = ["alt", "az"]
    evaluation_results = pd.DataFrame(results)
    plot_model_validation_regressions(evaluation_results, targets)
    ```
    """
    n_targets = len(targets)
    # Create Figure and axis
    fig, axs = plt.subplots(n_targets, 3, figsize=(19, n_targets*6))
    
    # Style
    fig.suptitle("Targets Regression")

    # For each target, generate two plots
    for i, target in enumerate(targets):
        # Target and prediction values
        prediction_points = evaluation_results[f"pred_{target}"]
        targets_points = evaluation_results[f"true_{target}"]
        score = r2_score(prediction_points, targets_points)
        # Show regression
        ax_r = axs[i][0] if n_targets > 1 else axs[0]
        show_regression_identity(prediction_points, targets_points, score, target, flip=True, axis=ax_r)
        # Show error
        ax_e = axs[i][1] if n_targets > 1 else axs[1]
        show_residual_error(prediction_points, targets_points, score, target,  axis=ax_e)
        # Show error distribution
        ax_d = axs[i][2] if n_targets > 1 else axs[2]
        show_residual_error_distribution(prediction_points, targets_points, score, target, vertical=True, axis=ax_d)
    # Save or Show
    if save_to is not None:
        fig.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()

 
"""
Predictions
----------
"""
def _label_formater(target, use_degrees=False, only_units=False):
    units = {
        "az" : "[deg]" if use_degrees else "[rad]",
        "alt": "[deg]" if use_degrees else "[rad]",
        "mc_energy": "[TeV]",
        "log10_mc_energy":  "$log_{10}$ [TeV]"
    }
    if only_units:
        return units[target]
    else:
        return f"{target} {units[target]}"


def plot_prediction(prediction, prediction_point, targets, target_domains,
                              target_resolutions=None,  title=None, targets_values=None,
                              save_to=None):
    """
    Display the assembled prediction of a event, the probability and the predicted point.
    If targets_values is not None, the target point is included in the figure.
    """

    # Create new Figure
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    # Style
    if isinstance(title, str):
        title = f"Prediction for event {title}"
    elif isinstance(title, tuple):
        title = f"Prediction for event {title[0]}\ntelescope id {title[1]}"
    else:
        title = f"Prediction for a event"
    plt.title(title)

    # point estimator
    if np.array_equal(prediction, prediction_point):
        if len(targets) == 1:
            ax = show_points_1d(prediction, prediction_point, targets, target_domains, targets_values, ax)
        elif len(targets) == 2:
            ax = show_points_2d(prediction, prediction_point, targets, target_domains, targets_values, ax)
        elif len(targets) == 3:
            raise NotImplementedError
    # probability density function estimator
    elif not isinstance(prediction, np.ndarray) and np.any(np.array(target_resolutions) == np.inf):
        # Show prediction according to targets dim
        if len(targets) == 1:
            raise NotImplementedError
        elif len(targets) == 2:
            ax = show_pdf_2d(prediction, prediction_point, targets, target_domains, targets_values, ax)
        elif len(targets) == 3:
            raise NotImplementedError
    # probability mass function estimator
    else:
        # Show prediction according to targets dim
        if len(targets) == 1:
            ax = show_pmf_1d(prediction, prediction_point, targets, target_domains, 
                        target_resolutions, targets_values, ax)
        elif len(targets) == 2:
            ax = show_pmf_2d(prediction, prediction_point, targets, target_domains, 
                        target_resolutions, targets_values, ax)
        elif len(targets) == 3:
            raise NotImplementedError

    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

"""
POINT predictions
----------------
"""
def show_points_1d(prediction, prediction_point, targets, target_domains, targets_values=None, axis=None):
    """
    Show predicted point in a 1d target domain.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    # Draw probability map
    x1=np.linspace(target_domains[0][0], prediction_point[0], 75)
    y1=np.zeros_like(x1)
    y1[-1] = 1
    x2=np.linspace(prediction_point[0], target_domains[0][1], 75)[1:]
    y2=np.zeros_like(x2) 
    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    axis.plot(x, y, "-",color="blue", alpha=0.9)
    def rainbow_fill(X,Y, axis, cmap=plt.get_cmap("jet")):
        axis.plot(X,Y, lw=0)  # Plot so the axes scale correctly
        dx = (X[1]-X[0])
        N  = float(X.size)
        y_max = Y.max()
        for x,y in zip(X,Y):
            polygon = plt.Rectangle((x,0),dx,y,color=cmap(y/y_max), alpha=0.9)
            axis.add_patch(polygon)
        rainbow_fill(x, y, axis)

    # Add predicted point
    axis.axvline(x=prediction_point[0], c="white", linestyle="--", linewidth=3,
                 label=f"prediction=({prediction_point[0]:.4f})", alpha=0.9)
    # Add target point
    if targets_values is not None:
      axis.axvline(x=targets_values[0], linestyle="--", c="black", linewidth=3,
                   label=f"target=({targets_values[0]:.4f})", alpha=0.9)

    # Style
    axis.set_facecolor('lightgrey')
    axis.set_xlim(target_domains[0])
    axis.set_xlabel(_label_formater(targets[0]))
    axis.legend()
    return axis

def show_points_2d(prediction, prediction_point, targets, target_domains, targets_values=None, axis=None):
    """
    Show predicted point in a 2d target domain.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    ## Probability map 
    xx, yy = np.mgrid[target_domains[0][0]:target_domains[0][1]:.005,  target_domains[1][0]:target_domains[1][1]:.005]
    im = axis.contourf(yy.T, xx.T, np.zeros_like(xx.T), cmap='jet') #, norm=LogNorm(vmin=pdf.min(), vmax=pdf.max()))
    axis.set_xlim(( target_domains[1][0], target_domains[1][1]))
    axis.set_ylim((target_domains[0][0], target_domains[0][1]))

    ## Add color bar
    plt.colorbar(im, ax=axis, extend='max')
    # Add prediction points
    if len(prediction.shape) == 1:
        prediction = np.array([prediction])

    axis.scatter(x=prediction[:, 1], y=prediction[:, 0], c="red", alpha=0.4)

    # Add predicted point
    axis.scatter(x=[prediction_point[1]], y=[prediction_point[0]], c="w", marker="*", 
                 label=f"prediction=({prediction_point[0]:.4f}, {prediction_point[1]:.4f})", alpha=1)
    # Add target point
    if targets_values is not None:
      axis.scatter(x=[targets_values[1]], y=[targets_values[0]], c="black",marker="o", 
                   label=f"target=({targets_values[0]:.4f}, {targets_values[1]:.4f})", alpha=0.9)
    # Style
    axis.set_ylabel(_label_formater(targets[0]))
    axis.set_xlabel(_label_formater(targets[1]))
    axis.legend()
    return axis


"""
PDF predictions
================
"""

def show_pdf_2d(prediction, prediction_point, targets, target_domains, targets_values=None, axis=None):
    """
    Show predicted pdf in a 2d target domain.
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    # Draw probability
    if isinstance(prediction, rv_continuous):
        xx, yy = np.mgrid[target_domains[0][0]:target_domains[0][1]:.005,  target_domains[1][0]:target_domains[1][1]:.005]
        pos = np.dstack((xx, yy))
        pdf = prediction.prob(pos)
        if not isinstance(pdf, np.ndarray):
            pdf = pdf.numpy()
    elif isinstance(prediction, gaussian_kde):
        xx, yy = np.mgrid[target_domains[0][0]:target_domains[0][1]:.005,  target_domains[1][0]:target_domains[1][1]:.005]
        pos = np.dstack((xx, yy))
        plot_shape = pos.shape
        pos = pos.reshape(-1, plot_shape[-1])
        pdf = prediction.pdf(pos.T).reshape(plot_shape[:-1])
    elif isinstance(prediction, multivariate_normal_frozen):
        #xx, yy = np.mgrid[target_domains[0][0]:target_domains[0][1]:.005,  target_domains[1][0]:target_domains[1][1]:.005]
        yy, xx = np.meshgrid(
            np.linspace(target_domains[1][0], target_domains[1][1],200),
            np.linspace(target_domains[0][0], target_domains[0][1], 200)
        )
        pos = np.dstack((xx, yy))
        plot_shape = pos.shape
        pos = pos.reshape(-1, plot_shape[-1])
        pdf = prediction.pdf(pos).reshape(plot_shape[:-1])

    ## Probability map 
    im = axis.contourf(yy.T, xx.T, pdf.T, cmap='jet') #, norm=LogNorm(vmin=pdf.min(), vmax=pdf.max()))
    axis.set_xlim(( target_domains[1][0], target_domains[1][1]))
    axis.set_ylim((target_domains[0][0], target_domains[0][1]))

    ## Add color bar
    plt.colorbar(im, ax=axis, extend='max')

    # Add predicted point
    axis.scatter(x=[prediction_point[1]], y=[prediction_point[0]], c="w", marker="*", 
                 label=f"prediction=({prediction_point[0]:.4f}, {prediction_point[1]:.4f})", alpha=0.9)
    # Add target point
    if targets_values is not None:
      axis.scatter(x=[targets_values[1]], y=[targets_values[0]], c="black",marker="o", 
                   label=f"target=({targets_values[0]:.4f}, {targets_values[1]:.4f})", alpha=0.9)
    # Style
    axis.set_ylabel(_label_formater(targets[0]))
    axis.set_xlabel(_label_formater(targets[1]))
    axis.legend()
    return axis

"""
PMF predictions
--------------
"""

def show_pmf_1d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions=None, targets_values=None, axis=None):
    """
    Show predicted pmf in a 1d target domain.
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    # Draw probability map
    x=np.linspace(target_domains[0][0], target_domains[0][1], len(prediction))
    axis.plot(x, prediction, "-",color="blue", alpha=0.9)
    def rainbow_fill(X,Y, axis, cmap=plt.get_cmap("jet")):
        axis.plot(X,Y, lw=0)  # Plot so the axes scale correctly
        dx = (X[1]-X[0])
        N  = float(X.size)
        y_max = Y.max()
        for x,y in zip(X,Y):
            polygon = plt.Rectangle((x,0),dx,y,color=cmap(y/y_max), alpha=0.9)
            axis.add_patch(polygon)
    rainbow_fill(x, prediction, axis)

    # Add predicted point
    axis.axvline(x=prediction_point[0], c="white", linestyle="--", linewidth=3,
                 label=f"prediction=({prediction_point[0]:.4f})", alpha=0.9)
    # Add target point
    if targets_values is not None:
      axis.axvline(x=targets_values[0], linestyle="--", c="black", linewidth=3,
                   label=f"target=({targets_values[0]:.4f})", alpha=0.9)

    # Style
    axis.set_facecolor('lightgrey')
    axis.set_xlim(target_domains[0])
    axis.set_xlabel(_label_formater(targets[0]))
    axis.legend()
    return axis

def show_pmf_2d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions=None, targets_values=None, axis=None):
    """
    Show predicted pmf in a 2d target domain.
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()

    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    # Draw probability map
    ## Probability map in Log scale
    epsilon = 2e-10
    extend = target_domains[1][0], target_domains[1][1], target_domains[0][0], target_domains[0][1]
    vmin = prediction.min()
    if vmin <= 0:
      vmin = epsilon
      prediction += epsilon
    vmax = prediction.max()  
    im = axis.imshow(prediction, origin="lower", cmap="jet", extent=extend, 
                     aspect=3,  norm=LogNorm(vmin=vmin, vmax=vmax))
    ## Add color bar
    plt.colorbar(im, ax=axis, extend='max')

    # Add predicted point
    axis.scatter(x=[prediction_point[1]], y=[prediction_point[0]], c="w", marker="*", 
                 label=f"prediction=({prediction_point[0]:.4f}, {prediction_point[1]:.4f})", alpha=0.9)
    # Add target point
    if targets_values is not None:
      axis.scatter(x=[targets_values[1]], y=[targets_values[0]], c="black",marker="o", 
                   label=f"target=({targets_values[0]:.4f}, {targets_values[1]:.4f})", alpha=0.9)
    # Style
    axis.set_ylabel(_label_formater(targets[0]))
    axis.set_xlabel(_label_formater(targets[1]))
    axis.legend()
    return axis

def show_pmf_3d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions=None, targets_values=None, axis=None):
    """
    Show predicted pmf in a 3d target domain.
    """
    if isinstance(target_domains, dict):
        target_domains = [[target_domains[t][0],target_domains[t][1]] for t in targets]

    raise NotImplementedError

"""
Regression Metrics
==================
"""

def show_regression_identity(prediction_points, targets_points, score, target, flip=False, axis=None):
    """
    Show a comparation between true values and predicted values, it uses a scatter plot
    for a small set or a hexbin plot for a set bigger than 500 samples.

    A nice fit means that points are distributed close to the identity diagonal
    """
    if flip:
        ylabel = "Predicted Values"
        y = prediction_points.values
        xlabel = "True Values"
        x = targets_points.values
    else:
        ylabel = "True Values"
        y = targets_points.values
        xlabel = "Predicted Values"
        x = prediction_points.values

    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()

    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max()) 
    if len(targets_points) < 500:
        axis.scatter(x=x, y=y, alpha=0.6)
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], "r--", label="identity", linewidth=3)
    else:
        x = np.append(x, vmin)
        x = np.append(x, vmax)
        y = np.append(y, vmin)
        y = np.append(y, vmax)
        axis.hexbin(x, y, gridsize=(41,41), cmap="jet")
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], "w--", label="identity", linewidth=3)

    # Style
    title = _label_formater(target)
    axis.set_title(f"Regression on {title}")
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.grid(True)
    axis.set_aspect("equal")

    # Empty plot for add legend score
    axis.plot([], [], ' ', label=f"$R^2$ score = {score:.4f}")
    axis.legend()

    return axis

def show_residual_error(prediction_points, targets_points, score, target, axis=None):
    """
    Show the distribution of the residual error along the predicted points.

    The residual error is calculated as the diference of targets_points and 
    prediction_points:

    residual_error = targets_points - prediction_points
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()
    
    # Residual Error
    residual_error = targets_points - prediction_points
    x_vmin = prediction_points.min()
    x_vmax = prediction_points.max()
    y_vmin = residual_error.min()
    y_vmax = residual_error.max()
    y_lim = 1.05*max(abs(y_vmin), abs(y_vmax))

    # Plot
    if len(residual_error) < 500:
        axis.scatter(x=prediction_points, y=residual_error, alpha=0.6)
    else:
        x = prediction_points.values
        y = residual_error.values
        x = np.append(x, x_vmin)
        x = np.append(x, x_vmax)
        y = np.append(y, -1*y_lim)
        y = np.append(y, y_lim)
        axis.hexbin(x, y, gridsize=(41,41), cmap="jet", zorder=0)
    axis.plot([x_vmin, x_vmax], [0, 0], "r--")

    # Style
    title = _label_formater(target)
    axis.set_title(f"Residual Error on {title}")
    axis.set_xlabel("Predicted Values")
    axis.set_ylabel("Residual Error")
    axis.set_ylim([-1*y_lim, y_lim])
    axis.grid(True)
    axis.set_aspect("auto")

    return axis

def show_residual_error_distribution(prediction_points, targets_points, score, target, vertical=False, axis=None):
    """
    Show the distribution of the residual error, caculate its mean and std.

    residual_error = targets_points - prediction_points
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()
    
    # Residual Error
    residual_error = targets_points - prediction_points
    vmin = residual_error.min()
    vmax = residual_error.max()
    lim = 1.05 * max(abs(vmin), abs(vmax))

    # Normalized
    weights = np.ones_like(residual_error)/len(residual_error)

    # Plot
    unit = _label_formater(target, only_units=True)
    legend = f"mean: {residual_error.mean():.4f} {unit}\nstd: {residual_error.std():.4f} {unit}"
    if vertical:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1*lim, lim),
                  orientation="horizontal", label=legend)
    else:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1*lim, lim),
                  label=legend)

    # Style
    title = _label_formater(target)
    axis.set_title(f"Residual Error Distribution on {title}")
    if vertical:
        axis.set_ylabel("Residual Error")
        axis.set_ylim([-1*lim, lim])
    else:
        axis.set_xlabel("Residual Error")
        axis.set_xlim([-1*lim, lim])
    axis.set_aspect("auto")
    axis.legend()    
    axis.set_aspect("auto")
    axis.legend()

    return axis


def plot_regression_evaluation(evaluation_results, targets, scores, save_to=None):
    """
    Display regression metrics for a model's predictions.
    """

    n_targets = len(targets)
    # Create Figure and axis
    fig, axs = plt.subplots(n_targets, 3, figsize=(19, n_targets*6))
    
    # Style
    plt.suptitle("Targets Regression")

    # For each target, generate two plots
    for i, (target, score) in enumerate(zip(targets, scores)):
        # Target and prediction values
        prediction_points = evaluation_results[f"pred_{target}"]
        targets_points = evaluation_results[f"true_{target}"]
        # Show regression
        ax_r = axs[i][0] if n_targets > 1 else axs[0]
        show_regression_identity(prediction_points, targets_points, score, target, axis=ax_r)
        # Show error
        ax_e = axs[i][1] if n_targets > 1 else axs[1]
        show_residual_error(prediction_points, targets_points, score, target, axis=ax_e)
        # Show error distribution
        ax_d = axs[i][2] if n_targets > 1 else axs[2]
        show_residual_error_distribution(prediction_points, targets_points, score, target, vertical=True, axis=ax_d)

    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


"""
CTA Metrics
==================
"""

def show_energy_resolution(predicted_mc_energy, true_mc_energy, 
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           label="this method", include_requirement=[], xlim=None, ylim=None, ax=None):
    """
    Show the energy resolution for a model's predictions.
    """
    
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()

    ax = ctaplot.plot_energy_resolution(true_mc_energy, predicted_mc_energy,
                                percentile, confidence_level, bias_correction, ax,
                                marker='o', label=label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()

    try:
        for include in include_requirement:
            ax = ctaplot.plot_energy_resolution_cta_requirement(include, ax)
    except:
        print("Unable to display cta requirements.")
    return ax

def show_absolute_error_energy(predicted_mc_energy, true_mc_energy, bins=60, percentile_plot_range=80,ax=None):
    """
    Show absolute energy error for a model's predictions.
    """
    
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()

    absolute_error = np.abs(predicted_mc_energy - true_mc_energy)
    
    x_max = np.percentile(absolute_error, 100)
    hist = ax.hist(absolute_error, bins=bins, range=(0, x_max))
    y_max = hist[0].max()

    sig_68 = np.percentile(absolute_error, 68.27)
    sig_95 = np.percentile(absolute_error, 95.45)
    sig_99 = np.percentile(absolute_error, 99.73)

    if percentile_plot_range >= 68:
        ax.vlines(sig_68, 0, y_max, label=f'68%: {sig_68:.4f} [TeV]', linestyle="--", color='red')
    if percentile_plot_range >= 95:
        ax.vlines(sig_95, 0, y_max, label=f'95%: {sig_95:.4f} [TeV]', linestyle="--", color='green')
    if percentile_plot_range >= 99:
        ax.vlines(sig_99, 0, y_max, label=f'99%: {sig_99:.4f} [TeV]', linestyle="--", color='yellow')

    # Style
    ax.set_title(f"Absolute Error Distribution")
    
    ax.set_aspect("auto")

    ax.set_ylabel("Count")
    ax.set_xlabel("$\Delta$ E [TeV]")
    ax.legend()
    return ax

def plot_error_and_energy_resolution(evaluation_results, bins=80, include_requirement=[], 
                                     percentile=68.27, confidence_level=0.95, bias_correction=False,
                                     percentile_plot_range=80, label="this method", xlim=None, ylim=None,
                                     save_to=None):
    """
    Display absolute energy error and energy resolution for a model's predictions.
    """
    predicted_mc_energy = evaluation_results["pred_log10_mc_energy"].apply(lambda log_e: np.power(log_e, 10))
    true_mc_energy = evaluation_results["true_mc_energy"]
    # Create Figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(14, 6))
    # Style
    plt.suptitle("Energy Reconstruction")
    # Generate two plots
    show_absolute_error_energy(predicted_mc_energy, true_mc_energy, bins=bins, 
                               percentile_plot_range=percentile_plot_range, ax=axis[0])
    show_energy_resolution(predicted_mc_energy, true_mc_energy, percentile, confidence_level,
                           bias_correction, label, include_requirement, xlim, ylim,
                           ax=axis[1])
    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

def plot_energy_resolution_comparison(evaluation_results_dict, include_requirement=[], 
                                     percentile=68.27, confidence_level=0.95, bias_correction=False,
                                     percentile_plot_range=80, xlim=None, ylim=None, save_to=None):
    """
    Display comparison of the energy resolution for different models.
    """
    # Create Figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.title("Energy Resolution Comparison")
    for label, results in evaluation_results_dict.items():
        # Prediction values
        predicted_log10_mc_energy = results["pred_log10_mc_energy"]
        predicted_mc_energy = np.power(10, predicted_log10_mc_energy)
        true_mc_energy = results["true_mc_energy"]
        show_energy_resolution(predicted_mc_energy, true_mc_energy, 
                           percentile, confidence_level, bias_correction,
                           label, [], xlim, ylim, ax)
    try:
        for include in include_requirement:
            ax = ctaplot.plot_energy_resolution_cta_requirement(include, ax)
    except:
        print("Unable to display cta requirements.")
    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()
    
def show_angular_resolution(predicted_alt, predicted_az, true_alt, true_az, true_mc_energy,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           label="this method", include_requirement=[], xlim=None, ylim=None, ax=None):
    """
    Show absolute angular error for a model's predictions.
    """
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()
    ax = ctaplot.plot_angular_resolution_per_energy(predicted_alt, predicted_az, true_alt, true_az, true_mc_energy,
                                percentile, confidence_level, bias_correction, ax,
                                marker='o', label=label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    try:
        for include in include_requirement:
            ax = ctaplot.plot_energy_resolution_cta_requirement(include, ax)
    except:
        print("Unable to display cta requirements.")
    return ax

def show_absolute_error_angular(predicted_alt, predicted_az, true_alt, true_az, bias_correction=False, 
                                ax=None, bins=40, percentile_plot_range=80):
    """
    Show the absolute error distribution of a method.
    """
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()
    
    bins = np.linspace(0.01,10,50)
    ax = ctaplot.plot_theta2(predicted_alt, predicted_az, true_alt, true_az, bias_correction, ax, bins=bins)
    return ax

def plot_error_and_angular_resolution(evaluation_results, bins=80, include_requirement=[], 
                                     percentile=68.27, confidence_level=0.95, bias_correction=False,
                                     percentile_plot_range=80, label="this method", xlim=None, ylim=None,
                                     save_to=None):
    """
    Display absolute angular error and angular resolution for a model's predictions.
    """
    # Predictted values
    predicted_alt = evaluation_results["pred_alt"]
    predicted_az = evaluation_results["pred_az"]
    # True values
    true_alt = evaluation_results["true_alt"]
    true_az = evaluation_results["true_az"]
    true_mc_energy = evaluation_results["true_mc_energy"]
    # Create Figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(14, 6))
    # Style
    plt.suptitle("Angular Reconstruction")
    # Generate two plots
    show_absolute_error_angular(predicted_alt, predicted_az, true_alt, true_az, bias_correction, axis[0], bins, 
                               percentile_plot_range)
    show_angular_resolution(predicted_alt, predicted_az, true_alt, true_az, true_mc_energy,
                            percentile, confidence_level, bias_correction, label, include_requirement, xlim, ylim,
                           ax=axis[1])
    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_angular_resolution_comparison(evaluation_results_dict, include_requirement=[], 
                                     percentile=68.27, confidence_level=0.95, bias_correction=False,
                                     percentile_plot_range=80, xlim=None, ylim=None, save_to=None):
    """
    Display comparison of the angular resolution for different models.
    """
    # Create Figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.title("Angular Resolution Comparison")
    for label, results in evaluation_results_dict.items():
        # Prediction values
        predicted_alt = results["pred_alt"]
        predicted_az = results["pred_az"]
        # True values
        true_alt = results["true_alt"]
        true_az = results["true_az"]
        true_mc_energy = results["true_mc_energy"]
        show_angular_resolution(predicted_alt, predicted_az, true_alt, true_az, true_mc_energy,
                           percentile, confidence_level, bias_correction,
                           label, [], xlim, ylim, ax)
    try:
        for include in include_requirement:
            ax = ctaplot.plot_angular_resolution_cta_requirement(include, ax)
    except:
        print("Unable to display cta requirements.")
    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()
