"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparations.
"""
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import norm
import numpy as np


"""
Training Metrics
================
"""

def plot_model_training_history(history, training_time, model_name, epochs, save_to=None):
    """
    Generate plot for training and validation Loss from a models history.
    """
    fig = plt.figure(figsize=(12,6))
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

"""
Regression Metrics
==================
"""
def show_prediction_1d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions, targets_values=None, axis=None):
    """
    Display prediction for a 1 dimensional models output.
    """

    # Create new figure
    if axis is None:
        #plt.figure(figsize=(8,8))
        axis = plt.gca()

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
    axis.set_xlabel(targets[0])
    axis.legend()
    return axis


def show_prediction_2d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions, targets_values=None, axis=None):
    """
    Display prediction for a 2 dimensional models output.
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(8,8))
        axis = plt.gca()

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
    axis.set_ylabel(targets[0])
    axis.set_xlabel(targets[1])
    axis.legend()
    return axis

def show_prediction_3d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions, targets_values=None, axis=None):
    """
    Display prediction for a 3 dimensional models output.
    """
    raise NotImplementedError

def plot_assembler_prediction(prediction, prediction_point, targets, target_domains,
                              target_resolutions,  event_id=None, targets_values=None,
                              save_to=None):
    """
    Display the assembled prediction of a event, the probability and the predicted point.
    If targets_values is not None, the target point is included in the figure.
    """

    # Create new Figure
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    # Style
    if event_id is not None:
        title = f"Prediction for event {event_id}"
    else:
        title = f"Prediction for a event"
    plt.title(title)

    # Show prediction according to targets dim 
    if len(targets) == 1:
        ax = show_prediction_1d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions, targets_values, ax)
    elif len(targets) == 2:
        ax = show_prediction_2d(prediction, prediction_point, targets, target_domains, 
                       target_resolutions, targets_values, ax)
    elif len(targets) == 3:
        pass

    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


def show_regression_identity(prediction_points, targets_points, score, target, axis=None):
    """
    Show a comparation between true values and predicted values, it uses a scatter plot
    for a small set or a hexbin plot for a set bigger than 500 samples.

    A nice fit means that points are distributed close to the identity diagonal
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(6,6))
        axis = plt.gca()

    vmin = min(prediction_points.min(), targets_points.min())
    vmax = max(prediction_points.max(), targets_points.max()) 
    if len(targets_points) < 500:
        axis.scatter(x=prediction_points, y=targets_points, alpha=0.6)
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], "r--", label="identity", linewidth=3)
    else:
        x = prediction_points.values
        y = targets_points.values
        x = np.append(x, vmin)
        x = np.append(x, vmax)
        y = np.append(y, vmin)
        y = np.append(y, vmax)
        axis.hexbin(x, y, gridsize=(41,41), cmap="jet")
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], "w--", label="identity", linewidth=3)

    # Style
    axis.set_title(f"Regression on {target.title()}")
    axis.set_ylabel("True Values")
    axis.set_xlabel("Predicted Values")
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
    axis.set_title(f"Residual Error on {target.title()}")
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
    legend = f"mean: {residual_error.mean():.4f} \nstd: {residual_error.std():.4f}"
    if vertical:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1*lim, lim),
                  orientation="horizontal", label=legend)
    else:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1*lim, lim),
                  label=legend)

    # Style
    axis.set_title(f"Residual Error Distribution on {target.title()}")
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
    Display regression metrics for the results of evaluate a assembler model.
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
        show_regression_identity(prediction_points, targets_points, score, target, ax_r)
        # Show error
        ax_e = axs[i][1] if n_targets > 1 else axs[1]
        show_residual_error(prediction_points, targets_points, score, target, ax_e)
        # Show error distribution
        ax_d = axs[i][2] if n_targets > 1 else axs[2]
        show_residual_error_distribution(prediction_points, targets_points, score, target, vertical=True, axis=ax_d)

    # Save or Show
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()