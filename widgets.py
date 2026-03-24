"""
Jupyter ipywidgets for Pseudo-Hamiltonian Neural Networks demonstrator.

Provides interactive widgets for system selection, model configuration,
hyperparameter tuning, and execution mode control in Jupyter notebooks.

Follows the UC2 widget pattern with helper functions,
tuple-based build_widgets, and individual-widget get_args_from_widgets.
"""

import ipywidgets as widgets
import argparse


def create_dropdown(options, value, description):
    """
    Helper function to create a Dropdown widget.
    """
    return widgets.Dropdown(options=options, value=value, description=description)


def create_select_multiple(options, value, description):
    """
    Helper function to create a SelectMultiple widget.
    """
    return widgets.SelectMultiple(options=options, value=value, description=description)


def create_int_slider(value, min_val, max_val, step, description):
    """
    Helper function to create an IntSlider widget.
    """
    return widgets.IntSlider(value=value, min=min_val, max=max_val, step=step, description=description)


def create_float_slider(value, min_val, max_val, step, description, readout_format='.2f'):
    """
    Helper function to create a FloatSlider widget.
    """
    return widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=step,
        description=description, readout_format=readout_format,
    )


def create_int_input(value, description):
    """
    Helper function to create an IntText widget.
    """
    return widgets.IntText(value=value, description=description)


def selector_func(multi_select, options, value, description):
    """
    Create either a SelectMultiple or Dropdown widget based on multi_select flag.
    """
    if multi_select:
        return create_select_multiple(options, value, description)
    return create_dropdown(options, value[0], description)


def build_widgets(systems, integrators, multi_select=False):
    """
    Build all widgets for PHNN experiment configuration.

    Args:
        systems (list): List of dynamical system names for selection.
        integrators (list): List of integrator options for selection.
        multi_select (bool): If True, uses SelectMultiple instead of Dropdown
            for system and integrator widgets.

    Returns:
        Tuple of widgets: (system_type, model_type, integrator, epochs,
                           batch_size, ntrajectories, damping, tmax,
                           hidden_dim, seed)
    """
    system_type_widget = selector_func(
        multi_select, systems, ['mass-spring'], 'System:',
    )
    model_type_widget = create_dropdown(
        ['phnn', 'baseline', 'phnn-forced'],
        'phnn',
        'Model:',
    )
    integrator_widget = selector_func(
        multi_select, integrators, ['midpoint'], 'Integrator:',
    )
    epochs_widget = create_int_slider(30, 5, 200, 5, 'Epochs:')
    batch_size_widget = create_int_slider(32, 16, 256, 16, 'Batch Size:')
    ntrajectories_widget = create_int_slider(300, 5, 500, 5, 'Trajectories:')
    damping_widget = create_float_slider(0.3, 0.0, 1.0, 0.05, 'Damping:')
    tmax_widget = create_float_slider(10.0, 1.0, 50.0, 1.0, 'T max:')
    hidden_dim_widget = create_int_slider(100, 20, 200, 10, 'Hidden Dim:')
    seed_widget = create_int_input(42, 'Seed:')

    return (
        system_type_widget,
        model_type_widget,
        integrator_widget,
        epochs_widget,
        batch_size_widget,
        ntrajectories_widget,
        damping_widget,
        tmax_widget,
        hidden_dim_widget,
        seed_widget,
    )


def create_execution_mode_dropdown():
    """
    Creates a dropdown widget for selecting the execution mode.

    Returns:
        A Dropdown widget for selecting the execution mode.
    """
    execution_mode_dropdown = create_dropdown(
        ['Single Run', 'No Run'],
        'No Run',
        'Execution Mode:',
    )

    def handle_dropdown_change(change):
        """
        Handler for dropdown value change. Outputs mode-specific configurations
        for demonstration.
        """
        config_map = {
            'Single Run': 'Single Run Configuration',
            'No Run': 'Skip runner, go to analysis of existing results',
        }
        custom_variable = config_map.get(change.new, 'No valid option selected')
        print(custom_variable + ' # For demonstration purposes')

    execution_mode_dropdown.observe(handle_dropdown_change, names='value')
    return execution_mode_dropdown


def get_args_from_widgets(
    system_type_widget,
    model_type_widget,
    integrator_widget,
    epochs_widget,
    batch_size_widget,
    ntrajectories_widget,
    damping_widget,
    tmax_widget,
    hidden_dim_widget,
    seed_widget,
):
    """
    Convert individual widget values to an argparse.Namespace.

    Args:
        system_type_widget: System type dropdown/select widget.
        model_type_widget: Model type dropdown widget.
        integrator_widget: Integrator dropdown/select widget.
        epochs_widget: Epochs IntSlider widget.
        batch_size_widget: Batch size IntSlider widget.
        ntrajectories_widget: Number of trajectories IntSlider widget.
        damping_widget: Damping coefficient FloatSlider widget.
        tmax_widget: Maximum simulation time FloatSlider widget.
        hidden_dim_widget: Hidden dimension IntSlider widget.
        seed_widget: Random seed IntText widget.

    Returns:
        argparse.Namespace with all experiment parameters.
    """
    args = argparse.Namespace(
        system_type=system_type_widget.value,
        model_type=model_type_widget.value,
        integrator=integrator_widget.value,
        epochs=epochs_widget.value,
        batch_size=batch_size_widget.value,
        ntrajectories=ntrajectories_widget.value,
        damping=damping_widget.value,
        tmax=tmax_widget.value,
        hidden_dim=hidden_dim_widget.value,
        seed=seed_widget.value,
    )

    return args


def display_widgets(widgets_tuple, exec_widget=None):
    """
    Display all widgets in a VBox layout.

    Args:
        widgets_tuple: Tuple of widgets returned by build_widgets.
        exec_widget: Optional execution mode widget to append.

    Returns:
        A VBox containing all widgets.
    """
    items = list(widgets_tuple)
    if exec_widget is not None:
        items.append(exec_widget)
    box = widgets.VBox(items)
    return box
