import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
import pandas as pd
import re


def resolve_key(key, index, name):
    # Convert key (int) to string
    if key in index.keys():
        return index[key]
    # If key is already a string
    elif key in index.values():
        return key
    else:
        # Handle incorrect values
        raise ValueError(f'{repr(key)} is not a valid value for {name}; available values are: {index}')


def resolve_plot_type(plot_type):
    plot_types = {0:'velocity', 1:'amplitude', 2:'correlation', 3:'orientation', 4:'PT', 5:'orientation+PT'}
    return resolve_key(plot_type, plot_types, 'plot_type')


def get_time_xlim1(xlim, xref):
    xlim = np.asarray(xlim)
    xref = np.asarray(xref)
    unit = np.datetime_data(xref.dtype)[0]
    xlim = xlim.astype(f'timedelta64[{unit}]')
    
    return xlim + xref


def get_time_xlim(xlim, xref):
    xlim = np.asarray(xlim)
    xref = np.asarray(xref)

    # Convert object dtype arrays to datetime64[ns] if elements are datetimes
    if np.issubdtype(xlim.dtype, np.object_):
        if all(isinstance(x, (np.datetime64, dt.datetime, pd.Timestamp)) for x in xlim):
            xlim = xlim.astype('datetime64[ns]')

    # If already datetime64, return as-is
    if np.issubdtype(xlim.dtype, np.datetime64):
        return xlim

    # If timedelta (Python, NumPy, or pandas), convert and add to xref
    if all(isinstance(x, (np.timedelta64, dt.timedelta, pd.Timedelta)) for x in xlim):
        unit = np.datetime_data(xref.dtype)[0]
        xlim = xlim.astype(f'timedelta64[{unit}]')
        return xlim + xref

    raise TypeError("xlim must contain either datetimes or timedeltas.")


def rescale_ylim_to_visible_data(ax, x_mode):
    xlim = ax.get_xlim()

    if x_mode == 'time':
        # Convert float xlim to datetime64
        xlim = mdates.num2date(xlim)  # gives Python datetime
        xlim = np.array(xlim).astype('datetime64[ns]')

    ymin, ymax = np.inf, -np.inf

    for line in ax.get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        if x_mode == 'time' and not np.issubdtype(xdata.dtype, np.datetime64):
            xdata = mdates.num2date(xdata)
            xdata = np.array(xdata).astype('datetime64[ns]')

        mask = (xdata >= xlim[0]) & (xdata <= xlim[1])
        if np.any(mask):
            y_visible = ydata[mask]
            ymin = min(ymin, np.min(y_visible))
            ymax = max(ymax, np.max(y_visible))

    if np.isfinite(ymin) and np.isfinite(ymax):
        ax.set_ylim(ymin, ymax)
    

def get_x_data(x_mode, ds):
    x_modes = {0: 'time', 1: 'index'}
    x_mode = resolve_key(x_mode, x_modes, 'x_mode')
    x = ds['TIME'] if x_mode == 'time' else list(range(len(ds['TIME'])))

    return x, x_mode


def get_y_data(y_mode, ds):
    y_modes = {0: 'cell', 1: 'range'}
    y_mode = resolve_key(y_mode, y_modes, 'y_mode')
    y = ds['CN'] if y_mode == 'cell' else ds['CP'].isel(TIME=-1)

    return y, y_mode


def get_plot_args(plot_type, cy='ENU'):
    # Convert to string
    plot_type = resolve_plot_type(plot_type)

    if plot_type == 'velocity':
        if cy == 'ENU':
            variables = ['VN','VE','VU','VU2']
        if cy == 'XYZ':
            variables = ['VX','VY','VZ','VZ2']
        if cy == 'BEAM':
            variables = ['V1','V2','V3','V4']
        names = variables
        
    if plot_type == 'amplitude':
        variables = ['A1','A2','A3','A4']
        names = ['Beam 1', 'Beam 2', 'Beam 3', 'Beam 4']
    
    if plot_type == 'correlation':
        variables = ['C1','C2','C3','C4']
        names = ['Beam 1', 'Beam 2', 'Beam 3', 'Beam 4']

    if plot_type == 'orientation':
        variables = ['H','PI','R']
        names = ['Heading', 'Pitch', 'Roll']
        
    if plot_type == 'PT':
        variables = ['P', 'T']
        names = ['Pressure', 'Temperature']

    if plot_type == 'orientation+PT':
        variables = ['H','PI','R','P','T']
        names = ['Heading', 'Pitch', 'Roll', 'Pressure', 'Temperature']
    
    return variables, names, plot_type


def average_pixels(arr, xpixavg=1, ypixavg=1):
    is_1d = arr.ndim == 1
    arr = np.atleast_2d(arr)

    ny, nx = arr.shape
    ny_crop = ny - ny % ypixavg
    nx_crop = nx - nx % xpixavg

    arr_cropped = arr[:ny_crop, :nx_crop]
    reshaped = arr_cropped.reshape(ny_crop // ypixavg, ypixavg, nx_crop // xpixavg, xpixavg)
    block_avg = reshaped.mean(axis=(1, 3))
    upscaled = np.kron(block_avg, np.ones((ypixavg, xpixavg)))

    # Pad to original size
    result = np.full_like(arr, np.nan, dtype=float)
    result[:upscaled.shape[0], :upscaled.shape[1]] = upscaled

    return result.squeeze() if is_1d else result


def plot_beam_data(ds, plot_type, x_mode=0, y_mode=0, xlim=None, ylim=None, clim=None, 
                   cmap='viridis', plot_only=None, xpixavg=1, ypixavg=1, dpi=150,
                   datefmt='%H:%M'):
    # Get x and y data
    x, x_mode = get_x_data(x_mode, ds)
    y, y_mode = get_y_data(y_mode, ds)
    
    # Get coordinate system of velocities
    cy = str(np.atleast_1d(ds['CY'])[0])

    # Get variables and names
    variables, names, plot_type = get_plot_args(plot_type, cy)

    # Filter by plot_only
    if plot_only is not None:
        variables = [variables[i] for i in range(len(variables)) if i in plot_only]
        names = [names[i] for i in range(len(names)) if i in plot_only]
    N_plots = len(variables)

    # Compute cmin/cmax
    all_data = [ds[var].values for var in variables]
    if clim is None:
        cmin = min(np.nanmin(Z) for Z in all_data)
        cmax = max(np.nanmax(Z) for Z in all_data)
    else:
        try:
            cmin, cmax = clim
        except TypeError:
            cmin, cmax = -clim, clim
    
    # Define figure and axes
    fig, axs = plt.subplots(N_plots, 1, figsize=(8, 2 * N_plots), dpi=dpi, sharex=True)
    axs = np.atleast_1d(axs)

    # Plot each subplot
    cmeshes = []
    for i, (var, name) in enumerate(zip(variables, names)):
        ax = axs[i]
        Z = ds[var].T
        Z = average_pixels(Z.values, xpixavg, ypixavg)
        cmesh = ax.pcolormesh(x, y, Z, vmin=cmin, vmax=cmax, cmap=cmap)
        cmeshes.append(cmesh)
    
        # Set xlabel
        if x_mode == 'index':
            ax.set_xlabel('Number of measurement')
        elif x_mode == 'time':
            ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))

        # Set ylabel
        if y_mode == 'cell':
            ax.set_ylabel('Cell number')
        elif y_mode == 'range':
            ax.set_ylabel('Range (m)')

        # Set ylimit
        if ylim is not None:
            ax.set_ylim(*ylim)
        
        # Set plot title and aspect
        ax.minorticks_on()
        ax.set_title(name, loc='left', fontsize=13)
        ax.set_aspect('auto')

    # Set xlimit (once is enough)
    if xlim is not None:
        if x_mode == 'time':
            xlim = get_time_xlim(xlim, x[-1])
        ax.set_xlim(*xlim)

    # Determine if clim exceeds data extent, in which case, add 'extend' indicators
    all_min = np.nanmin([np.nanmin(Z) for Z in all_data])
    all_max = np.nanmax([np.nanmax(Z) for Z in all_data])
    extend = (
        'both' if cmax < all_max and cmin > all_min else
        'max'  if cmax < all_max else
        'min'  if cmin > all_min else
        'neither'
    )
    
    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.025, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(cmeshes[0], cax=cbar_ax, extend=extend)

    # Set colorbar label
    label = plot_type.capitalize()
    if plot_type == 'velocity':
        label += ' (m/s)'
    if plot_type == 'amplitude':
        label += ' (dB)'
    if plot_type == 'correlation':
        label += ' (%)'
    cbar.set_label(label=label, size=13)

    plt.close(fig)
    return fig


def plot_sensor_data(ds, plot_type, x_mode=0, xlim=None, ylim=None, xpixavg=1, dpi=150, 
                     datefmt='%H:%M', grid=False):
    orientation_colors = ['k', 'r', 'g']
    PT_colors = ['tab:blue', 'tab:orange']
    
    # Helper function
    def configure_ax(ax, gridcolor='gray'):
        # Set xlabel
        if x_mode == 'index':
            ax.set_xlabel('Number of measurement')
        elif x_mode == 'time':
            ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))

        # Set ylimit
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.minorticks_on()
        if grid:
            ax.grid(alpha=0.3, lw=1, color=gridcolor)
    
    # Get x data
    x, x_mode = get_x_data(x_mode, ds)

    # Get variables and names
    variables, names, plot_type = get_plot_args(plot_type)

    
    # ORIENTATION
    if plot_type == 'orientation':
        fig = plt.figure(figsize=(8, 2), dpi=dpi)
        ax = fig.add_subplot()
        
        # Plot heading, pitch, roll data
        for i, (var, name) in enumerate(zip(variables, names)):
            color = orientation_colors[i]
            y = ds[var]
            y = average_pixels(y.values, xpixavg)
            ax.plot(x, y, color=color, label=name)
            configure_ax(ax)

        ax.legend()
        ax.set_ylabel('Degrees (°)')

    
    # PRESSURE AND TEMPERATURE
    if plot_type == 'PT':
        fig = plt.figure(figsize=(8, 2), dpi=dpi)
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        # Plot pressure and temperature
        for i, (var, ax) in enumerate(zip(variables, (ax1, ax2))):
            color = PT_colors[i]
            y = ds[var]
            y = average_pixels(y.values, xpixavg)
            ax.plot(x, y, color=color)
            configure_ax(ax, gridcolor=color)

        ax1.set_ylabel('Pressure (dBar)', color=PT_colors[0])
        ax1.tick_params(which='both', axis='y', colors=PT_colors[0])
        ax2.set_ylabel('Temperature (°C)', color=PT_colors[1])
        ax2.tick_params(which='both', axis='y', colors=PT_colors[1])


    # ORIENTATION, PRESSURE AND TEMPERATURE
    if plot_type in ['orientation+PT']:
        fig, axs = plt.subplots(2, 1, figsize=(8, 4), dpi=dpi, sharex=True)

        # Plot heading, pitch, roll data
        ax = axs[0]
        for i, (var, name) in enumerate(zip(variables[:3], names[:3])):
            color = orientation_colors[i]
            y = ds[var]
            y = average_pixels(y.values, xpixavg)
            ax.plot(x, y, color=color, label=name)
            configure_ax(ax)

        ax.legend()
        ax.set_ylabel('Degrees (°)')
        ax.set_title('Orientation', loc='left', fontsize=13)

        # Plot pressure and temperature
        ax1 = axs[1]
        ax2 = ax1.twinx()
        for i, (var, ax) in enumerate(zip(variables[3:], (ax1, ax2))):
            color = PT_colors[i]
            y = ds[var]
            y = average_pixels(y.values, xpixavg)
            ax.plot(x, y, color=color)
            configure_ax(ax, gridcolor=color)

        ax1.set_ylabel('Pressure (dBar)', color=PT_colors[0])
        ax1.tick_params(which='both', axis='y', colors=PT_colors[0])
        ax2.set_ylabel('Temperature (°C)', color=PT_colors[1])
        ax2.tick_params(which='both', axis='y', colors=PT_colors[1])
        ax.set_title('Pressure & temperature', loc='left', fontsize=13)
    
    # Set xlimit
    if xlim is not None:
        if x_mode == 'time':
            xlim = get_time_xlim(xlim, x[-1])
        ax.set_xlim(*xlim)
        rescale_ylim_to_visible_data(ax, x_mode)
    
    fig.tight_layout()
    plt.close(fig)
    return fig
        

def plot_data(ds, plot_type, x_mode=0, y_mode=0, xlim=None, ylim=None, clim=None, cmap='viridis', 
              plot_only=None, xpixavg=1, ypixavg=1, dpi=150, datefmt='%H:%M', grid=False):
    # Beam data
    if plot_type in [0,1,2]:
        fig = plot_beam_data(ds, plot_type, x_mode, y_mode, xlim, ylim, clim, 
                             cmap, plot_only, xpixavg, ypixavg, dpi, datefmt)
    # Plot sensor data
    if plot_type in [3,4,5]:
        fig = plot_sensor_data(ds, plot_type, x_mode, xlim, ylim, xpixavg, dpi,
                               datefmt, grid)
    
    return fig

