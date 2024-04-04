import marimo

__generated_with = "0.3.8"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Analyse einer Zeitreihe durch Zeit-Frequenz-Analyse (Testbeispiel)
        """
    )
    return


app._unparsable_cell(
    r"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    from pandas.plotting import autocorrelation_plot
    from pandas import Series
    import scipy.signal as sp
    from scipy.signal import chirp
    import numpy as np
    #from spectrum import *
    %matplotlib inline
    from stockwell import st
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        r"""
        Erzeugen einer Testzeitreihe
        """
    )
    return


@app.cell
def __():
    #Tmax = 10
    #fs = 500
    #t = np.linspace(0, Tmax, int(Tmax*fs))
    #w = chirp(t, f0=12.5, f1=2.5, t1=10, method='linear')
    #fmin = 0  # Hz
    #fmax = 25  # Hz
    #df = 1./(t[-1]-t[0])  # sampling step in frequency domain (Hz)
    #fmin_samples = int(fmin/df)
    #fmax_samples = int(fmax/df)
    #timeseries = w
    return


@app.cell
def __(pd):
    testdata = pd.read_csv("AE_test.csv", names = ("time","val"))

    timeseries = testdata.val
    t = testdata.time
    return t, testdata, timeseries


@app.cell
def __(t, timeseries):
    t =  t.array
    t = t[1:]
    timeseries = timeseries[1:]
    return t, timeseries


@app.cell
def __(t):
    fmin = 0
    fmax = 5000
    df = 1./(t[-1]-t[0])
    fs = 1/(t[1]-t[0])
    fmin_samples = int(fmin/df)
    fmax_samples = int(fmax/df)
    return df, fmax, fmax_samples, fmin, fmin_samples, fs


@app.cell
def __(plt, timeseries):
    plt.figure()
    plt.plot(timeseries)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Parametrierung der Stockwell Transformation: Zeit- und Frequenzachse
        """
    )
    return


@app.cell
def __(df, fmax_samples, fs, t):
    print("Frequenzauflösung:",df)
    print("Abtastfrequenz:", fs)
    print("f_max_samples:",fmax_samples)
    print("Zeitreihenwerte:",len(t))
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Ausführen der Transformation
        """
    )
    return


@app.cell
def __(fmax_samples, fmin_samples, st, timeseries):
    stock = st.st(timeseries, fmin_samples, fmax_samples)
    return stock,


@app.cell
def __(stock):
    stock.shape
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Plot 
        """
    )
    return


@app.cell
def __(fmax, fmin, np, plt, stock, t, timeseries):
    extent = (t[0], t[len(t)-1], fmin, fmax)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, timeseries)
    ax[0].set(ylabel='amplitude')
    ax[0].axis('tight')
    ax[0].set_xlim([t[0], t[len(t)-1]])
    ax[1].imshow(np.abs(stock), origin='lower', extent=extent)
    #ax[1].pcolormesh(np.abs(stock) , shading='gouraud')
    ax[1].axis('tight')
    ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    plt.show()
    return ax, extent, fig


@app.cell
def __(fs, sp, timeseries):
    ff, tt, spectrogram = sp.spectrogram(timeseries, nperseg = 1024, noverlap = 512, fs = fs)
    return ff, spectrogram, tt


@app.cell
def __(spectrogram):
    spectrogram.shape
    return


@app.cell
def __(ff, fmax, spectrogram):
    # Frequenzbereich dem des Stockwell-Diagramms angleichen
    # ff auf fmax verkürzen
    # spectrogram abschneiden
    spectrogram = spectrogram[ff  < fmax,:]
    return spectrogram,


@app.cell
def __(spectrogram):
    spectrogram.shape
    return


@app.cell
def __(extent, np, plt, spectrogram, t, timeseries):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, timeseries)
    ax[0].set(ylabel='amplitude')
    ax[0].set_xlim([t[0], t[len(t)-1]])
    ax[1].imshow(np.abs(spectrogram), origin='lower', extent=extent)
    #ax[1].pcolormesh(np.abs(spectrogram) , shading='gouraud')
    ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    #ax[1].set_ylim([fmin, fmax])
    #ax[1].set_xlim([t[0], t[len(t)-1]])
    ax[1].axis('tight')
    plt.show()
    return ax, fig


@app.cell
def __(extent, np, plt, spectrogram, stock, t, timeseries):
    fig, ax = plt.subplots(2, 2)
    # Plot Spectrogram
    ax[0,0].plot(t, timeseries)
    ax[0,0].set(ylabel='amplitude')
    ax[0,0].set_xlim([t[0], t[len(t)-1]])
    ax[1,0].imshow(np.abs(spectrogram), origin='lower', extent=extent)
    ax[1,0].set(xlabel='time (s)', ylabel='frequency (Hz)')
    ax[1,0].axis('tight')

    # Plot Stockwell 
    ax[0,1].plot(t, timeseries)
    ax[0,1].set(ylabel='amplitude')
    ax[0,1].set_xlim([t[0], t[len(t)-1]])
    ax[1,1].imshow(np.abs(stock), origin='lower', extent=extent)
    ax[1,1].axis('tight')
    ax[1,1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axs in ax.flat:
        axs.label_outer()
    plt.show()
    return ax, axs, fig


@app.cell
def __(fmax, fmin, np, stock, t):
    y = np.linspace(fmin,fmax,stock.shape[0])
    x = t
    return x, y


@app.cell
def __(np, plt, stock, x, y):
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)
    Z = abs(stock)

    surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()
    return X, Y, Z, ax, fig, surf


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

