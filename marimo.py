import marimo

__generated_with = "0.4.7"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Analyse einer Zeitreihe durch Zeit-Frequenz-Analyse (Testbeispiel)
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    from pandas.plotting import autocorrelation_plot
    from pandas import Series
    import scipy.signal as sp
    from scipy.signal import chirp
    import numpy as np
    #from spectrum import *
    from stockwell import st
    return (
        Series,
        autocorrelation_plot,
        chirp,
        datetime,
        np,
        pd,
        plt,
        sp,
        st,
    )


@app.cell
def __(mo):
    mo.md(r"Erzeugen einer Testzeitreihe")
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
    t =  t.array
    t = t[1:]
    timeseries = timeseries[1:]
    return t, testdata, timeseries


@app.cell
def __(t):
    fmin = 0
    fmax = 1500
    df = 1./(t[-1]-t[0])
    fs = 1/(t[1]-t[0])
    fmin_samples = int(fmin/df)
    fmax_samples = int(fmax/df)
    return df, fmax, fmax_samples, fmin, fmin_samples, fs


@app.cell
def __(plt, timeseries):
    #plt.figure()
    plt.plot(timeseries)
    return


@app.cell
def __(df, fmax_samples, fs, mo, t):
    mo.md(
        r"""
        Parametrierung der Stockwell Transformation: Zeit- und Frequenzachse
        """
    )
    print("Frequenzauflösung:",df)
    print("Abtastfrequenz:", fs)
    print("f_max_samples:",fmax_samples)
    print("Zeitreihenwerte:",len(t))
    return


@app.cell
def __(fmax_samples, fmin_samples, mo, st, timeseries):
    mo.md(
        r"""
        Ausführen der Transformation
        """
    )
    stock = st.st(timeseries, fmin_samples, fmax_samples)
    print(stock.shape)
    return stock,


@app.cell
def __(fmax, fmin, np, plt, stock, t, timeseries):
    extent = (t[0], t[len(t)-1], fmin, fmax)
    _fig, _ax = plt.subplots(2, 1)
    _ax[0].plot(t, timeseries)
    _ax[0].set(ylabel='amplitude')
    _ax[0].axis('tight')
    _ax[0].set_xlim([t[0], t[len(t)-1]])
    _ax[1].imshow(np.abs(stock), origin='lower', extent=extent)
    #ax[1].pcolormesh(np.abs(stock) , shading='gouraud')
    _ax[1].axis('tight')
    _ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    #plt.show()
    return extent,


@app.cell
def __(fmax, fs, sp, timeseries):
    ff, tt, spectrogram = sp.spectrogram(timeseries, nperseg = 1024, noverlap = 512, fs = fs)
    print(spectrogram.shape)
    # Frequenzbereich dem des Stockwell-Diagramms angleichen
    # ff auf fmax verkürzen
    # spectrogram abschneiden
    spectrogram = spectrogram[ff  < fmax,:]
    return ff, spectrogram, tt


@app.cell
def __(extent, np, plt, spectrogram, t, timeseries):
    _fig, _ax = plt.subplots(2, 1)
    _ax[0].plot(t, timeseries)
    _ax[0].set(ylabel='amplitude')
    _ax[0].set_xlim([t[0], t[len(t)-1]])
    _ax[1].imshow(np.abs(spectrogram), origin='lower', extent=extent)
    #ax[1].pcolormesh(np.abs(spectrogram) , shading='gouraud')
    _ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    #ax[1].set_ylim([fmin, fmax])
    #ax[1].set_xlim([t[0], t[len(t)-1]])
    _ax[1].axis('tight')
    #plt.show()
    return


@app.cell
def __(extent, np, plt, spectrogram, stock, t, timeseries):
    _fig, _ax = plt.subplots(2, 2)
    # Plot Spectrogram
    _ax[0,0].plot(t, timeseries)
    _ax[0,0].set(ylabel='amplitude')
    _ax[0,0].set_xlim([t[0], t[len(t)-1]])
    _ax[1,0].imshow(np.abs(spectrogram), origin='lower', extent=extent)
    _ax[1,0].set(xlabel='time (s)', ylabel='frequency (Hz)')
    _ax[1,0].axis('tight')

    # Plot Stockwell 
    _ax[0,1].plot(t, timeseries)
    _ax[0,1].set(ylabel='amplitude')
    _ax[0,1].set_xlim([t[0], t[len(t)-1]])
    _ax[1,1].imshow(np.abs(stock), origin='lower', extent=extent)
    _ax[1,1].axis('tight')
    _ax[1,1].set(xlabel='time (s)', ylabel='frequency (Hz)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axs in _ax.flat:
        axs.label_outer()
    #plt.show()
    return axs,


@app.cell
def __(fmax, fmin, np, stock, t):
    y = np.linspace(fmin,fmax,stock.shape[0])
    x = t
    return x, y


@app.cell(disabled=True)
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
