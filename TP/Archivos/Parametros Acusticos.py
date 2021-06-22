#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sym
import soundfile as sf
import matplotlib.pyplot as plt
import simpleaudio as sa
import sounddevice as sd
from scipy import signal
from scipy.signal import iirfilter, sosfreqz, sosfilt, freqs, hilbert, fftconvolve, lfilter, medfilt
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.io.wavfile import write
from astropy.table import QTable, Table, Column
from scipy.optimize import least_squares
from scipy import stats


# In[2]:


def carga_datas_wav(audio):
    
    '''Esta funcion permite importar la información archivos .wav y
    almacenar los datos en una lista.
    
    devuelve array 
    
    PARAMETROS:
    
    audio: ruta del archivo de audio.'''
    
    base_data, fs = sf.read(audio)
    
    #print("LISTO PARA UTILIZAR")
    
    return  base_data


# In[3]:


def filtros2(audio):
    
    #Octava - G = 1.0/2.0 / 1/3 de Octava - G=1.0/6.0
    G_octava = 1.0/2.0
    factor_octava = np.power(2, G_octava)
    G_tercio = 1.0/6.0
    factor_tercio = np.power(2, G_tercio)
    fs = 44100
    
    frecuencias_centrales = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000]
    frecuencias_centrales_tercios = [19.69, 24.8, 31.25, 39.37, 49.61, 62.5, 78.75, 99.21, 
                                    125, 157.5, 198.4, 250, 315, 396.9, 500, 630, 793.7, 1000, 1260, 
                                    1587, 2000, 2520, 3175, 4000, 5040, 6350, 8000, 10079, 12699, 16000]
    
    for i in frecuencias_centrales:
        lowerCutoffFrequency_Hz = i / factor_octava;
        upperCutoffFrequency_Hz = i * factor_octava;
        if i < 500:
            grado = 3
        elif i<4000:
            grado = 4
        else:
            grado = 6
        b,a = signal.iirfilter(grado, [lowerCutoffFrequency_Hz, upperCutoffFrequency_Hz],
                                rs=60, btype='band', analog=False,
                                ftype='butter', fs=fs, output='ba')
        w, h = signal.freqz(b,a)
        #plt.plot(w, 20 * np.log10(abs(h)), 'b')
        filtro = signal.lfilter(b, a, audio)
        # Generación de archivo de audio .wav
        filename = '../audio/filtro'+str(i)+'.wav'
        write(filename, fs, filtro)
        
    for i in frecuencias_centrales_tercios:
        lowerCutoffFrequency_Hz = i / factor_tercio;
        upperCutoffFrequency_Hz = i * factor_tercio;
        if i < 40:
            grado = 2
        elif i<500:
            grado = 3
        elif i<4000:
            grado = 4
        else:
            grado = 6
        b,a = signal.iirfilter(grado, [lowerCutoffFrequency_Hz, upperCutoffFrequency_Hz],
                                rs=60, btype='band', analog=False,
                                ftype='butter', fs=fs, output='ba')
        w, h = signal.freqz(b,a)
        #plt.plot(w, 20 * np.log10(abs(h)), 'b')
        filtro = signal.lfilter(b, a, audio)
        # Generación de archivo de audio .wav
        filename = '../audio/filtrotercio'+str(i)+'.wav'
        write(filename, fs, filtro)
 


# In[4]:


def log_scale(A, fs=44100):
    """
    Esta funcion genera la conversión de un archivo de audio a escala logarítmica normalizada,
    arroja como salida un nuevo archivo de audio "audio_logscale.wav".
    
    Parametros
    ----------
    A : NumPy array correspondiente a la señal que se desea transformar.
    fs: int, frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
    
    returns: NumPy array
        Datos de la señal generada."""
    
    R = np.array(20*np.log10(A/np.max(A)))
    
    
    # Generación de archivo de audio .wav
    filename = "audio_logscale.wav"
    write(filename, fs, R)
    
    #print ("nombre del archivo: audio_logscale.wav")
    
    return R


# In[5]:


def suavizado(metodo,audio,fs=44100):
    
    '''Esta funcion aplica suavizado a una señal de entrada en formato array
       
     
       metodo : establezca el tipo de suavizado a realizar Hilbert o medfilt
    
       audio : indicar el la matriz correspondiente al audio al cual se le aplicará la transformada
       
       Devuelve un archivo .wav de la señal suavizada.
       
       si metodo = "hilbert" -------> "audio_Sa.wav"
       
       si metodo = "medfilter" ------> "audio_medfilt.wav" '''  
  
    
    if metodo == "hilbert": 
        Sa = hilbert(audio)
        #convierte en archivo audio .wav
        audio_Sa = np.real(Sa) * (2**15 - 1) / np.max(np.abs(Sa)) #normalizado del audio
        audio_Sa = audio_Sa.astype(np.int16) #convierte el archivo a 16-bit 
        sf.write("audio_Sa.wav", audio_Sa, 44100) 
        print("nombre del archivo: audio_Sa.wav")
        
        
    elif metodo == "medfilter": 
        Mf = medfilt(audio)
        #convierte en archivo audio .wav
        audio_medfilt = np.real(Mf) * (2**15 - 1) / np.max(np.abs(Mf)) #normalizado del audio
        audio_medfilt = audio_medfilt.astype(np.int16) #convierte el archivo a 16-bit 
        sf.write("audio_medfilt.wav", audio_medfilt, 44100) 
        # print("nombre del archivo: audio_medfilt.wav")
    
    
        return audio_medfilt
        

  


# In[6]:


def Schroeder_int(matriz,fs=44100):

    """Esta función aplica la integral de Schroeder, la matriz que se le ingrese.
    
    METODO DE APROXIMACIÓN DE LA INTEGRAL:
    formulas  de cuadratura de newton-cotes compuestas (formula del punto medio compuesta)

    PARAMETROS: 
     
    matriz : array correspondiente a la señal de audio.
    
    fs : sample rate, por default 44100."""
    
    
    t = len(matriz/fs)
    muestras = t*fs
    i_inf1 = 0
    i_sup1 = t*1000 #lo multiplica por 1000 para hacer tender el indice superio de la integral a un numero grande en comparacion a la duración del audio.
    h1 = (i_sup1 - i_inf1)/muestras


    integrate1 = h1*np.cumsum(matriz[::-1]**2)
    #integrate1 = np.cumsum(matriz[::1]**2)

    #esto por si es necesario considerar la integral de 0 a t
    #tengo que definir T
    #T = len(matriz)
    #i_inf2 = 0
    #i_sup2 = T
    #h2 = (i_sup2 - i_inf2)/muestras
    #integrate2 = h2*np.cumsum(matriz**2)



    #hay que hacer esto para que lo grafique "bien". 
    integrate_sch = integrate1[::-1]
    
    #integrate_sch = integrate1 - integrate2
    #integrate_sch = integrate_sch[::-1]

    #grafica resultados
    #length = integrate_sch.shape[0]/fs
    #time = np.linspace(0., length, integrate_sch.shape[0])
    #plt.rcParams['figure.figsize'] = (10,5) # set plot size
    #plt.scatter(time,integrate_sch)
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Amplitud [dB]")

    return integrate_sch


# In[7]:


def min_squares(matriz,fs=44100):
    
    """Esta función calcula la regreción lineal de una base de datos insertada.
     
       PARAMETROS: 
     
       matriz: array correspondiente a los resultados de cierta medición/comportamiento.
       
       fs: sample rate, por default 44100"""
    

    
    #esto elimina todo los "-inf" producto de que la función conversión a escala logaritmica genera log(0) = -inf
    matriz2 = np.delete(matriz,np.where(matriz == np.log10(0)))
    #matriz2 = matriz
    
    
    n = len(matriz2) #cantidad de datos de la matriz
    lenght = n/fs #duracion en segundos
    time = np.linspace(0.,lenght,n)

    sumY = np.sum(matriz2)
    sumX = np.sum(time)
    sumY2 = np.sum(matriz2**2)
    sumX2 = np.sum(time**2)
    sumXY = np.sum(matriz2*time)
    promedio_X = sumX/n
    promedio_Y =sumY/n

    #estimación de la recta de ajuste
    m = (sumX*sumY - n*sumXY) / (sumX**2 - n*sumX2)
    b = promedio_Y - m*promedio_X
    recta_ajuste = m*time + b 

    #calculo de desviación estandar (R)
    σx = np.sqrt((sumX2/n) - promedio_X**2)
    σy = np.sqrt((sumY2/n) - promedio_Y**2)
    σxy = (sumXY/n) - promedio_X*promedio_Y
    R = (σxy/(σx*σy))**2

    #print ("desviación estandar promedio: ",R)
    
    #print("-"*30)
    #print("wainting")
    #print("-"*30)

    #plt.plot(time,matriz2)
    #plt.plot(time,recta_ajuste)
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Amplitud [dBFs]")
    
    return recta_ajuste


# In[8]:


def minimos_cuadrados(data):
    
    i = np.arange(len(data))
    v = data
    param = np.polyfit(i,v,1)
    
    # print('A*x+B')
    # print('A:',param[0] , '   B:',param[1])
    
    # yerr=np.array([0.26,0.5,0.76,1.1,1.6])

    mod = np.polyval(param,i)
    #R = stats.pearsonr(i, v)
    
    
    # plt.plot(i, v,'b.', label='Mediciones')
    # plt.plot(i, mod  ,'r-', label='Ajuste lineal')
    # plt.errorbar(i, v, yerr, fmt='b.', label='')
    
    # plt.xlabel('Corriente [μA]')
    # plt.ylabel('Tensión [V]')
    # plt.plot([], [], ' ', label="R de Pearson = 0.999")
    # plt.legend()
    # plt.grid(b=True)
    
    return mod


# In[9]:


def edt(impulse, fs = 44100):
    '''
    Input ndarray normalized impulse response in dBFS, return Early Decay
    Time value in seconds.
    
    Parameters
    ----------
    impulse: ndarray
        Numpy array containing the impulse response signal in dBFS
    fs: int
        The sample rate of the impluse response array.
    method: str, optional.
        Optional string to determine the desired smoothing method:
            + 'hilbert' for a Hilbert transform
            + 'median' to apply a median filter.
            + 'savgol' to apply a Savitzky-Golay filter.
    window_len: int
        The length of the filter window, must be a positive odd integer.
    polyorder: int
        The order of the polynomial used to fit the samples. 
        This value value must be less than window_length.
    '''
    for i in impulse:
        if i > 0:
            raise ValueError('Input should have no positive values.')
            
    vectorT = np.arange(len(impulse))/fs # Tiempo del impulso
    index_edt = np.where(((impulse <= -1) & (impulse >= -10))) # Crea array desde -1 a -10
    coeff_edt = np.polyfit(vectorT[index_edt[0]],
                       impulse[index_edt[0]], 1)
    
    fit_edt = coeff_edt[0]*vectorT + coeff_edt[1] # Recta cuadrados minimos
    edt = len(fit_edt[fit_edt>=-10])/fs 
    
    return edt

def t60(impulse, fs = 44100, method = 't30'):
    '''
    Input ndarray normalized impulse response in dBFS, returns the t60 value
    in seconds. Method should be chosen according to the background noise 
    level of the input signal.
    
    Parameters
    ----------
    impulse: ndarray
        Numpy array containing the impulse response signal in dBFS
    fs: int
        The sample rate of the impluse response array.
    method: str, optional.
        Optional string to determine the desired t60 method:
            + 't10' calculate from t10.
            + 't20' calculate from t20.
            + 't30' calculate from t30.
    '''
    
    for i in impulse:
        if i > 0:
            raise ValueError('Input should have no positive values.')
    
    vectorT = np.arange(len(impulse))/fs 
    
    if method == 't10':
        index_t10 = np.where(((impulse <= -5) & (impulse >= -15)))
        coeff_t10 = np.polyfit(vectorT[index_t10[0]], impulse[index_t10[0]], 1)
        fit_t10 = coeff_t10[0]*vectorT + coeff_t10[1]
        t10 = len(fit_t10[fit_t10>=-10])/fs
        t60 = t10*6
        
    elif method == 't20':
        index_t20 = np.where(((impulse <= -5) & (impulse >= -25)))
        coeff_t20 = np.polyfit(vectorT[index_t20[0]], impulse[index_t20[0]], 1)
        fit_t20 = coeff_t20[0]*vectorT + coeff_t20[1]
        t20 = len(fit_t20[fit_t20>=-20])/fs
        t60 = t20*3

    elif method == 't30':
        index_t30 = np.where(((impulse <= -5) & (impulse >= -35)))
        coeff_t30 = np.polyfit(vectorT[index_t30[0]], impulse[index_t30[0]], 1)
        fit_t30 = coeff_t30[0]*vectorT + coeff_t30[1]
        t30 = len(fit_t30[fit_t30>=-30])/fs
        t60 = t30*2
        
    else:
        raise ValueError('Invalid Method.')
        
    return t60
        
def d50(impulse, fs):
    '''
    Input ndarray normalized impulse response in dBFS, return d50 value.
    The function uses Numpy to integrate the impulse.
    
    Parameters
    ----------
    impulse: ndarray
        Numpy array containing the impulse response signal in dBFS
    fs: int
        The sample rate of the impluse response array.
    '''
    t = round(0.050 * fs)
    d50 = 100 * (np.sum(impulse[:t]) / np.sum(impulse))
    
    return d50

def c80(impulse, fs):
    '''
    Input ndarray normalized impulse response in dBFS, return c80 value.
    The function uses Numpy to integrate the impulse.
    
    Parameters
    ----------
    impulse: ndarray
        Numpy array containing the impulse response signal in dBFS
    fs: int
        The sample rate of the impluse response array.
    '''
    t = round(0.080 * fs)
    c80 = 10 * np.log10(np.sum(impulse[:t]) / np.sum(impulse[t:]))
    
    return c80


# In[10]:


def parametros1(impulso, fs=44100):
    
    """
    Genera un array con los datos de un archivo de audio en formato .wav
    
    Parametros
    ----------
    impulso: NumPy array
        Datos de la señal a procesar
    fs: int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
        
    returns: Table
        Datos de parametros acusticos en una tabla.
    
    Ejemplo
    -------
        data = carga_datas_wav("nombre_arcchivo.wav")
        parametros1(data)
    """
    
    # Filtrado
    filtros2(impulso)
    
    # Inicio de listas
    EDT = []
    T10 = []
    T20 = []
    T30 = []
    D50 = []
    C80 = []
    
    # Frecuencias Octavas
    frecuencias = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000]
    
    for i in frecuencias:
        # Cargar archivo
        data_filtro = carga_datas_wav('../audio/filtro'+str(i)+'.wav')
        
        #Suavizado
        audio_suavizado = suavizado("medfilter", data_filtro)
        
        # Integral Schroeder
        integrate_sch  = Schroeder_int(audio_suavizado)
        
        # Escala logaritmica
        data_log = log_scale(integrate_sch)
        
        # Minimos cuadrados
        data = min_squares(data_log, fs)
        
        #Parametros
        index_edt = np.where(((data <= np.max(data)) & (data >= np.max(data)-10)))
        EDT.append(len(index_edt[0])/fs)
        index_t10 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-15)))
        #fit_t10 = min_squares(data_log[index_t10], fs)
        T10.append((len(index_t10[0])/fs)*6)
        index_t20 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-25)))
        T20.append((len(index_t20[0])/fs)*3)
        index_t30 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-35)))
        T30.append((len(index_t30[0])/fs)*2)
        #T10.append(t60(data_log, fs, method = 't10'))
        #T20.append(t60(data_log, fs, method = 't20'))
        #T30.append(t60(data_log, fs, method = 't30'))
        D50.append(d50(data, fs))
        C80.append(c80(data, fs))
    
    Tabla = Table([frecuencias[:], EDT[:], T10[:], T20[:], T30[:], D50[:], C80[:]], names=('Frecuencias centrales', 'EDT', 'T10', 'T20', 'T30', 'D50', 'C80'))
    
    return Tabla


# In[11]:


# TEST 1

base_data_prueba = carga_datas_wav("impulseresponseheslingtonchurch-001.wav")
parametros1(base_data_prueba)


# In[122]:


def parametros2(impulso, fs=44100):
    
    """
    Genera un array con los datos de un archivo de audio en formato .wav
    
    Parametros
    ----------
    impulso: NumPy array
        Datos de la señal a procesar
    fs: int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
        
    returns: Table
        Datos de parametros acusticos en una tabla.
    
    Ejemplo
    -------
        data = carga_datas_wav("nombre_arcchivo.wav")
        parametros2(data)
    """
    
    # Filtrado
    filtros2(impulso)
    
    # Inicio de listas
    EDT = []
    T10 = []
    T20 = []
    T30 = []
    D50 = []
    C80 = []
    
    # Frecuencias Octavas
    frecuencias = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000]
    
    for i in frecuencias:
        # Cargar archivo
        data_filtro = carga_datas_wav('../audio/filtro'+str(i)+'.wav')
        
        #Suavizado
        audio_suavizado = suavizado("medfilter", data_filtro)
        
        # Integral Schroeder
        integrate_sch  = Schroeder_int(audio_suavizado)
        
        # Escala logaritmica
        data_log = log_scale(integrate_sch)
        
        # Minimos cuadrados
        data = min_squares(data_log, fs)
        
        #Parametros
        #index_edt = np.where(((data <= np.max(data)) & (data >= np.max(data)-10)))
        #EDT.append(len(index_edt[0])/fs)
        #index_t10 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-15)))
        #fit_t10 = min_squares(data_log[index_t10], fs)
        #T10.append((len(index_t10[0])/fs)*6)
        #index_t20 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-25)))
        #T20.append((len(index_t20[0])/fs)*3)
        #index_t30 = np.where(((data <= np.max(data)-5) & (data >= np.max(data)-35)))
        #T30.append((len(index_t30[0])/fs)*2)
        EDT.append(edt(data_log, fs))
        T10.append(t60(data_log, fs, method = 't10'))
        T20.append(t60(data_log, fs, method = 't20'))
        T30.append(t60(data_log, fs, method = 't30'))
        D50.append(d50(data, fs))
        C80.append(c80(data, fs))
    
    Tabla = Table([frecuencias[:], EDT[:], T10[:], T20[:], T30[:], D50[:], C80[:]], names=('Frecuencias centrales', 'EDT', 'T10', 'T20', 'T30', 'D50', 'C80'))
    
    return Tabla


# In[123]:


# TEST 2

base_data_prueba = carga_datas_wav("../IR_samples/r1-nuclear-reactor-hall/mono/r1_omni.wav")
parametros2(base_data_prueba)


# In[ ]:




