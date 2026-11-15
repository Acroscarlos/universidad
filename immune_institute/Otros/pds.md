\documentclass[journal, a4paper]{IEEEtran}
\usepackage{bookmark}
\usepackage{booktabs}
% --- PAQUETES BÁSICOS ---
\usepackage[utf8]{inputenc} % Codificación
\usepackage[spanish]{babel} % Idioma español
\usepackage{amsmath, amssymb, amsfonts} % Paquetes matemáticos para PDS
\usepackage{graphicx} % Para insertar gráficas
\usepackage{cite} % Para gestionar la bibliografía
\usepackage{hyperref} % Para enlaces en correos

% --- BLOQUE 1: ENCABEZADO ---

% Título Principal
\title{Detección Acústica de Sobrevivientes en Estructuras Colapsadas: Un Enfoque no Invasivo Basado en PDS y Modelos Recurrentes Convolucionales}
% Otras opciones de título:
% \title{Clasificación Pasiva de Vibraciones Estructurales en Mampostería mediante Sensores Piezoeléctricos de Contacto y Redes Híbridas para Búsqueda y Rescate Urbano}
% \title{Sistema Táctico USAR V2.1: Discriminación de Señales de Vida Bajo Escombros mediante Captación Pasiva y PDS}

\author{
    \IEEEauthorblockN{Carlos Carreño Castaños} \\
    \vspace{0.2cm}
    \IEEEauthorblockA{\textit{Postgrado en Ingeniería Eléctrica / Procesamiento Digital de Señales} \\
    \textit{Facultad de Ingeniería, Escuela de Ingeniería Eléctrica, Universidad Central de Venezuela (UCV)}\\
    Caracas, Venezuela \\
    Correo electrónico: carlosaccantonio@gmail.com}
}

\begin{document}

\maketitle

\begin{abstract}
En operaciones de Búsqueda y Rescate Urbano (USAR) tras colapsos estructurales, las lesiones impiden a las víctimas emitir gritos, mientras que el polvo inutiliza los sensores ópticos. Este trabajo presenta la captación pasiva de vibraciones elásticas de baja frecuencia (150--7500 Hz) en mampostería mediante el sensor piezoeléctrico de contacto HIKMICRO AD21P, combinado con un pipeline de Procesamiento Digital de Señales (PDS). El sistema discrimina patrones rítmicos de impacto entre el ruido de la Zona Cero. Para acondicionar la señal, el canal analógico se digitaliza a 16 kHz, cumpliendo el criterio de Nyquist. La cuantificación PCM de 16 bits, combinada con un piso de estática a -80 dBFS que actúa como dither natural, previene la distorsión de micro-señales. Mediante un filtrado IIR Butterworth de 4.º orden de fase cero, se eliminan la deriva de línea base y el ruido estructural sin introducir retardo de grupo. Posteriormente, se sintetiza un corpus de 65\,576 tensores acústicos para alimentar una red híbrida Convolucional Recurrente (CRNN \textit{Wide \& Deep}) que extrae descriptores discretos y espectrogramas Mel, alcanzando un F1-Score Macro de 0.8631 y un ROC-AUC de 0.9927.
\end{abstract}


\begin{IEEEkeywords}
Procesamiento Digital de Señales, Detección Acústica, Criterio de Nyquist-Shannon, Cuantificación PCM, Filtrado de Fase Cero, Búsqueda y Rescate Urbano (USAR).
\end{IEEEkeywords}

% --- FIN DEL BLOQUE 1 ---

% --- INICIO DEL BLOQUE 2 (INTRODUCCIÓN) ---
\section{Introducción}

En un colapso estructural, la severa atenuación acústica de los escombros dificulta la detección de supervivientes. El ruido de la maquinaria pesada superficial enmascara aún más las tenues emisiones de auxilio. La escucha aérea resulta ineficaz; la interfaz aire-concreto refleja casi toda la energía y bloquea la captación de señales biológicas.

Para superar este desacople, el sistema USAR V2.1 acopla el transductor piezoeléctrico HIKMICRO AD21P directamente a la mampostería. Esto permite registrar pasivamente las ondas elásticas mecánicas que viajan por la estructura, prescindiendo de la vía aérea. Así, el sistema capta señales de baja frecuencia (150--7500 Hz) y aísla la firma de vida de la interferencia superficial.

Las señales capturadas presentan un alto componente estocástico debido al ruido térmico, la interferencia mecánica y la atenuación elástica del concreto. El Procesamiento Digital de Señales (PDS) actúa como puente entre este entorno físico continuo y el análisis digital determinista. Sin una etapa de acondicionamiento y digitalización, la información esencial quedaría sepultada bajo el piso de ruido y deformada por distorsiones de fase.

Este artículo articula la teoría de PDS con la plataforma USAR V2.1. Sus objetivos son: 1) caracterizar las señales acústicas según su física estadística y energética; 2) definir el proceso de conversión A/D, validando el Criterio de Nyquist-Shannon ($f_s = 16\text{ kHz}$) y evaluando el error de cuantificación PCM; 3) detallar el filtrado de fase cero y la extracción de características; 4) modelar la síntesis estocástica acoplada a una red neuronal (CRNN); y 5) demostrar gráficamente la efectividad del pipeline para preservar la alineación temporal.

% --- FIN DEL BLOQUE 2 (INTRODUCCIÓN) ---

% --- INICIO DEL BLOQUE 2 (SECCION II) ---
\section{Funciones Discretas, Continuas y Propiedades}

\subsection{Clasificación Física y Parámetros Operativos}
La señal continua $x_a(t)$ proviene de ondas elásticas que se propagan a través de la mampostería y el concreto armado. Generadas por emisiones vocales o impactos intencionales de las víctimas, estas oscilaciones sufren una fuerte atenuación. La densidad del material actúa como un filtro pasa-bajo natural, restringiendo el ancho de banda efectivo de las señales de vida a unos 1000 Hz.

Para modelar los eventos sísmico-acústicos en el dominio discreto, el sistema se fundamenta en tres secuencias elementales de PDS \cite{Oppenheim}: el impulso unitario $\delta[n]$ (definido como $\delta[n]=1$ para $n=0$ y $0$ para $n \neq 0$), el escalón unitario $u[n] = \sum_{k=-\infty}^{n} \delta[k]$ (que garantiza la causalidad temporal de la señal para $n \ge 0$), y la secuencia exponencial decreciente $a^n u[n]$, la cual describe la disipación de energía elástica en los escombros.

La respuesta del sensor HIKMICRO AD21P acota la transducción al rango de 150 a 7500 Hz. Las vibraciones incidentes producen tensiones en el orden de microvoltios ($\mu$V) a milivoltios (mV), acondicionadas posteriormente por el hardware. El sistema ofrece un rango dinámico efectivo de 80 a 96 dB, delimitado por el ruido térmico ($\approx -80$ dBFS) y la saturación analógica.
\subsection{Clasificación Estadística y Modelo Energético}

La señal $x_a(t)$ superpone dos regímenes estadísticos. El colapso estructural presenta un proceso estocástico no estacionario debido al siseo térmico y los crujidos mecánicos. En contraste, las señales de auxilio se modelan como transitorios percusivos pseudo-deterministas. Estos corresponden a patrones rítmicos (código Morse) con envolvente de decaimiento exponencial que destacan sobre el ruido de fondo.

Dado que el sistema segmenta el canal en ventanas de observación de $T_w = 4.0\text{ s}$ ($N = 64\,000$ muestras a $f_s = 16\text{ kHz}$), la señal se analiza como una \textit{Señal de Energía} de duración finita. La energía total $E_w$ satisface la integrabilidad en tiempo continuo:


\begin{equation}
    E_w = \int_0^{T_w} |x_a(t)|^2 \, dt < \infty
\end{equation}

Esta propiedad de energía acotada se traslada directamente al dominio discreto mediante la suma finita de cuadrados, garantizando la convergencia uniforme de la Transformada de Fourier en Tiempo Discreto (DTFT), la existencia de la transformada y la estabilidad de las operaciones de filtrado en el procesador:
\begin{equation}
    E_w = \sum_{n=0}^{N-1} |x[n]|^2
\end{equation}

% --- FIN DEL BLOQUE 2 (SECCION II) ---

% --- INICIO DEL BLOQUE 3 (SECCION III) ---


% \section{Etapa de Conversión y Acondicionamiento}

\subsection{Muestreo Discreto y Criterio de Nyquist (Conversor A/D)}
La discretización del dominio temporal se define mediante la transformación de la señal analógica continua $x_a(t)$ en la secuencia discreta $x[n] = x_a(n T_s)$ mediante un muestreo periódico a una frecuencia de $f_s = 16\,000 \text{ Hz}$. Esto establece un periodo de muestreo constante de $T_s = \frac{1}{f_s} = 62.5\ \mu\text{s}$. Dado que la respuesta en frecuencia del transductor piezoeléctrico HIKMICRO AD21P acota el ancho de banda acústico a una frecuencia máxima de $f_{\max} = 7500 \text{ Hz}$, la frecuencia teórica mínima de Nyquist requerida es $f_{\text{Nyquist}} = 2 f_{\max} = 15\,000 \text{ Hz}$. 

Al fijar la tasa de digitalización a $16 \text{ kHz}$, el sistema satisface el Teorema de Nyquist-Shannon ($f_s \ge 2 f_{\max}$), posicionando la frecuencia de corte de plegado en $f_N = \frac{f_s}{2} = 8000 \text{ Hz}$. Esta arquitectura guarda una banda de guarda espectral de $500 \text{ Hz}$ ($8000 \text{ Hz} - 7500 \text{ Hz}$), la cual absorbe la zona de transición de los filtros analógicos de acondicionamiento y previene el fenómeno de solapamiento espectral (\textit{aliasing}) en la banda base útil.

De este proceso de muestreo se deriva la relación fundamental entre la frecuencia analógica continua $f$ (en Hz) y la frecuencia digital normalizada $\omega$ (en radianes por muestra), expresada como $\omega = \Omega T_s = 2\pi \frac{f}{f_s}$ \cite{Proakis}. Bajo esta transformación lineal, el límite superior de la banda del sensor ($f_{\max} = 7500\text{ Hz}$) se mapea a $\omega_{\max} = 0.9375\pi\text{ rad/muestra}$, manteniéndose por debajo del límite de plegamiento digital situado en $\omega_N = \pi\text{ rad/muestra}$ ($8000\text{ Hz}$).
\subsection{Cuantificación, Error y Efecto Dither}
En el eje de amplitud, las muestras de tiempo discreto $x[n]$ se cuantifican de manera uniforme mediante codificación PCM de $B = 16 \text{ bits}$ con signo, lo que otorga $2^{16} = 65\,536$ niveles de representación. Asumiendo un rango dinámico normalizado a escala completa entre $V_{\min} = -1.0$ y $V_{\max} = +1.0$, el paso o escalón de cuantificación se calcula como:
\begin{equation}
    \Delta = \frac{V_{\max} - V_{\min}}{2^B - 1} = \frac{2}{65535} \approx 3.0518 \times 10^{-5} \text{ V}
\end{equation}

El error aleatorio de cuantificación asociado, definido por $e_q[n] = x_q[n] - x[n]$, se modela como una variable estocástica uniformemente distribuida en el intervalo $[-\Delta/2, \Delta/2]$, cuya potencia de ruido o varianza teórica es:
\begin{equation}
    \sigma_e = \frac{\Delta}{\sqrt{12}} \approx 8.81 \times 10^{-6} \text{ V}
\end{equation}

Esto proporciona una Relación Señal a Ruido de Cuantificación máxima (SNR$_Q$) de $6.02 B + 1.76 = 98.08 \text{ dB}$. Adicionalmente, el piso de estática inyectado por el hardware a $-80 \text{ dBFS}$ ($\sigma = 1 \times 10^{-4} V$) opera como un \textit{dither} natural, descorrelacionando el error $e_q[n]$ de la señal de entrada y linealizando el cuantificador. Este fenómeno es vital en PDS, pues evita que las señales débiles de auxilio (como gemidos o micro-vibraciones de víctimas) sufran distorsión armónica o queden atrapadas en el valor de cero absoluto del cuantificador.

\subsection{Ausencia de la Etapa de Conversión D/A}
A diferencia de las arquitecturas de sonar activo o de ecografía médica que requieren la síntesis y transmisión periódica de pulsos de excitación en la gama ultrasónica mediante conversores Digital-Analógicos (D/A) y filtros de reconstrucción, el sistema USAR V2.1 opera bajo un paradigma estrictamente pasivo (de solo escucha). 

Bajo esta configuración, el transductor piezoeléctrico HIKMICRO AD21P actúa de forma exclusiva como un sensor mecano-acústico de entrada, lo que elimina por completo la necesidad de un módulo de conversión D/A o de circuitos de emisión activa en el diagrama de bloques del hardware. Esta omisión de la etapa de emisión responde a tres imperativos de campo: 1) evita inducir perturbaciones vibratorias externas sobre una mampostería inestable con riesgo de colapso inminente; 2) optimiza la autonomía eléctrica del hardware de la brigada; y 3) preserva la fidelidad de escucha y el sigilo requeridos para aislar las micro-emisiones acústicas de las víctimas sin contaminar el canal elástico de la estructura.



% --- FIN DEL BLOQUE 3 (SECCION III) ---


% --- INICIO DEL BLOQUE 4 (SECCION IV) ---

%\section{Operaciones con Señales Discretas}

\subsection{Desplazamiento Temporal: Encuadre y \textit{Crossfade}}
El desplazamiento temporal $x[n - n_0]$ facilita el encuadre (\textit{framing}). Así, segmenta la secuencia de 4.0 s en micro-ventanas solapadas $x_m[m] = x[m + m \cdot H] \cdot w[m]$. Cada ventana posee $N_w = 400$ muestras (25 ms) y avanza $H = 160$ muestras (10 ms), preservando la continuidad energética.

El desplazamiento también alinea ráfagas acústicas en el centro del lienzo. Una rampa de desvanecimiento (\textit{crossfade}) de 50 ms, definida por $w_{\text{cross}}[n] = \frac{1}{2} \left[ 1 - \cos\left(\frac{\pi n}{N_{\text{fade}}}\right) \right]$, previene las discontinuidades que causarían artefactos espectrales.

\subsection{Inversión y Convolución: Filtrado de Fase Cero}
En cuanto al filtrado de la señal, la convolución discreta describe el paso del canal a través de la respuesta al impulso $h[n]$ de un filtro digital IIR Butterworth pasa-banda de 4.º orden (150 Hz -- 7500 Hz):
\begin{equation}
    y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n - k]
\end{equation}

Para resolver el problema del retardo de grupo no lineal inherente a los filtros IIR, expresado como $\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$ (el cual dispersaría temporalmente los picos de los impactos rítmicos), el algoritmo implementa la técnica \textit{Forward-Backward} (\texttt{filtfilt}). 

Este proceso combina la convolución directa $v[n] = x[n] * h[n]$ con la operación matemática de \textbf{inversión temporal} de la secuencia intermedia $g[n] = v[-n]$. Al filtrar la secuencia invertida $r[n] = g[n] * h[n]$ y re-invertir el resultado final $y[n] = r[-n]$, la respuesta espectral global del sistema se transforma en:
\begin{equation}
    Y(e^{j\omega}) = X(e^{j\omega}) H(e^{j\omega}) H^*(e^{j\omega}) = X(e^{j\omega}) |H(e^{j\omega})|^2
\end{equation}

Puesto que la magnitud al cuadrado $|H(e^{j\omega})|^2$ es una función real y positiva, la respuesta de fase resultante es nula ($\phi_{\text{total}}(\omega) = 0$). Esta cancelación por inversión elimina por completo el retardo de grupo, garantizando que la posición temporal exacta $n_0$ de los transitorios percusivos se conserve inalterada en las gráficas de salida para el análisis táctico.



% =========================================================================
% NUEVA SECCIÓN: SÍNTESIS ESTOCÁSTICA Y CARACTERIZACIÓN FÍSICA (FASE 1)
% =========================================================================
\section{Caracterización Espectro-Temporal y Síntesis Estocástica de Impactos}

\subsection{Señal Analítica, Transformada de Hilbert y Morfología de Decaimiento}
Para cuantificar la disipación de energía elástica en los escombros reales de la Zona Cero sin la interferencia del rizado de fase de la portadora acústica, el sistema construye la señal analítica compleja $z[n] = x[n] + j\hat{x}[n]$, donde $\hat{x}[n]$ representa la Transformada Discreta de Hilbert de la secuencia de entrada \cite{Oppenheim}. La envolvente de amplitud instantánea se extrae mediante el módulo cuadrático:
\begin{equation}
    A[n] = |z[n]| = \sqrt{x^2[n] + \hat{x}^2[n]}
\end{equation}

A partir del pico de ataque $A_{\max}$, el decaimiento estructural $t_{\text{thud}}$ se mide hasta que la envolvente cae por debajo del umbral de disipación. La frecuencia de resonancia $f_0$ se estima mediante el algoritmo YIN (150--7500 Hz). Simultáneamente, la Densidad Espectral de Potencia (PSD) se calcula usando el método de Welch ($N_{\text{FFT}} = 1024$, solapamiento del 50\%) \cite{Proakis}.

\subsection{Modelado Procedimental de Impactos y Piscinas Disjuntas}
Dado el severo desbalance natural de muestras reales de víctimas bajo escombros frente al ruido ambiental, se diseñó un motor de síntesis estocástica en tiempo discreto para generar patrones rítmicos de supervivencia (código Morse: \texttt{SOS}, \texttt{HELP}, \texttt{OK}, \texttt{USAR}). Para preservar la coherencia física del medio sólido y evitar anomalías psicoacústicas durante el entrenamiento neuronal, los parámetros de resonancia $f_0$ y amortiguamiento $\beta = \frac{\ln(100)}{t_{\text{thud}}}$ se fijan de manera invariante a lo largo de cada lienzo de $4.0\text{ s}$. Cada colisión individual se sintetiza como la superposición de una componente elástica amortiguada (\textit{Thud}) y un transitorio de ataque de banda ancha (\textit{Crack}):
\begin{equation}
    s_{\text{imp}}[n] = 0.7 \sin\left(2\pi \frac{f_0}{f_s} n\right) e^{-\beta n T_s} + 0.3 \, w_c[n] e^{-\gamma n}
\end{equation}

Los instantes de impacto (\textit{onsets}) incorporan una perturbación gaussiana (\textit{jitter} biomecánico con $\sigma = 0.3$) acotada por un límite fisiológico máximo de $12.5\text{ golpes/s}$ ($\Delta t_{\min} = 80\text{ ms}$). Posteriormente, la señal útil $s[n]$ se contamina aditivamente con grabaciones reales de ruido de la Zona Cero $r[n]$ segregadas en piscinas disjuntas (\textit{Disjoint Pools} para entrenamiento y validación, erradicando la fuga de datos o \textit{data leakage}). El factor de escala $\alpha_{\text{noise}}$ fuerza una Relación Señal a Ruido ($\text{SNR}_{\text{dB}} \sim \mathcal{N}(4.60, 20^2)\text{ dB}$) destructiva:
\begin{equation}
    \alpha_{\text{noise}} = \frac{\text{RMS}(s[n])}{\text{RMS}(r[n]) \cdot 10^{\text{SNR}_{\text{dB}} / 20}}
\end{equation}

En la Tabla~\ref{tab:verdad_base} se contrastan los estadísticos físicos extraídos de los impactos reales en la Zona Cero frente a los $29\,953$ tensores sintéticos generados.

\begin{table}[htbp]
\centering
\caption{Comparativa de Parámetros Físicos: Verdad Base Real (Zona Cero) vs. Síntesis Estocástica PDS}
\label{tab:verdad_base}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Datos Reales (Zona Cero)}} & \multicolumn{3}{c}{\textbf{Síntesis Estocástica (Morse)}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Descriptor Físico} & \textbf{Mín} & \textbf{Máx} & \textbf{Prom.} & \textbf{Mín} & \textbf{Máx} & \textbf{Prom.} \\
\midrule
Cadencia ($\text{golpes/s}$) & 0.15 & 12.50 & 4.43 & 0.30 & 12.50 & 3.26 \\
Resonancia $f_0$ ($\text{Hz}$) & 754.19 & 8000.0 & 1883.24 & 162.04 & 8000.0 & 1643.02 \\
Centroide Espectral ($\text{Hz}$) & 1659.00 & 6410.83 & 3167.21 & 644.77 & 5052.36 & 2598.65 \\
Dinámica de Pico ($\text{dBFS}$) & $-76.33$ & $0.00$ & $-26.25$ & $-71.22$ & $0.00$ & $-36.85$ \\
Energía RMS Lineal & $6.0\times 10^{-5}$ & 0.2104 & 0.0180 & $9.6\times 10^{-5}$ & 0.9668 & 0.0169 \\
Relación Señal/Ruido ($\text{dB}$) & $-50.09$ & 64.16 & 4.60 & $-42.90$ & 63.31 & 6.99 \\
Decaimiento $t_{\text{thud}}$ ($\text{ms}$) & 0.44 & 149.88 & 67.73 & 0.00 & 149.94 & 60.62 \\
\bottomrule
\end{tabular}%
}
\end{table}

% =========================================================================
% NUEVA SECCIÓN: EXTRACCIÓN VECTORIZADA Y MODELO CRNN (FASE 2)
% =========================================================================
\section{Extracción Vectorizada en GPU y Arquitectura Híbrida CRNN (\textit{Wide \& Deep})}

\subsection{Descriptores en Tiempo Discreto y Normalización Física en GPU}
Durante la ingesta hacia el modelo neuronal, las secuencias inferiores a $N = 64\,000$ muestras se estandarizan mediante relleno circular (\textit{wrap padding}), $x_{\text{pad}}[n] = x[n \bmod L]$, preservando la continuidad de fase y el piso de ruido estacionario sin introducir ceros artificiales. El motor de extracción vectorizado ejecuta las transformaciones de PDS en GPU sobre lotes de dimensión $[B, 64\,000]$:

\begin{enumerate}
    \item \textbf{Rama \textit{Wide} (Heurísticas Físicas Globales):} Para cada ventana completa se computa un vector tridimensional $\mathbf{x}_{\text{wide}} \in \mathbb{R}^3$ integrado por: a) el valor eficaz global:
    
    \begin{equation}
        \text{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x^2[n]} 
    \end{equation}

    b) la Tasa de Cruces por Cero definida mediante la función signo discreta:

    \begin{equation}
        \text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\operatorname{sgn}(x[n]) - \operatorname{sgn}(x[n-1])|
    \end{equation}

    lo que permite discriminar la fricción estructural de alta frecuencia frente a formantes vocales, y c) el Centroide Espectral promedio obtenido vía STFT con ventana de Hann ($N_w = 400$, $H = 160$), normalizado respecto a la frecuencia de plegamiento de Nyquist ($f_N = f_s / 2 = 8000\text{ Hz}$):
    
    \begin{equation}
        \bar{C}_{\text{norm}} = \frac{1}{f_N \cdot M} \sum_{m=0}^{M-1} \frac{\sum_{k=0}^{K-1} f[k] \, |X[m, k]|}{\sum_{k=0}^{K-1} |X[m, k]| + \epsilon}
    \end{equation}

    \item \textbf{Rama \textit{Deep} (Representación Tiempo-Frecuencia):} Se calcula el espectrograma de potencia sobre $128$ bancos de filtros triangulares en escala Mel ($150\text{--}7500\text{ Hz}$) y se transforma a escala logarítmica $S_{\text{dB}}[m, t] = 10\log_{10}(S_{\text{Mel}}[m, t])$. En lugar de aplicar una normalización estadística por muestra (\textit{Min-Max}), que amplificaría el siseo térmico en ventanas de silencio induciendo falsos positivos, se implementa un escalado físico anclado al piso de \textit{dither} natural de $-80\text{ dBFS}$:
    \begin{equation}
        \tilde{S}[m, t] = \operatorname{clip}\left( \frac{S_{\text{dB}}[m, t] + 40.0}{40.0}, \, -1.0, \, +1.0 \right)
    \end{equation}
\end{enumerate}

\subsection{Topología Convolucional Recurrente y Balanceo}
El clasificador emplea una arquitectura \textit{Wide \& Deep} (\texttt{USARWideDeepModel}), integrando redes convolucionales, recurrentes (GRU) y perceptrones multicapa (MLP):

\begin{itemize}
    \item \textbf{Codificador Espectro-Temporal:} El tensor $\tilde{S}$ atraviesa regularización por ruido gaussiano ($\sigma = 0.05$). Posteriormente, dos bloques \texttt{Conv2d} ($3\times 3$) comprimen el eje espectral a 32 bandas y el temporal a 101 pasos. La secuencia aplanada alimenta una red GRU de 2 capas ($h_t \in \mathbb{R}^{64}$), cuyo último estado $h_{T_{\text{out}}}$ condensa la firma rítmica.
    \item \textbf{Proyección Física y Fusión:} El vector $\mathbf{x}_{\text{wide}} \in \mathbb{R}^3$ se normaliza y proyecta a $\mathbb{R}^{16}$. Ambas ramas se concatenan formando $\mathbf{z}_{\text{fus}} \in \mathbb{R}^{80}$, alimentando un clasificador final hacia 4 clases.
\end{itemize}

Para mitigar la asimetría residual en la distribución de clases sobre el conjunto consolidado de $65\,576$ tensores ($52\,417$ de entrenamiento y $13\,159$ de validación, particionados sin fuga mediante \texttt{StratifiedGroupKFold}), la función de pérdida de Entropía Cruzada se pondera asignando a cada clase $c$ un peso inversamente proporcional a su frecuencia empírica $N_c$:
\begin{equation}
    w_c = \frac{N_{\text{total}}}{C \cdot N_c} \implies \mathbf{w} = [2.686, \, 12.055, \, 0.366, \, 1.233]^T
\end{equation}

\subsection{Desempeño Experimental del Pipeline Integrado}
El entrenamiento se ejecutó durante $20$ épocas sobre una unidad de procesamiento gráfico NVIDIA GeForce RTX 4060 utilizando el optimizador Adam ($\text{lr} = 10^{-3}$, decaimiento de pesos de $10^{-4}$ y tamaño de lote $B = 64$). En la Tabla~\ref{tab:resultados_crnn} se resume la evolución de las métricas de validación, donde el sistema alcanzó su convergencia óptima en la época 19 con un \textbf{F1-Score Macro de $0.8631$} y un área bajo la curva característica operativa del receptor \textbf{ROC-AUC (One-vs-Rest) de $0.9927$}.

\begin{table}[htbp]
\centering
\caption{Evolución del Entrenamiento y Validación de la Arquitectura Híbrida CRNN (\textit{Wide \& Deep})}
\label{tab:resultados_crnn}
\begin{tabular}{ccccc}
\toprule
\textbf{Época} & \textbf{Train Loss} & \textbf{Val Loss} & \textbf{Val F1-Macro} & \textbf{ROC-AUC (OvR)} \\
\midrule
01 / 20 & 0.6294 & 0.4418 & 0.7452 & --- \\
05 / 20 & 0.3755 & 0.3396 & 0.8241 & --- \\
10 / 20 & 0.3313 & 0.3454 & 0.8044 & --- \\
14 / 20 & 0.3133 & 0.3097 & 0.8512 & --- \\
\textbf{19 / 20} & \textbf{0.2694} & \textbf{0.3184} & \textbf{0.8631} & \textbf{0.9927} \\
20 / 20 & 0.2664 & 0.3089 & 0.8459 & --- \\
\bottomrule
\end{tabular}
\end{table}



% --- FIN DEL BLOQUE 4 (SECCION IV) ---




% --- INICIO DEL BLOQUE 5 (SECCION V) ---

\section{Análisis Gráfico}


En la Figura 1 se ilustra el proceso de digitalización de un evento acústico transitorio (un impacto percusivo sobre mampostería) mediante la superposición del trazo analógico continuo $x_a(t)$ (representado en tono gris) y su correspondiente secuencia en tiempo discreto $x[n]$ mediante un gráfico de tallos (\textit{stem plot} en color azul). La señal se modela matemáticamente como un impulso sinusoidal amortiguado y atenuado por la masa de concreto, expresado por:

\begin{equation}
    x[n] = A \cdot e^{-\alpha n T_s} \cdot \sin(2\pi f_0 n T_s) \cdot u[n] + w[n]
\end{equation}
Donde $A$ representa la amplitud pico del impacto, $\alpha$ es el coeficiente de absorción elástica del medio estructural, $f_0 = 450\text{ Hz}$ es la frecuencia de resonancia dominante en el escombro y $w[n]$ es el ruido de estática de hardware. La discretización temporal respeta la tasa uniforme de muestreo $f_s = 16\text{ kHz}$, reflejando un periodo de muestreo constante de $T_s = 62.5\ \mu\text{s}$ entre muestras consecutivas.

% --- ESPACIO PARA LA FIGURA 1 ---
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.45\textwidth]{../LaTex_figuras/fig1_impacto_real0.png}
    \caption{Señal analógica de impacto amortiguado $x_a(t)$ vs. secuencia muestreada discreta $x[n]$.}
    \label{fig:muestreo}
\end{figure}

La Figura 2 presenta la evaluación del proceso de cuantificación uniforme PCM de 16 bits. En el panel superior, la superposición entre la secuencia de amplitud continua $x[n]$ y la secuencia cuantificada $x_q[n]$ evidencia una coincidencia visual casi perfecta debido a la alta resolución del convertidor analógico-digital ($2^{16} = 65\,536$ niveles). En el panel inferior, la gráfica del vector de error de cuantificación $e_q[n] = x_q[n] - x[n]$ demuestra que la desviación residual permanece acotada dentro del intervalo $[-\Delta/2, +\Delta/2]$. Para una escala normalizada de amplitud en el rango $[-1.0, 1.0]$ ($V_{\text{pp}} = 2.0$), el escalón de cuantificación equivale a $\Delta = \frac{2}{65\,535} \approx 3.05 \times 10^{-5} V$, lo que limita el margen de error máximo a $\pm 1.52 \times 10^{-5} V$. Esto confirma que la potencia del error de cuantificación se mantiene en un piso imperceptible de $-98.09\text{ dBFS}$, preservando la integridad de las señales de muy baja amplitud.

% --- ESPACIO PARA LA FIGURA 2 ---
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.45\textwidth]{../LaTex_figuras/fig2_cuantificacion.png}
    \caption{Arriba: Superposición de la señal analógica y cuantificada. Abajo: Vector de error de cuantificación $e_q[n]$.}
    \label{fig:cuantificacion}
\end{figure}

Finalmente, en la Figura 3 se compara la señal discreta cruda de terreno $x[n]$ frente a la secuencia procesada $y[n]$ tras la etapa de acondicionado digital. El registro crudo presenta severas distorsiones provocadas por deriva de línea base de baja frecuencia ($<150\text{ Hz}$) e interferencias de alta frecuencia derivadas del entorno. Tras aplicar la cascada de filtros IIR Butterworth de 4.º orden (filtro pasa-banda de $150\text{ Hz} - 7500\text{ Hz}$ para acoplamiento al transductor AD21P, procesados mediante filtrado bidireccional de fase cero (\texttt{filtfilt}), se observa cómo la señal $y[n]$ suprime la deriva de continua y atenúa el ruido de alta frecuencia. Esta operación rescata la firma temporal limpia de la señal de vida (los picos transitorios atenuados o los formantes de voz), eliminando los desfases de grupo para alimentar con máxima fidelidad las ramas \textit{Wide} y \textit{Deep} del clasificador.

% --- ESPACIO PARA LA FIGURA 3 ---
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.45\textwidth]{../LaTex_figuras/fig3_filtrado_fase_cero_out.png}
    \caption{Comparativa espectro-temporal de la señal cruda con ruido de baja/alta frecuencia y la señal procesada $y[n]$.}
    \label{fig:filtrado}
\end{figure}


\section{Conclusiones}
Este artículo validó la aplicabilidad del Procesamiento Digital de Señales para la detección pasiva de sobrevivientes. El cumplimiento del criterio de Nyquist a $16\text{ kHz}$ y la cuantificación PCM de $16\text{ bits}$ con \textit{dither} garantizaron la captura de micro-vibraciones sin \textit{aliasing} ni distorsión \cite{Oppenheim, Proakis}. La convolución mediante filtrado Butterworth de fase cero suprimió el ruido estructural sin comprometer la alineación temporal de los impactos. 

La Transformada de Hilbert y la síntesis estocástica permitieron entrenar una arquitectura CRNN sobre $65\,576$ tensores, alcanzando un F1-Score de $0.8631$. La integración de \LaTeX{} y librerías vectorizadas de Python comprobó cómo las ecuaciones teóricas se traducen en herramientas computacionales reales para ingeniería de rescate.

\section*{Agradecimientos}
Se agradece el apoyo académico y técnico brindado por el profesor Gerlis Caropresse de la Universidad Central de Venezuela, al profesor Daniel Paredero del Instituto IMMUNE Technology, y a las brigadas de rescate mexicanas Los Topos por la caracterización operativa del entorno.

% --- BIBLIOGRAFÍA ---
\begin{thebibliography}{9}
    \bibitem{Proakis}
    J. G. Proakis y D. G. Manolakis, \textit{Digital Signal Processing: Principles, Algorithms, and Applications}, 4ta ed. Upper Saddle River, NJ, EE. UU.: Pearson, 2006.
    
    \bibitem{Oppenheim}
    A. V. Oppenheim y R. W. Schafer, \textit{Discrete-Time Signal Processing}, 3ra ed. Upper Saddle River, NJ, EE. UU.: Prentice Hall, 2010.
\end{thebibliography}


\end{document}