# **FASE #2:** **Motor de Procesamiento Digital de Señales (DSP) y DataLoaders**


```python
# 1. Librerías Estándar de Python
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# 2. Análisis de Datos, Machine Learning y Audio
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

# 3. Ecosistema Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

```

### **BLOQUE: Parseo y Unificación**

Actuará como el mapa de PyTorch consultará iterativamente para cargar, aplicar transformaciones en vuelo y enviar los tensores a la GPU de manera eficiente.



```python
# Mapeo taxonómico constante del sistema
LABEL_MAP = {
    "Noise_General": 0,
    "Target_Animal": 1,
    "Target_Rhythmic": 2,
    "Target_Voice": 3
}

def parsear_y_unificar_datasets(lista_directorios_raiz: List[Union[str, Path]]) -> pd.DataFrame:
    """Escanea recursivamente una lista de directorios raíz para indexar tensores de audio.

    Busca archivos con extensión .wav y extrae sus metadatos basándose en la jerarquía
    de carpetas. Incorpora un mecanismo de elevación de ruta (path lifting) para
    asimilar directorios de Pools Disjuntos ('train', 'val', 'test') sin alterar la 
    taxonomía aprobada en LABEL_MAP.
    """
    datos_indexados = []

    for directorio in lista_directorios_raiz:
        ruta_raiz = Path(directorio)
        
        # Validacion de existencia del directorio
        if not ruta_raiz.exists() or not ruta_raiz.is_dir():
            print(f"[ADVERTENCIA] El directorio raíz especificado no existe o no es válido: {ruta_raiz}")
            continue
            
        # Búsqueda recursiva de archivos .wav
        for archivo_wav in ruta_raiz.rglob('*.wav'):
            if not archivo_wav.is_file():
                continue
                
            try:
                partes_relativas = archivo_wav.relative_to(ruta_raiz).parts
                # Estructura Disjoint (4 partes): dataset / categoria / split / archivo.wav
                if len(partes_relativas) >= 3:
                    dataset_origen = partes_relativas[0]
                    categoria = partes_relativas[1]
                else:
                    print(f"[ADVERTENCIA] Profundidad jerárquica inválida para {archivo_wav.name}")
                    continue
                
                # Validación estricta contra la taxonomía aprobada
                if categoria not in LABEL_MAP:
                    print(f"[ADVERTENCIA] Categoría anómala '{categoria}' detectada en {archivo_wav.name}. Archivo ignorado.")
                    continue
                    
                # Resolución de ruta absoluta para evitar punteros rotos en el DataLoader
                ruta_absoluta = str(archivo_wav.resolve())
                label_idx = LABEL_MAP[categoria]
                
                datos_indexados.append({
                    "ruta_absoluta": ruta_absoluta,
                    "dataset_origen": dataset_origen,
                    "categoria": categoria,
                    "label_idx": label_idx
                })
                
            except Exception as e:
                # Captura de errores de lectura de pathing o permisos
                print(f"[ADVERTENCIA] Error procesando el archivo {archivo_wav}: {str(e)}")
                continue

    # Construcción y retorno del DataFrame analítico
    df_unificado = pd.DataFrame(datos_indexados, columns=[
        "ruta_absoluta", 
        "dataset_origen", 
        "categoria", 
        "label_idx"
    ])
    
    return df_unificado
```

### **Módulo de División Estratificada**

Este componente realiza el enrutamiento y la partición de los metadatos de entrenamiento, sin realizar operaciones de lectura de archivos de audio en disco para preservar la eficiencia computacional de la estación de trabajo.


```python
def dividir_datos_estratificados(
    df_unificado: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza una división híbrida del dataset previniendo Data Leakage en todos los frentes.

    Fase A (Enrutamiento Estricto): Detecta audios provenientes de 'Disjoint Pools' 
    sintéticos (carpetas /train o /val) y respeta su asignación física incondicionalmente.
    
    Fase B (División Automática): Para los datos reales restantes, extrae la raíz del 
    archivo original y aplica StratifiedGroupKFold. Esto garantiza que todos los 
    fragmentos derivados de un mismo evento acústico permanezcan en el mismo conjunto, 
    conservando la proporción matemática de las clases.

    Parameters
    ----------
    df_unificado : pd.DataFrame
        DataFrame unificado proveniente del bloque de Parseo.
    test_size : float, optional
        Proporción para validación (rango 0.0 a 1.0). Por defecto es 0.2 (20%).
    random_state : int, optional
        Semilla aleatoria para reproducibilidad. Por defecto es 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames (df_entrenamiento, df_validacion) ensamblados, barajados y listos.
    """
    # 1. Validaciones estrictas de tipo e integridad de datos ( Safety)
    if not isinstance(df_unificado, pd.DataFrame):
        raise TypeError("El parámetro 'df_unificado' debe ser un pd.DataFrame de pandas.")

    if df_unificado.empty:
        raise ValueError("El DataFrame de entrada está vacío. No hay datos para dividir.")

    columnas_requeridas = ["ruta_absoluta", "dataset_origen", "categoria", "label_idx"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_unificado.columns]
    
    if columnas_faltantes:
        raise ValueError(
            f"El DataFrame no es compatible con la Fase 2. Faltan las columnas: {columnas_faltantes}"
        )

    # 2. ESCÁNER DE ENRUTAMIENTO 
    df_temp = df_unificado.copy()
    df_temp['split_asignado'] = 'auto'
    
    # Asigna 'train' si la ruta contiene \train\ o /train/
    mask_train = df_temp['ruta_absoluta'].astype(str).str.contains(r'[\\/]train[\\/]', case=False, regex=True)
    df_temp.loc[mask_train, 'split_asignado'] = 'train'
    
    # Asigna 'val' si la ruta contiene \val\, /val/, \test\ o /test/
    mask_val = df_temp['ruta_absoluta'].astype(str).str.contains(r'[\\/](?:val|test)[\\/]', case=False, regex=True)
    df_temp.loc[mask_val, 'split_asignado'] = 'val'

    # 3. SEGREGACIÓN DE DATOS
    df_pre_train = df_temp[df_temp['split_asignado'] == 'train'].drop(columns=['split_asignado'])
    df_pre_val = df_temp[df_temp['split_asignado'] == 'val'].drop(columns=['split_asignado'])
    df_auto = df_temp[df_temp['split_asignado'] == 'auto'].drop(columns=['split_asignado'])

    print(f"[] Enrutamiento: {len(df_pre_train)} a Train pre-asignado | {len(df_pre_val)} a Val pre-asignado | {len(df_auto)} a StratifiedGroupKFold")

    df_train_list = [df_pre_train]
    df_val_list = [df_pre_val]

    # 4. PARTICIÓN ESTRATIFICADA Y AGRUPADA
    if not df_auto.empty:
        df_auto = df_auto.copy()
        
        
        def extraer_raiz_grupo(ruta):
            nombre_base = Path(ruta).name
            # Intenta remover sufijos de segmentación
            grupo = re.sub(r'(_part\d+|_chunk\d+|_\d+)\.wav$', '', nombre_base, flags=re.IGNORECASE)
            
            # Fallback de seguridad: Si la regex no detectó sufijo, se usa el nombre original sin extensión (stem)
            if grupo == nombre_base or not grupo:
                return Path(ruta).stem
            return grupo

        df_auto['grupo_base'] = df_auto['ruta_absoluta'].apply(extraer_raiz_grupo)

        n_splits = int(1.0 / test_size) if test_size > 0 else 5
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        train_idx, val_idx = next(sgkf.split(
            X=df_auto, 
            y=df_auto['label_idx'], 
            groups=df_auto['grupo_base']
        ))

        df_train_list.append(df_auto.iloc[train_idx].drop(columns=['grupo_base']))
        df_val_list.append(df_auto.iloc[val_idx].drop(columns=['grupo_base']))

    # 5. ENSAMBLAJE FINAL Y BARAJADO ESTOCÁSTICO
    df_train_final = pd.concat(df_train_list, ignore_index=True)
    df_val_final = pd.concat(df_val_list, ignore_index=True)

    # Reajuste de Índices y Barajado
    df_train_limpio = df_train_final.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_val_limpio = df_val_final.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df_train_limpio, df_val_limpio
```

### **Módulo de Balanceo Estructural y Dataset de PyTorch**


```python
def calcular_pesos_clase(df_train: pd.DataFrame) -> torch.Tensor:
    """Calcula los pesos balanceados para cada clase basados en el split de entrenamiento.

    Esta función evalúa la frecuencia absoluta de cada etiqueta en el conjunto de
    entrenamiento y aplica la formulación estándar de balanceo de clases:
    w = N_total / (C * N_clase), compensando la pérdida por desbalance de datos.
    Para garantizar la seguridad de , no recibe información del subconjunto de
    validación, evitando sesgos cognitivos en el paisaje de optimización.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame de entrenamiento que contiene la columna obligatoria 'label_idx'.

    Returns
    -------
    torch.Tensor
        Tensor unidimensional de PyTorch de tipo float32 con los pesos compensatorios.

    Raises
    ------
    TypeError
        Si 'df_train' no es una instancia de pandas.DataFrame.
    ValueError
        Si el DataFrame está vacío o carece de la columna 'label_idx'.
    """
    if not isinstance(df_train, pd.DataFrame):
        raise TypeError("El parámetro 'df_train' debe ser un pd.DataFrame de pandas.")

    if df_train.empty:
        raise ValueError("El DataFrame de entrenamiento se encuentra vacío.")

    if 'label_idx' not in df_train.columns:
        raise ValueError("La columna requerida 'label_idx' no está presente en el DataFrame.")

    # Conteo de la distribución absoluta de clases sobre el split de Train
    conteo_clases = df_train['label_idx'].value_counts()
    num_clases = len(conteo_clases)
    total_muestras = len(df_train)

    # Inicialización del array alineando los pesos con el índice de cada clase
    pesos_array = np.zeros(num_clases, dtype=np.float32)
    for clase_idx, count in conteo_clases.items():
        # Formulación matemática clásica para balanceo simétrico de la entropía cruzada
        pesos_array[clase_idx] = total_muestras / (num_clases * count)

    return torch.tensor(pesos_array, dtype=torch.float32)

# 1. DATASET DE I/O LIGERO 

class USARDatasetLight(torch.utils.data.Dataset):
    """Cargador de datos de alto rendimiento (CPU Bound para I/O).
    
    Delega el procesamiento pesado a la GPU. Resuelve la fuga de datos de 
    longitud mediante 'wrap padding', manteniendo la coherencia de fase y el 
    piso de ruido ambiental continuo sin inyectar ceros absolutos.
    """
    def __init__(self, df: pd.DataFrame, sr: int = 16000, is_train: bool = True):
        self.df = df
        self.sr = sr
        self.is_train = is_train
        self.target_len = int(4.0 * self.sr)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        ruta = row['ruta_absoluta']
        label = row['label_idx']

        # 1. Lectura I/O 
        try:
            waveform, sample_rate = torchaudio.load(ruta)
            # Forzar a mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Remuestreo dinámico si es necesario
            if sample_rate != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)
                waveform = resampler(waveform)
            waveform = waveform.squeeze(0).numpy()
        except Exception as e:
            print(f"[ERROR I/O] Fallo físico en disco para {ruta}: {e}")
            return None, None

        # 2. Homogeneización Temporal Continua 
        len_audio = len(waveform)
        if len_audio < self.target_len:
            waveform = np.pad(waveform, (0, self.target_len - len_audio), mode='wrap')
        elif len_audio > self.target_len:
            waveform = waveform[:self.target_len]

        # 3. Volume Augmentation Estocástico Anti-Clipping
        if self.is_train:
            factor = np.random.uniform(0.8, 1.2)
            waveform = waveform * factor
            max_abs = np.max(np.abs(waveform))
            if max_abs > 1.0:
                waveform = waveform / max_abs

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



```

### **Módulo: Red Neuronal Híbrida Wide & Deep**

Esta arquitectura híbrida unifica la capacidad de generalización secuencial de las redes convolucionales recurrentes (rama **Deep**) con el control heurístico y físico de los descriptores globales absolutos (rama **Wide**). 


```python

class GaussianNoise(nn.Module):
    """Capa de regularización por inyección de ruido blanco gaussiano aditivo.

    Esta capa añade ruido gaussiano de media cero y desviación estándar parametrizable 
    a los tensores de entrada únicamente durante la etapa de entrenamiento (self.training).
    Funciona como un regularizador activo de primer orden para perturbar las amplitudes 
    espectrales locales, forzando al modelo a aprender patrones robustos y previniendo 
    el sobreajuste (overfitting).
    """

    def __init__(self, stddev: float = 0.05):
        """Inicializa la capa de ruido gaussiano.

        Parameters
        ----------
        stddev : float, optional
            Desviación estándar del ruido gaussiano aditivo. Por defecto es 0.05.
        """
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la perturbación aditiva en el flujo hacia adelante.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada de dimensiones arbitrarias.

        Returns
        -------
        torch.Tensor
            Tensor perturbado si el modelo está en modo entrenamiento, o el tensor
            original intacto si está en modo de evaluación.
        """
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x



class USARFeatureExtractorGPU(nn.Module):
    """Motor DSP vectorizado para GPU.
    
    Procesa lotes completos de formas de onda [B, 64000] en paralelo.
    Fija la física del HIKMICRO AD21P (150-7500Hz) y extrae las ramas WIDE y DEEP.
    """
    def __init__(self, sr: int = 16000):
        super().__init__()
        self.sr = sr
        self.n_fft = 400
        self.hop_length = 160
        
        # Transformadas acotadas al hardware
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128,
            f_min=150.0,
            f_max=7500.0,
            power=2.0 
        )
        self.db_transform = T.AmplitudeToDB(stype='power', top_db=None)

    @torch.no_grad() 
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- EXTRACCIÓN DE RAMA WIDE ---
        
        # 1. RMS Global
        rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + 1e-10)
        
        # 2. Tasa de Cruces por Cero (ZCR)
        sign_changes = torch.abs(torch.diff(torch.sign(waveform), dim=-1)) > 0
        zcr = torch.mean(sign_changes.float(), dim=-1, keepdim=True)
        
        # 3. Centroide Espectral con ventana de Hann
        window = torch.hann_window(self.n_fft, device=waveform.device)
        spec = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window,
            return_complex=True
        )
        mag = torch.abs(spec) # [B, Freqs, Frames]
        freqs = torch.linspace(0, self.sr / 2, mag.size(1), device=waveform.device).view(1, -1, 1)
        
        centroid_frames = torch.sum(freqs * mag, dim=1) / (torch.sum(mag, dim=1) + 1e-10)
        centroid = torch.mean(centroid_frames, dim=-1, keepdim=True)
        
        # Normalización de Nyquist (Propuesta A) para mantener estabilidad en W&D
        centroid_norm = centroid / (self.sr / 2.0)
        
        # Ensamble Wide [B, 3]
        tensor_wide = torch.cat([rms, zcr, centroid_norm], dim=-1)

        # --- EXTRACCIÓN DE RAMA DEEP ---
        mel_spec = self.mel_transform(waveform)
        mel_db = self.db_transform(mel_spec)
        
        # Normalización Física Absoluta: Centrado en -40dB y escalado matemático [-1, 1]
        
        mel_norm = (mel_db + 40.0) / 40.0
        mel_norm = torch.clamp(mel_norm, -1.0, 1.0)
        
        # Formateo a [Batch, Channels, Mels, Time]
        tensor_deep = mel_norm.unsqueeze(1)
        
        return tensor_deep, tensor_wide

        
class USARWideDeepModel(nn.Module):
    """Modelo híbrido Wide & Deep para la detección acústica de vida bajo escombros.

    Esta arquitectura implementa un procesamiento bifurcado:
    - Rama DEEP (Secuencial): Procesa espectrogramas Mel [B, 1, n_mels, T] mediante
      bloques convolucionales (CNN2D) para reducción dimensional y una red recurrente
      de compresión de secuencia (GRU) para capturar el patrón temporal (ej. Código Morse).
    - Rama WIDE (Global): Procesa descriptores estadísticos globales absolutos [B, 3] 
      (RMS, ZCR y Centroide Espectral) mediante un Perceptrón Multicapa (MLP) lineal
      para conservar las heurísticas físicas del sensor.

    Ambas representaciones se fusionan y clasifican mediante un cabezal denso multitarea.
    La capa de salida entrega logits puros, delegando la estabilidad numérica de la
    función de activación no lineal a nn.CrossEntropyLoss o nn.BCEWithLogitsLoss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        n_mels: int = 128,
        deep_hidden_dim: int = 64,
        wide_dim: int = 3,
        wide_proj_dim: int = 16,
        dropout_prob: float = 0.3,
        noise_std: float = 0.05
    ):
        """Inicializa los módulos y capas del modelo híbrido.

        Parameters
        ----------
        num_classes : int, optional
            Cantidad de clases taxonómicas de la salida (ej. 4 clases). Por defecto es 4.
        in_channels : int, optional
            Canales de entrada de los espectrogramas Mel. Por defecto es 1.
        n_mels : int, optional
            Cantidad de bandas Mel empleadas en la extracción de la Fase 2. Por defecto es 128.
        deep_hidden_dim : int, optional
            Dimensión del espacio oculto de la unidad GRU secuencial. Por defecto es 64.
        wide_dim : int, optional
            Dimensión de entrada de los descriptores globales. Por defecto es 3.
        wide_proj_dim : int, optional
            Dimensión de proyección de la rama lineal Wide. Por defecto es 16.
        dropout_prob : float, optional
            Probabilidad de regularización por Dropout en la cabeza de fusión. Por defecto es 0.3.
        noise_std : float, optional
            Intensidad de la inyección de estática Gaussiana en el espectrograma. Por defecto es 0.05.
        """
        super().__init__()

        # Capa de regularización para robustecer las huellas espectrales
        self.deep_noise = GaussianNoise(stddev=noise_std)

        # RAMA DEEP (Procesamiento Convolucional + Recurrente Secuencial)
        # Bloques de reducción dimensional espectral
        # Input: [B, 1, 128, T]
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # -> [B, 16, 64, T/2]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # -> [B, 32, 32, T/4]
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Cálculo de la dimensión de características aplanadas por paso temporal
        # Con 32 canales y una dimensión de frecuencia reducida a 32 (128 / 2 / 2)
        rnn_input_size = 32 * (n_mels // 4)

        # Red Recurrente para modelado rítmico con paso de secuencia optimizado para GPU
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=deep_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob if dropout_prob > 0 else 0.0
        )


        # RAMA WIDE 

        self.wide_mlp = nn.Sequential(
            nn.BatchNorm1d(wide_dim),
            nn.Linear(wide_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, wide_proj_dim),
            nn.ReLU()
        )

        # FUSIÓN Y CABEZAL MULTITAREA (Clasificador Final)
        # La dimensión combinada es la suma de las proyecciones de ambas ramas
        combined_dim = deep_hidden_dim + wide_proj_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, num_classes)  # Salida de logits puros para CrossEntropyLoss
        )

    def forward(self, x_deep: torch.Tensor, x_wide: torch.Tensor) -> torch.Tensor:
        """Realiza la propagación hacia adelante unificando ambas ramas.

        Parameters
        ----------
        x_deep : torch.Tensor
            Tensor 4D de espectrogramas Mel normalizados de forma [Batch, 1, 128, Time_Steps].
        x_wide : torch.Tensor
            Tensor 2D de características estadísticas absolutas de forma [Batch, 3].

        Returns
        -------
        torch.Tensor
            Tensor 2D de logits puros de clasificación de forma [Batch, num_classes].
        """
        # 1. Ejecución de la Rama DEEP
        x_deep_noisy = self.deep_noise(x_deep)
        
        # Reducción dimensional convolucional
        conv_out = self.conv_blocks(x_deep_noisy)  # Shape: [B, C_out, F_out, T_out]
        
        # Preparación de secuencia temporal: Reorganización a [Batch, Time_Steps, Features]
        # Esto permite procesar dinámicamente cualquier longitud temporal (T)
        B, C, F, T = conv_out.shape
        # Permutamos para alinear el eje temporal T con la dimensión secuencial del RNN
        conv_permuted = conv_out.permute(0, 3, 1, 2).contiguous()  # -> [B, T_out, C_out, F_out]
        rnn_input = conv_permuted.view(B, T, C * F)                # -> [B, T_out, C_out * F_out]

        # Procesamiento secuencial recurrente
        rnn_out, _ = self.rnn(rnn_input)  # Shape: [B, T_out, deep_hidden_dim]
        
        # Extracción del último estado temporal de la secuencia
        deep_features = rnn_out[:, -1, :]  # Shape: [B, deep_hidden_dim]

        # 2. Ejecución de la Rama WIDE
        # Proyección del vector estadístico absoluto
        wide_features = self.wide_mlp(x_wide)  # Shape: [B, wide_proj_dim]

        # 3. Fusión de Características por Concatenación Directa
        combined_features = torch.cat([deep_features, wide_features], dim=1)  # Shape: [B, combined_dim]

        # 4. Clasificación y Obtención de Logits Puros
        logits = self.classifier(combined_features)  # Shape: [B, num_classes]

        return logits

```

### **Módulo Bucle de Entrenamiento**


```python
def usardataset_collate_fn(batch):
    """Filtra dinámicamente audios corruptos (None) para evitar Label Noise en la GPU."""
    # Filtra las tuplas donde la onda cruda falló en la lectura I/O
    batch_limpio = [item for item in batch if item[0] is not None]
    
    # Si por rareza estadística todo el lote está corrupto, retorna nulos
    if not batch_limpio:
        return None, None
        
    # Ensambla el tensor final solo con los datos acústicamente íntegros
    return default_collate(batch_limpio)

def train_model(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    class_weights: torch.Tensor,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """Orquesta el bucle de entrenamiento y validación de la red Wide & Deep.

    Optimiza el flujo de ingesta de datos explotando la arquitectura de hardware local
    mediante hilos concurrentes y memoria paginada para la GPU. Evalúa el desempeño
    en cada época calculando de forma estricta la pérdida ponderada, el F1-Score
    y el Área Bajo la Curva ROC (AUC-ROC) multiclase.

    Parameters
    ----------
    model : nn.Module
        Instancia de la red neuronal híbrida USARWideDeepModel.
    train_dataset : torch.utils.data.Dataset
        Conjunto de datos de entrenamiento (USARDataset con is_train=True).
    val_dataset : torch.utils.data.Dataset
        Conjunto de datos de validación (USARDataset con is_train=False).
    class_weights : torch.Tensor
        Tensor unidimensional float32 con los pesos compensatorios de clase.
    batch_size : int, optional
        Tamaño del lote de entrenamiento. Por defecto es 64.
    learning_rate : float, optional
        Tasa de aprendizaje inicial para el optimizador Adam. Por defecto es 1e-3.
    num_epochs : int, optional
        Número total de épocas a entrenar. Por defecto es 20.
    device : str, optional
        Dispositivo físico de cómputo (cuda o cpu).

    Returns
    -------
    Dict[str, List[float]]
        Historial detallado con las métricas registradas en cada época:
        'train_loss', 'val_loss', 'val_f1_macro', 'val_auc'.
    """
    
    num_workers = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=usardataset_collate_fn 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=usardataset_collate_fn 
    )

    # Enviar el modelo y los pesos de pérdida al acelerador físico (GPU)
    model = model.to(device)
    class_weights = class_weights.to(device)

    # Instanciar el motor DSP y enviarlo a la GPU (Modo evaluación permanente)
    feature_extractor = USARFeatureExtractorGPU(sr=16000).to(device)
    feature_extractor.eval()

    # 2. Inicialización de Optimizador y Función de Pérdida Ponderada
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Historial de métricas para trazabilidad 
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_macro": [],
        "val_auc": []
    }
    
    
    best_f1_macro = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    
    # Bucle forzado de épocas
    for epoch in range(num_epochs):
        # FASE DE ENTRENAMIENTO
        model.train()
        train_loss_accum = 0.0
        num_train_batches = len(train_loader)

        train_pbar = tqdm(train_loader, desc=f"Época {epoch+1:02d}/{num_epochs:02d} [Entrenamiento]", leave=False)

        for x_wave, y in train_pbar:
            if x_wave is None:
                continue
                
            # 1. Enviar la onda cruda a la memoria de video
            x_wave = x_wave.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 2. Ejecutar Extracción DSP asíncrona masiva en GPU
            x_deep, x_wide = feature_extractor(x_wave)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x_deep, x_wide)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = train_loss_accum / num_train_batches

        # FASE DE VALIDACIÓN
        
        model.eval()
        val_loss_accum = 0.0
        num_val_batches = len(val_loader)

        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []

        val_pbar = tqdm(val_loader, desc=f"Época {epoch+1:02d}/{num_epochs:02d} [Validación]   ", leave=False)

        with torch.no_grad():
            for x_wave, y in val_pbar:
                if x_wave is None:
                    continue
                
                # 1. Enviar la onda cruda a la memoria de video
                x_wave = x_wave.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # 2. Ejecutar Extracción DSP asíncrona masiva en GPU
                x_deep, x_wide = feature_extractor(x_wave)

                logits = model(x_deep, x_wide)
                
                loss = criterion(logits, y)
                val_loss_accum += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_val_loss = val_loss_accum / num_val_batches

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        # 3. CÓMPUTO DE MÉTRICAS ESTRICTAS
        epoch_f1 = float(f1_score(y_true, y_pred, average="macro"))

        try:
            epoch_auc = float(roc_auc_score(
                y_true, 
                y_probs, 
                multi_class="ovr", 
                average="macro"
            ))
        except Exception:
            epoch_auc = 0.5

        # Registro en el historial
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_f1_macro"].append(epoch_f1)
        history["val_auc"].append(epoch_auc)

        # LÓGICA DE CHECKPOINTING
        if epoch_f1 > best_f1_macro:
            best_f1_macro = epoch_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            marcador_mejora = "[NUEVO MEJOR MODELO]"
        else:
            marcador_mejora = ""

        # Monitorización de progreso en tiempo real (Con el marcador incluido)
        print(
            f"Epoca {epoch+1:02d}/{num_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"F1-Score: {epoch_f1:.4f} " # | "
            #f"ROC-AUC: {epoch_auc:.4f}{marcador_mejora}"
        )

    # Restauración final
    model.load_state_dict(best_model_wts)
    print(f"\n[] Entrenamiento completado. Recuperando los pesos de la mejor época (F1-Score: {best_f1_macro:.4f}).")
    
    return history
```

### **Módulo de Checkpointing y Exportación**

Este bloque actúa como el Filtro de Calidad de Producción del pipeline de entrenamiento. Su propósito es garantizar la resiliencia operativa en la estación de rescate de campo, automatizando la decisión de promover los pesos entrenados de la red híbrida **Wide & Deep** o descartarlos de inmediato si no superan el rendimiento de los modelos ya desplegados, evitando regresiones en la precisión de la detección de vida.



```python
def guardar_modelo_produccion(
    model: nn.Module,
    metricas_actuales: Dict[str, float],
    ruta_archivo_pth: Union[str, Path],
    metrica_objetivo: str = "val_f1_macro",
    ruta_registro_json: Union[str, Path] = None
) -> bool:
    """Evalúa el rendimiento de los pesos entrenados frente a la marca histórica de producción.

    Esta función lee de forma segura el archivo de registro JSON local que mantiene
    la trazabilidad del mejor modelo en la Estación de Campo. Compara el desempeño actual 
    bajo una métrica objetivo (ej. F1-Score o AUC-ROC) y, si es superior, sobrescribe 
    el archivo de producción .pth, actualiza el JSON con los nuevos metadatos y emite 
    un reporte formal de promoción. En caso contrario, los pesos se descartan para 
    salvaguardar la precisión táctica.

    Parameters
    ----------
    model : nn.Module
        Instancia de la red neuronal híbrida USARWideDeepModel cuyos pesos se desean evaluar.
    metricas_actuales : Dict[str, float]
        Diccionario que contiene las métricas obtenidas durante la validación del modelo actual.
    ruta_archivo_pth : Union[str, Path]
        Ruta física destino del archivo de pesos en producción (.pth).
    metrica_objetivo : str, optional
        La métrica clave empleada para el control de la promoción. Por defecto es "val_f1_macro".
    ruta_registro_json : Union[str, Path], optional
        Ruta del archivo JSON de metadatos históricos. Si es None, se autogenerará en el 
        mismo directorio que 'ruta_archivo_pth' con el nombre 'registro_produccion.json'.

    Returns
    -------
    bool
        True si el modelo actual superó la marca histórica y fue promovido a producción.
        False si los pesos fueron descartados para evitar regresión.

    Raises
    ------
    ValueError
        Si la métrica objetivo especificada no existe en las métricas actuales del ciclo.
    """
    path_pth = Path(ruta_archivo_pth)
    path_pth.parent.mkdir(parents=True, exist_ok=True)

    # Autogeneración de la ruta del registro JSON de metadatos si no se define
    if ruta_registro_json is None:
        path_json = path_pth.parent / "registro_produccion.json"
    else:
        path_json = Path(ruta_registro_json)

    # Validar que la métrica de control esté presente en las evaluaciones del ciclo
    if metrica_objetivo not in metricas_actuales:
        raise ValueError(
            f"La métrica objetivo '{metrica_objetivo}' no está en las métricas provistas: "
            f"{list(metricas_actuales.keys())}"
        )

    nueva_marca = metricas_actuales[metrica_objetivo]
    mejor_marca_historica = 0.0
    registro_existe = path_json.exists()

    # Carga y validación del registro histórico existente
    if registro_existe:
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                registro = json.load(f)
            mejor_marca_historica = registro.get("mejor_marca", 0.0)
        except Exception as e:
            print(
                f"[] Advertencia al leer {path_json.name}: {e}. "
                f"Se asumirá mejor_marca = 0.0 para resguardar la promoción."
            )

    # Decisión automatizada de despliegue
    promover_a_produccion = (not registro_existe) or (nueva_marca > mejor_marca_historica)

    if promover_a_produccion:
        # 1. Serialización binaria de los parámetros (State Dict)
        torch.save(model.state_dict(), str(path_pth))

        # 2. Persistencia de metadatos de control para auditoría forense posterior
        nuevo_registro = {
            "mejor_marca": nueva_marca,
            "metrica_objetivo": metrica_objetivo,
            "metricas_detalladas": metricas_actuales,
            "ruta_pesos": str(path_pth.resolve())
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(nuevo_registro, f, indent=4, ensure_ascii=False)

        # 3. Reporte de ascenso en consola de grado industrial
        print("                 REPORTE DE PROMOCIÓN A PRODUCCIÓN           ")
        print(f" [ESTADO]   NUEVO MODELO PROMOVIDO EXITOSAMENTE")
        print(f" [RUTA .PTH] {path_pth.resolve()}")
        print(f" [MÉTRICA]  {metrica_objetivo}")
        print(f"   - Anterior Marca Histórica: {mejor_marca_historica:.6f}")
        print(f"   - Nueva Marca Registrada:   {nueva_marca:.6f}")
        print(f"   - Margen de Mejora:         {+(nueva_marca - mejor_marca_historica):+.6f}")
        print("-"*80)
        print(" [DETALLES DE LA INSTANCIA ADQUIRIDA]")
        for k, v in metricas_actuales.items():
            print(f"   - {k:<20}: {v:.6f}")
        return True
    else:
        # Alerta de descarte para el Ingeniero de Machine Learning (Evitar degradación)
        print("                 ALERTA DE RECHAZO DE PESOS                ")
        print(f" [ESTADO]   MODELO DESCARTADO - NO SUPERA EL RENDIMIENTO HISTÓRICO")
        print(f" [MÉTRICA]  {metrica_objetivo}")
        print(f"   - Mejor Marca Histórica:  {mejor_marca_historica:.6f}")
        print(f"   - Intento Reciente:       {nueva_marca:.6f}")
        print(f"   - Margen de Deficiencia:  {-(mejor_marca_historica - nueva_marca):+.6f}")
        print(" [ACCIÓN]   Los pesos actuales se descartan para prevenir regresiones en producción.")
        print("            El archivo de producción .pth no ha sido modificado.")
        return False

```


```python
# CONSTANTES DE CONFIGURACIÓN DEL PIPELINE
DIRECTORIO_DATASET = Path(r"C:\Users\carlo\Documents\detector_vida_acustico\detector_vida_acustico\dataset_preprocesado")
MODELO_SALIDA_PATH = Path("produccion/modelo_usar_v2.pth")
REGISTRO_JSON_PATH = Path("produccion/registro_produccion.json")
SEMILLA_ = 42
PROPORCION_VAL = 0.2
NUM_CLASES = 4
EPOCAS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def main() -> None:
    """Orquesta y ejecuta de forma secuencial las Fases 2 del pipeline

    """
    print("             INICIANDO FASE 2            ")

    # FASE 2: CARGA, PREPARACIÓN Y BALANCEO DE DATOS
    print("\n[] Fase 2: Iniciando parseo y unificación de datasets...")
    
    # Uso correcto de la variable DIRECTORIO_DATASET
    if not DIRECTORIO_DATASET.exists():
        print(f"[CRÍTICO] Directorio de datos de entrada no encontrado: {DIRECTORIO_DATASET.resolve()}")
        sys.exit(1)

    try:
        # CORRECCIÓN: Se pasa la ruta dentro de una lista []
        df_unificado = parsear_y_unificar_datasets([DIRECTORIO_DATASET])
    except NameError:
        print("[CRÍTICO] La función 'parsear_y_unificar_datasets' no está definida en el entorno.")
        sys.exit(1)

    if df_unificado is None or df_unificado.empty:
        print("[CRÍTICO] El DataFrame unificado está vacío. Abortando ejecución del pipeline.")
        sys.exit(1)

    print(f"[] Dataset unificado exitosamente. Total de muestras físicas: {len(df_unificado)}")

    print(f"[] Aplicando división estratificada (proporción val: {PROPORCION_VAL})...")
    df_train, df_val = dividir_datos_estratificados(
        df_unificado=df_unificado,
        test_size=PROPORCION_VAL,
        random_state=SEMILLA_
    )
    print(f"[] Partición completada. Entrenamiento: {len(df_train)} | Validación: {len(df_val)}")

    print("[] Calculando pesos compensatorios de pérdida para el conjunto de entrenamiento...")
    class_weights = calcular_pesos_clase(df_train)
    print(f"[] Pesos de clase calculados y validados: {class_weights.tolist()}")

    print("[] Instanciando cargadores de datos dinámicos (USARDataset) en PyTorch...")
    # Se activa la aumentación estocástica exclusivamente para el conjunto de entrenamiento
    train_dataset = USARDatasetLight(df=df_train, sr=16000, is_train=True)
    val_dataset = USARDatasetLight(df=df_val, sr=16000, is_train=False)
    print("[] Instanciación de datasets de PyTorch completada de forma correcta.")

    # FASE 3: INSTANCIACIÓN DE MODELO, ENTRENAMIENTO Y OPTIMIZACIÓN
    print("\n[] Fase 3: Evaluando disponibilidad de aceleración de hardware...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[] Dispositivo seleccionado para el entrenamiento: {device}")
    if device == "cuda":
        print(f"[] Aceleradora detectada: {torch.cuda.get_device_name(0)}")

    print(f"[] Inicializando arquitectura híbrida Wide & Deep (Clases: {NUM_CLASES})...")
    model = USARWideDeepModel(num_classes=NUM_CLASES)

    print(f"[] Iniciando ciclo de optimización forzado a {EPOCAS} épocas...")
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=EPOCAS,
        device=device
    )
    print("[] Entrenamiento finalizado. Extrayendo métricas de la última época...")

    # FASE 3: AUDITORÍA DE MÉTRICAS
    mejor_indice = int(np.argmax(history["val_f1_macro"]))

    metricas_actuales = {
        "val_loss": history["val_loss"][mejor_indice],
        "val_f1_macro": history["val_f1_macro"][mejor_indice],
        "val_auc": history["val_auc"][mejor_indice]
    }

    print(f"[] Métricas obtenidas en el último ciclo:")
    print(f"  - Pérdida de Validación (Loss) : {metricas_actuales['val_loss']:.6f}")
    print(f"  - F1-Score Macro               : {metricas_actuales['val_f1_macro']:.6f}")
    print(f"  - ROC-AUC (OvR)                : {metricas_actuales['val_auc']:.6f}")

    promovido = guardar_modelo_produccion(
        model=model,
        metricas_actuales=metricas_actuales,
        ruta_archivo_pth=MODELO_SALIDA_PATH,
        metrica_objetivo="val_f1_macro",
        ruta_registro_json=REGISTRO_JSON_PATH
    )

    if promovido:
        print("[] Operación de despliegue finalizada con éxito. Nuevo modelo activo en producción.")
    else:
        print("[] Operación finalizada. Los pesos del entrenamiento actual no fueron promovidos.")

    print("                         PIPELINE COMPLETADO EXITOSAMENTE                       ")


if __name__ == "__main__":
    main()

```


```python

```
