import matplotlib.pyplot as plt

def plot_normalized_optimization(gtex_mir_numeric, tissues_to_plot, results_k):
    fig, axes = plt.subplots(len(tissues_to_plot), 2, figsize=(15, 5 * len(tissues_to_plot)))
    
    if len(tissues_to_plot) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, tissue in enumerate(tissues_to_plot):
        tissue_data = gtex_mir_numeric.loc[tissue]
        n_samples = tissue_data.shape[0]
        limit_k = min(40, n_samples)
        ks = np.arange(1, limit_k + 1)
        
        distances = []
        variances = []
        
        for k in ks:
            # Усредняем 100 раз для точности оценки
            batch = np.array([tissue_data.iloc[np.random.choice(n_samples, k)].mean(axis=0) for _ in range(100)])
            distances.append(np.mean(pdist(batch, metric='euclidean')))
            variances.append(np.var(batch, axis=0).mean())

        # НОРМИРОВКА: Делим всё на значение при K=1
        norm_distances = np.array(distances) / distances[0]
        norm_variances = np.array(variances) / variances[0]
        
        opt_k = results_k[tissue]
        remaining_var = norm_variances[opt_k - 1] * 100 # % вариативности при Opt K

        # --- График 1: Normalized Distance ---
        ax_dist = axes[i, 0]
        ax_dist.plot(ks, norm_distances, 'b-o', markersize=4)
        ax_dist.axvline(x=opt_k, color='green', linestyle=':', label=f'Opt K = {opt_k}')
        ax_dist.set_title(f"{tissue}\nNormalized Distance (1.0 = Original)")
        ax_dist.set_ylim(0, 1.1)
        ax_dist.grid(True, alpha=0.3)

        # --- График 2: Normalized Variance ---
        ax_var = axes[i, 1]
        ax_var.plot(ks, norm_variances, 'm-s', markersize=4, label='Real Variance')
        ax_var.plot(ks, 1/ks, 'k--', alpha=0.5, label='Theoretical 1/K')
        ax_var.axvline(x=opt_k, color='green', linestyle=':', label=f'Opt K = {opt_k}')
        
        ax_var.set_title(f"Remaining Variance at K={opt_k}: {remaining_var:.1f}%")
        ax_var.set_ylabel("Fraction of Original Variance")
        ax_var.set_ylim(0, 1.1)
        ax_var.legend()
        ax_var.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()





import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

def find_optimal_k_refined(data_tissue, max_k=40):
    n_samples = data_tissue.shape[0]
    # K не может быть больше количества реальных образцов
    limit_k = min(max_k, n_samples)
    
    distances = []
    ks = np.arange(1, limit_k + 1)
    
    for k in ks:
        # Генерируем 100 синтетических образцов для замера стабильности
        synthetic_batch = []
        for _ in range(100):
            idx = np.random.choice(n_samples, k, replace=True)
            synthetic_batch.append(data_tissue.iloc[idx].mean(axis=0))
        
        # Считаем среднее расстояние между ними
        avg_dist = np.mean(pdist(np.array(synthetic_batch), metric='euclidean'))
        distances.append(avg_dist)
    
    # Если точек слишком мало для поиска локтя, возвращаем минимум
    if len(distances) < 3:
        return 1
    
    # Поиск "локтя" через нормализованные расстояния
    # Мы ищем точку, где падение замедляется максимально сильно
    coords = np.vstack((ks, distances)).T
    # Вектор от первой до последней точки
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Расстояние от каждой точки до прямой, соединяющей начало и конец
    vec_from_first = coords - coords[0]
    scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_prod, line_vec_norm)
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    # Точка с максимальным расстоянием до прямой — это и есть наш "локоть"
    best_k = ks[np.argmax(dist_to_line)]
    
    return best_k

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.spatial.distance import pdist

def save_gtex_optimization_report(gtex_mir_numeric, results_k, output_dir='results', filename='GTEx_K_Optimization_Results.pdf'):
    """
    Создает многостраничный PDF-отчет с графиками оптимизации для всех тканей.
    """
    # Создаем директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    full_path = os.path.join(output_dir, filename)

    with PdfPages(full_path) as pdf:
        tissues = gtex_mir_numeric.index.unique()
        print(f"Начинаю генерацию отчета для {len(tissues)} тканей...")

        for tissue in tissues:
            tissue_data = gtex_mir_numeric.loc[tissue]
            
            # Проверка на достаточное количество образцов
            if len(tissue_data.shape) == 1 or tissue_data.shape[0] < 5:
                continue
                
            n_samples = tissue_data.shape[0]
            limit_k = min(40, n_samples)
            ks = np.arange(1, limit_k + 1)
            
            distances, variances = [], []
            for k in ks:
                # Бутстреп: 100 итераций для каждой точки K
                batch = np.array([tissue_data.iloc[np.random.choice(n_samples, k, replace=True)].mean(axis=0) for _ in range(100)])
                distances.append(np.mean(pdist(batch, metric='euclidean')))
                variances.append(np.var(batch, axis=0).mean())

            # Нормировка
            norm_distances = np.array(distances) / distances[0]
            norm_variances = np.array(variances) / variances[0]
            
            # Получаем K из переданного словаря
            opt_k = results_k[tissue]
            remaining_var = norm_variances[opt_k - 1] * 100

            # Отрисовка
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Tissue: {tissue} (N={n_samples})", fontsize=16)

            # График 1: Distance
            ax1.plot(ks, norm_distances, 'b-o', markersize=4)
            ax1.axvline(x=opt_k, color='green', linestyle='--', label=f'Opt K = {opt_k}')
            ax1.set_title("Normalized Distance (Elbow Method)")
            ax1.set_ylabel("Fraction of Initial Distance")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # График 2: Variance
            ax2.plot(ks, norm_variances, 'm-s', markersize=4, label='Real Variance')
            ax2.plot(ks, 1/ks, 'k--', alpha=0.3, label='Theoretical 1/K')
            ax2.axvline(x=opt_k, color='green', linestyle='--', label=f'Opt K = {opt_k}')
            ax2.set_title(f"Normalized Variance (Remains: {remaining_var:.1f}%)")
            ax2.set_ylabel("Fraction of Initial Variance")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Сохранение и очистка памяти
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Готово! Отчет сохранен по адресу: {full_path}")




import numpy as np

def find_optimal_k_balanced(data_tissue, max_k=40, n_bootstrap=100,
                           min_variance=0.25,  # сколько дисперсии хотим сохранить
                           improvement_tol=0.02  # минимальное улучшение
                          ):
    """
    Подбор K с балансом между сглаживанием и сохранением вариации.

    Параметры:
    - min_variance: минимальная доля сохраняемой дисперсии (например 0.25 = 25%)
    - improvement_tol: насколько должно улучшаться (уменьшаться variance), чтобы имело смысл увеличивать K
    """

    n_samples = data_tissue.shape[0]
    limit_k = min(max_k, n_samples)
    ks = np.arange(1, limit_k + 1)

    variances = []

    for k in ks:
        batch = np.array([
            data_tissue.iloc[np.random.choice(n_samples, k, replace=True)].mean(axis=0)
            for _ in range(n_bootstrap)
        ])
        
        # средняя variance по всем фичам
        variances.append(np.var(batch, axis=0).mean())

    variances = np.array(variances)
    norm_variances = variances / variances[0]

    # --- 1. Ограничение по дисперсии ---
    valid_ks = ks[norm_variances >= min_variance]

    if len(valid_ks) == 0:
        # если все слишком сглажено — берем минимальный K
        return 1

    # --- 2. diminishing returns ---
    # считаем, насколько уменьшается variance при увеличении K
    improvements = -np.diff(norm_variances)  # положительные значения = уменьшение variance

    # нормируем относительно первого шага
    improvements = improvements / improvements[0]

    # ищем точку, где улучшение становится слишком маленьким
    plateau_idx = np.where(improvements < improvement_tol)[0]

    if len(plateau_idx) > 0:
        best_k_plateau = ks[plateau_idx[0]]
    else:
        best_k_plateau = valid_ks[-1]

    # --- финальный выбор ---
    # берем минимальный из:
    # - не слишком сглаженного
    # - и уже без сильного выигрыша
    best_k = min(valid_ks[-1], best_k_plateau)

    return int(best_k)
    
import pandas as pd
import numpy as np

def bootstrap_miRNA_adaptive(gtex_mir: pd.DataFrame, 
                             rna_samples: pd.Index, 
                             results_k: dict) -> pd.DataFrame:
    """
    gtex_mir: DataFrame с экспрессией miRNA (индексы - названия тканей).
    rna_samples: Список имен образцов мРНК, для которых ищем пары.
    results_k: Словарь {ткань: оптимальное_K}.
    """
    df_list = []
    
    # Чтобы ускорить работу, заранее получим список уникальных тканей из индекса miRNA
    available_tissues = gtex_mir.index.unique()

    print(f"Начинаю генерацию синтетических профилей для {len(rna_samples)} образцов...")

    for i, sample_name in enumerate(rna_samples):
        # 1. Определяем ткань образца мРНК
        # (Логика split зависит от того, как именно у тебя названы образцы)
        # Если в rna_samples просто названия тканей, берем целиком. 
        # Если там ID вроде "Adipose_Subcutaneous_123", берем часть до ткани.
        # В данном примере ищем совпадение в словаре results_k
        
        found_tissue = None
        for tissue in results_k.keys():
            if sample_name.startswith(tissue):
                found_tissue = tissue
                break
        
        if not found_tissue:
            # Если точного совпадения нет, попробуем упрощенный split
            found_tissue = sample_name.split(' - ')[0] 

        # 2. Получаем индивидуальное K для этой ткани
        # Если ткани нет в словаре (мало образцов), по умолчанию берем 1
        k_val = results_k.get(found_tissue, 1)

        # 3. Отбираем кандидатов (микроРНК той же ткани)
        try:
            candidates = gtex_mir.loc[found_tissue]
        except KeyError:
            # Если ткани нет в miRNA датасете, пропускаем или заполняем нулями
            continue

        # Обработка случая, если loc вернул Series (когда образец всего один)
        if isinstance(candidates, pd.Series):
            mean_profile = candidates
        else:
            # 4. БУТСТРЕП: берем k_val образцов с возвращением (replace=True)
            # Это гарантирует, что мы получим k образцов, даже если их в ткани меньше
            subset = candidates.sample(n=k_val, replace=True)
            mean_profile = subset.mean(axis=0)

        mean_profile.name = sample_name
        df_list.append(mean_profile)
        
        # Периодический отчет о прогрессе
        if (i + 1) % 5000 == 0:
            print(f"Обработано {i + 1} образцов...")

    # Собираем финальный датафрейм
    df_bootstrap = pd.DataFrame(df_list)
    
    print(f"Сборка завершена. Итоговый размер: {df_bootstrap.shape}")
    return df_bootstrap



import pandas as pd
import numpy as np

def generate_noise_reduction_stats(gtex_mir_numeric, results_k):
    stats_list = []
    
    for tissue in gtex_mir_numeric.index.unique():
        tissue_data = gtex_mir_numeric.loc[tissue]
        if len(tissue_data.shape) == 1 or tissue_data.shape[0] < 2:
            continue
            
        n_samples = tissue_data.shape[0]
        opt_k = results_k[tissue]
        
        # 1. Считаем абсолютную дисперсию для каждого miRNA и берем среднее
        # Это дает средний разброс экспрессии на одну miRNA
        var_initial_abs = tissue_data.var(axis=0).mean()
        
        # 2. Считаем дисперсию после бутстрепа (усредняем 100 синтетических пар)
        synthetic_samples = np.array([
            tissue_data.iloc[np.random.choice(n_samples, opt_k, replace=True)].mean(axis=0) 
            for _ in range(100)
        ])
        var_optimized_abs = synthetic_samples.var(axis=0).mean()
        
        # 3. Разница (сколько "силы" шума мы убрали в абсолютных числах)
        noise_removed_abs = var_initial_abs - var_optimized_abs
        
        stats_list.append({
            'Tissue': tissue,
            'N': n_samples,
            'K': opt_k,
            'Initial_Var_Abs': round(var_initial_abs, 4),
            'Remaining_Var_Abs': round(var_optimized_abs, 4),
            'Noise_Removed_Abs': round(noise_removed_abs, 4),
            'Remaining_Var_%': round((var_optimized_abs / var_initial_abs) * 100, 2),
            'Noise_Reduction_%': round((1 - var_optimized_abs / var_initial_abs) * 100, 2)
        })

    df_stats = pd.DataFrame(stats_list)
    return df_stats