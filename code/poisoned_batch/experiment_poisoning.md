======================================================
Эксперимент 1: Poisoned Batch & Causal Discovery (CCM)
======================================================

В данном разделе представлен код для валидации метода Convergent Cross Mapping (CCM) на скалярных логах обучения нейронных сетей (ResNet-18 на датасете CIFAR-10). Архитектура проекта построена по принципу "Data Generation -> Data Analysis".

Модули ядра (Python Scripts)
----------------------------
Вся тяжелая математика и ООП-логика вынесены в переиспользуемые библиотеки:

* ``batch_poisoning.py``: Объектно-ориентированная реализация искусственных драйверов (Sinusoidal, Logistic Map, Random, Square Wave).
* ``search_for_optimal_parameters.py``: Математическое ядро фазовой реконструкции (поиск $\tau$ через DMI, оценка $E$ через FNN/Cao/MLE).
* ``visualisation_ccm.py``: Инструменты отрисовки графиков конвергенции $\rho(L)$.
* ``ccm_pipeline.py``: Высокоуровневые функции-обертки для автоматизации полного цикла CCM (интеграция поиска параметров и генерации графиков).

Исследовательские ноутбуки (Jupyter Notebooks)
----------------------------------------------
Эксперименты разбиты на логические пары "Генератор логов" и "Анализатор".

**1. Основной эксперимент (Base Scenarios)**

* ``Time_series_generation.ipynb`` — Обучение ResNet-18 с базовыми сценариями отравления(Discrete, Random, Sinusoidal, Progressive Noise and Logistic map) и сохранение логов.
* ``Time_series_analysis.ipynb`` — Применение CCM пайплайна к полученным логам. Вычисление параметров вложения. Как результат: получение границ применимости метода работающего в простых случаях и не работающего при наложении шумов

**2. Анализ границ применимости (Limit / Without Convergence)**

* ``Time_series_generation_without_convergence.ipynb`` — Тестирование Ghost Drivers(const = 0.0, const = 0.1, Normal and Uniform).
* ``Time_series_analysis_without_convergence.ipynb`` — Доказательство того что CCM не выдаёт ложно положительных результатов.

**3. Визуальный эксперимент**

* ``Time_series_generation_visual_experiment.ipynb`` / ``Time_series_analysis_visual_experiment.ipynb`` — Ноутбуки деманстрируют наглядный эксперимент, где с увелечением rate для Discrete Randomness метод начинает работать лучше и уверенее находит связи

Директории с результатами
-------------------------
* ``folder_for_raw_series/``, ``ghost_raw_series_logs/``, ``visual_experiment_raw_series/`` — Сгенерированные CSV файлы логов.
* ``ccm_results/``, ``ccm_ghost_results/``, ``ccm_visual_experiment_results`` — Финальные графики конвергенции. Папки с суффиксом ``_with_tau_one`` содержат результаты эксперимента, где параметр $\tau$ жестко зафиксирован равным 1.
