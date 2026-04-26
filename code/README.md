# Detecting Optimization Regimes via Convergent Cross Mapping

Данная директория содержит исходный код для вычислительных экспериментов, представленных в статье. Проект демонстрирует применимость методов эмпирического динамического моделирования (EDM) и фазовой реконструкции к анализу логов глубокого обучения.

Исследование разделено на два независимых модуля:

1.[**Poisoned Batch (Causal Discovery)**](poisoned_batch/experiment_poisoning.rst)  
   Эксперимент по выявлению скрытых причинно-следственных связей в условиях стохастического шума оптимизатора SGD с использованием метода Convergent Cross Mapping (CCM).

2. [**Grokking (Dimensionality Collapse)**](grokking/experiment_grokking.rst)  
   Исследование феномена отложенной генерализации (гроккинга). Содержит код для обучения Трансформеров на алгоритмических задачах (модульная арифметика и группа $S_5$) и скрипты для отслеживания коллапса внутренней размерности аттрактора (через MLE Intrinsic Dimension и другие методы).

## Зависимости

Для запуска экспериментов требуются следующие библиотеки:
```bash
pip install torch torchvision pandas numpy matplotlib tqdm scikit-learn
pip install causal-ccm einops
