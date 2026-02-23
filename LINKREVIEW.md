# LinkReview

- Here we collect all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them

> [!NOTE]
> This review table will be updated, so it is not a final version.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| Edge of Stability(экспериментальное получение эффекта) | Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability | 2021 | Cohen et al. | [arXiv](https://openreview.net/pdf?id=jh-rTtvkGeM) | [GitHub](https://github.com/locuslab/edge-of-stability) | Эмпирически показывают, что при обучении нейросетей полносвязным градиентным спуском динамика обычно выходит в режим Edge of Stability, где максимальное собственное значение гессиана($\lambda_{max}$) держится чуть выше порога $\frac{2}{\eta}$, loss ведёт себя немонотонно на коротких масштабах, но убывает на длинных; это показывает, что классическая теория далеко не польностью описывает процесс оптимизации(а именно то, что система обычно входит в EoS) и указывает на необходимость новой теории оптимизации для нейросетей, чтобы можно было более оптимально подбирать шаг, опираясь только на теорию. |
| CCM / Causal Inference / EDM | Detecting Causality in Complex Ecosystems | 2012 | Sugihara et al. | [Science](https://www.science.org/doi/10.1126/science.1227079) | [GitHub](https://github.com/SugiharaLab/pyEDM) | Вводят Convergent Cross Mapping (CCM) для выявления нелинейной причинности в динамических системах черех временные ряды и реконструкцию аттракторов (Теорема Такенса). Конвергенция ρ(L) при росте sizeLibrary L отличает реальные связи от ложных корреляций. Показывают двусторонние связи (хищник-жертва Paramecium-Didinium), одностороннее влияние (SST→сардины/анчоусы) и отличают общий драйвер от конкуренции. Превосходит Granger causality в нелинейных системах с слабой связью, а также в системах где есть драйвер. |
| Simplex projection  / EDM | Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series | 1990 | Sugihara G., May R.M. | [Nature 344:734-741](https://www.nature.com/articles/344734a0) | [GitHub](https://github.com/SugiharaLab/pyEDM) | Вводит simplex projection для различения детерминированного хаоса от шума/измерительных ошибок через нелинейное краткосрочное прогнозирование. Анализ кривой предсказуемости ρ(τ) на реальных данных. |
