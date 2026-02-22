# LinkReview

- Here we collect all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them

> [!NOTE]
> This review table will be updated, so it is not a final version.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| Edge of Stability(экспериментальное получение эффекта) | Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability | 2021 | Cohen et al. | [arXiv](https://openreview.net/pdf?id=jh-rTtvkGeM) | [GitHub](https://github.com/locuslab/edge-of-stability) | Эмпирически показывают, что при обучении нейросетей полносвязным градиентным спуском динамика обычно выходит в режим Edge of Stability, где максимальное собственное значение гессиана($\lambda_{max}$) держится чуть выше порога $\frac{2}{\eta}$, loss ведёт себя немонотонно на коротких масштабах, но убывает на длинных; это показывает, что классическая теория далеко не польностью описывает процесс оптимизации(а именно то, что система обычно входит в EoS) и указывает на необходимость новой теории оптимизации для нейросетей, чтобы можно было более оптимально подбирать шаг, опираясь только на теорию. |
