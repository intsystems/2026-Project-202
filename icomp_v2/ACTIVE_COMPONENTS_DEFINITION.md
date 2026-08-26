# Как определить активные компоненты и активную размерность

## 1. Главная рекомендация

Не следует вводить неопределённое множество

$$
\mathcal A_W
$$

как «множество состояний, которое траектория занимает в окне». У этой фразы есть три несовместимые интерпретации:

1. Если это конечный набор
   $$
   \{\theta_t:t\in W\},
   $$
   его классическая box-counting dimension равна нулю.
2. Если соединить соседние точки непрерывной кривой, размерность этой кривой обычно равна единице.
3. Если имеется в виду множество, которое рекуррентная динамика заполняет за неограниченное время, это уже support долгосрочной occupation measure, а не конечное окно.

Для статьи лучше строго разделить:

- **динамический режим** $R$, для которого существует популяционная occupation measure;
- **геометрическое множество режима** $\mathcal A_R$, определённое как support этой меры;
- **конечное окно** $W$, которое даёт выборку из режима и используется только для оценки;
- **MG-оценку** $\widehat d_{\mathrm{MG}}(W)$;
- **effective rank**, который является отдельной линейной характеристикой.

Основная теоретическая величина должна обозначаться

$$
d_{\mathrm{act}}(R),
$$

а не

$$
d_{\mathrm{act}}(W),
$$

если только явно не предполагается, что всё окно принадлежит одному локально стационарному режиму $R(W)$.

---

## 2. Полное состояние и параметрическая траектория

Пусть

$$
s_t\in\mathcal S
$$

— полное состояние обучения. Оно может включать параметры, momentum, Adam moments, состояние scheduler и другие переменные.

Параметры являются проекцией полного состояния:

$$
\theta_t=\pi_\theta(s_t)\in\mathbb R^P.
$$

Если параметры по построению лежат в аффинном подпространстве

$$
\theta=\theta_0+V^\top c,
\qquad
V\in\mathbb R^{k\times P},
\qquad
VV^\top=I_k,
$$

то свободные координаты равны

$$
c=V(\theta-\theta_0)\in\mathbb R^k.
$$

Поскольку это изометрическая замена координат внутри доступного подпространства, intrinsic dimension в $c$-space и parameter space одинакова. Далее можно работать с $c_t$.

---

## 3. Что такое динамический режим

Динамический режим $R$ — участок работы системы, на котором:

1. закон обновления и внешние условия считаются неизменными;
2. существует стационарная или инвариантная вероятностная мера полного состояния;
3. наблюдаемый временной ряд достаточно долго семплирует эту меру.

Обозначим эту меру через

$$
\nu_R.
$$

Для детерминированной эргодической системы её можно определить через долгосрочную частоту посещений:

$$
\nu_R(C)
=
\lim_{T\to\infty}
\frac1T
\sum_{t=0}^{T-1}
\mathbf 1\{s_t\in C\},
$$

если предел существует для рассматриваемых измеримых множеств $C\subseteq\mathcal S$.

Для стационарной стохастической системы $\nu_R$ — её стационарный закон.

Мера параметрической траектории получается проекцией:

$$
\mu_R=(\pi_c)_\#\nu_R,
$$

то есть для любого борелевского множества $B\subseteq\mathbb R^k$

$$
\mu_R(B)
=
\nu_R\!\left(\pi_c^{-1}(B)\right).
$$

Здесь $(\pi_c)_\#$ обозначает pushforward меры.

---

## 4. Нормальное определение множества $\mathcal A_R$

Геометрическое множество, занимаемое параметрами в режиме $R$, определяется как support меры $\mu_R$:

$$
\boxed{
\mathcal A_R
:=
\operatorname{supp}\mu_R
=
\left\{
c\in\mathbb R^k:
\mu_R(B(c,r))>0
\text{ для каждого }r>0
\right\}.
}
$$

Иными словами, точка $c$ принадлежит $\mathcal A_R$ тогда и только тогда, когда любая её окрестность имеет ненулевую вероятность посещения в режиме $R$.

Это точное определение. Оно не использует расплывчатую фразу «состояния, которые динамика могла бы занимать».

Для типичной рекуррентной траектории эргодической детерминированной системы

$$
\mathcal A_R
=
\overline{\{c_t:t\geq0\}},
$$

где рассматривается траектория после начального транзиента и внутри одного режима.

### Почему обозначение $\mathcal A_W$ хуже

Пусть

$$
W=\{t_0,\ldots,t_0+L-1\}
$$

— конечное окно. Оно определяет эмпирическую меру

$$
\widehat\mu_W
=
\frac1L\sum_{t\in W}\delta_{c_t}.
$$

Support этой меры равен конечному набору наблюдавшихся точек. Поэтому

$$
\dim\operatorname{supp}\widehat\mu_W=0
$$

в классическом пределе малого масштаба.

Следовательно, окно не определяет популяционное многообразие. Оно только предоставляет конечную зависимую выборку из $\mu_R$.

Если необходимо сохранить оконную нотацию, нужно написать явно:

$$
\mu_W:=\mu_{R(W)},
\qquad
\mathcal A_W:=\operatorname{supp}\mu_{R(W)},
$$

где $R(W)$ — режим, из которого, согласно отдельному предположению, семплировано окно. В таком определении $\mathcal A_W$ — не множество точек окна.

Если окно пересекает фазовый переход и ему нельзя приписать один локально стационарный режим, единственного $\mathcal A_W$ нет. В этом случае можно сообщать оконную статистику, но её нельзя без дополнительных аргументов называть размерностью одного инвариантного множества.

---

## 5. Определение активной размерности

Есть два совместимых способа сформулировать определение.

### 5.1. Через размерность многообразия

Если $\mathcal A_R$ является гладким $d$-мерным многообразием, по крайней мере почти всюду относительно $\mu_R$, то

$$
\boxed{
d_{\mathrm{act}}(R)
:=
\dim\mathcal A_R
=d.
}
$$

Около почти каждой точки $c\in\mathcal A_R$ существует карта

$$
\Psi_R:U\subseteq\mathbb R^d\to\mathcal A_R,
$$

для которой

$$
c=\Psi_R(z_1,\ldots,z_d),
\qquad
\operatorname{rank}D\Psi_R(z)=d.
$$

Локальные координаты

$$
z_1,\ldots,z_d
$$

можно называть **активными компонентами режима $R$**. Их количество равно $d$.

Сами компоненты не уникальны: любую гладкую обратимую замену координат можно считать другой системой активных компонент. Инвариантным является их количество.

### 5.2. Через локальную размерность меры

Более общая формулировка, напрямую связанная с nearest-neighbour estimator, использует локальное масштабирование массы:

$$
d_{\mu_R}(c)
:=
\lim_{r\downarrow0}
\frac{\log\mu_R(B(c,r))}{\log r},
$$

если предел существует.

Если

$$
d_{\mu_R}(c)=d
$$

для $\mu_R$-почти всех $c$, мера называется exact-dimensional, и можно определить

$$
\boxed{
d_{\mathrm{act}}(R):=d.
}
$$

Если $\mu_R$ имеет положительную гладкую плотность на гладком $d$-мерном многообразии $\mathcal A_R$, оба определения совпадают:

$$
d_{\mu_R}(c)
=
\dim\mathcal A_R
=d
$$

почти всюду.

В статье лучше дать оба определения: первое объясняет смысл «числа компонент», второе точно соответствует mass-scaling предпосылке MG.

Если размерность нецелая, допустимо говорить об active dimension, но уже не следует буквально говорить о $d$ отдельных компонентах.

---

## 6. Конечномасштабная активная размерность

Теоретический предел $r\downarrow0$ недоступен по конечной записи. Поэтому отдельно определяется конечномасштабный показатель:

$$
d_R(c;r_1,r_2)
:=
\frac{
\log\mu_R(B(c,r_2))
-
\log\mu_R(B(c,r_1))
}{
\log r_2-
\log r_1
},
\qquad
0<r_1<r_2.
$$

Агрегированную конечномасштабную размерность режима можно определить как

$$
d_{\mathrm{act}}(R;r_1,r_2)
:=
\operatorname{median}_{c\sim\mu_R}
d_R(c;r_1,r_2).
$$

Выбор медианы не является единственно возможным; можно использовать среднее или явно сообщать распределение локальных значений. Правило агрегации должно быть зафиксировано заранее.

Если на диапазоне масштабов выполняется

$$
\mu_R(B(c,r))\approx C(c)r^d,
$$

то конечномасштабная размерность близка к $d$.

Эта версия позволяет корректно описать слабые компоненты. Если амплитуда одной координаты меньше $r_1$, она практически не разрешается, хотя теоретическая размерность при $r\downarrow0$ может быть больше.

---

## 7. Что измеряет конечное окно

Окно $W$ не определяет истинную размерность, а даёт оценку:

$$
\widehat d_{\mathrm{act}}(W)
\approx
d_{\mathrm{act}}(R)
$$

или конечномасштабную оценку

$$
\widehat d_{\mathrm{act}}(W;r_1,r_2)
\approx
d_{\mathrm{act}}(R;r_1,r_2),
$$

если выполнены следующие условия:

1. окно целиком принадлежит одному режиму $R$;
2. динамика достаточно покрывает $\mathcal A_R$;
3. масштаб больше уровня шума и дискретизации;
4. масштаб меньше радиуса кривизны и расстояния между различными ветвями множества;
5. результат устойчив к длине окна и параметрам estimator.

Если эти условия не выполнены, корректное обозначение — название вычисленной статистики, например

$$
\widehat d_{\mathrm{MG}}(W),
$$

а не безусловное равенство

$$
\widehat d_{\mathrm{MG}}(W)=d_{\mathrm{act}}(R).
$$

---

## 8. Квазипериодические системы с известным числом компонент

Пусть

$$
\varphi_t
=(\varphi_1(t),\ldots,\varphi_q(t))
\in\mathbb T^q
$$

— $q$ рационально независимых фаз, а

$$
c_t=F(\varphi_t).
$$

Если $F:\mathbb T^q\to\mathbb R^k$ является гладким вложением, то

$$
\mathcal A_R=F(\mathbb T^q)
$$

и

$$
d_{\mathrm{act}}(R)=q.
$$

Одной рациональной независимости фаз недостаточно. Необходимо также проверить, что $F$ не теряет координаты. Достаточное условие для используемых конструкций — доказать, что $F$ является вложением; как минимум нужно проверить

$$
\operatorname{rank}DF(\varphi)=q
$$

почти всюду и исключить вырождение образа.

Effective rank здесь является sanity check того, что компоненты имеют разрешимые амплитуды. Ground truth $q$ следует из фазовой конструкции и свойств $F$, а не из effective rank.

---

## 9. Связь с delay reconstruction и MG

Пусть наблюдается скаляр

$$
x_t=\phi(s_t),
$$

и строятся delay coordinates

$$
y_t
=
(x_t,x_{t-\tau},\ldots,x_{t-(E-1)\tau}).
$$

Если delay map является вложением $\mathcal A_R$, intrinsic dimension его образа совпадает с

$$
d_{\mathrm{act}}(R).
$$

MG исходит из локальной модели

$$
\mu_R(B(c,r))\approx C(c)r^d
$$

и оценивает показатель $d$ по радиусам ближайших соседей. Его рабочий масштаб не фиксирован глобально: для каждой точки он задаётся расстояниями до её соседей.

Поэтому MG следует описывать так:

> MG is an estimator of the local mass-scaling dimension at the data-dependent nearest-neighbour scale. It estimates the active dimension only when the delay reconstruction is an embedding, the window samples one regime sufficiently densely, and a stable scaling range exists at the neighbour radii.

Это точнее, чем утверждение, что MG по определению возвращает $d_{\mathrm{act}}$.

---

## 10. Связь с effective rank

Participation ratio ковариации равен

$$
d_{\mathrm{PR}}
=
\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

Он измеряет число линейных направлений, несущих сопоставимую дисперсию. Это не intrinsic dimension.

Например, двумерное многообразие с очень слабой второй координатой сохраняет

$$
d_{\mathrm{act}}=2,
$$

но может иметь

$$
d_{\mathrm{PR}}\approx1.
$$

Поэтому effective rank можно использовать как:

- проверку возбуждения известных компонент;
- характеристику анизотропии;
- отдельную описательную статистику сохранённой траектории.

Его нельзя объявлять ground truth active dimension в неизвестной системе.

---

## 11. Детерминированный транзиент и стохастическая динамика

### Детерминированный транзиент

Конечный невозвратный транзиент обычно не семплирует инвариантную occupation measure режима. Поэтому в предлагаемом определении он не имеет отдельной асимптотической $\mathcal A_R$, которую можно восстановить по возвратам.

Можно сказать, что непрерывно интерполированный путь является кривой размерности один, но это другая величина: геометрическая размерность конкретного пути, а не размерность множества, заполняемого рекуррентным режимом. Эти определения нельзя смешивать.

Если система сходится к неподвижной точке, её асимптотическая occupation measure является delta-measure и

$$
d_{\mathrm{act}}=0.
$$

### Стохастическая динамика

У стационарной mini-batch динамики occupation measure может существовать. Её размерность включает направления, возбуждённые batch noise. Она не обязана совпадать с числом степеней свободы средней оптимизационной динамики.

Поэтому следует различать:

- observed occupation dimension стохастической траектории;
- structural dimension условного среднего update.

MG по одному noisy run автоматически не отделяет эти величины.

---

## 12. Полный текст определения для вставки в статью

Ниже приведён цельный LaTeX-блок. Он может заменить нынешнюю subsection `The active dimension` от первого предложения до перехода к описанию estimator. В нём $R$ — режим, а $\gW$ — конечное окно, используемое для оценки.

```latex
\subsection{Active components and active dimension}\label{sec:three}

A finite training window is a sample, not itself a population manifold. We therefore distinguish a
dynamical regime from the window used to estimate it. Let $\vs_t\in\gS$ denote the full optimiser
state and let $\vtheta_t=\pi_\theta(\vs_t)\in\R^P$ be its parameter component. When the parameters
are confined to the affine subspace
\begin{equation}
  \vtheta=\vtheta_0+\mV^\top\vc,
  \qquad
  \mV\mV^\top=\mI_k,
\end{equation}
we work in the free coordinates $\vc=\mV(\vtheta-\vtheta_0)\in\R^k$. This change of coordinates is
an isometry on the available subspace and therefore preserves intrinsic dimension. We call $k$ the
\emph{available dimension}.

Let $R$ be a dynamical regime whose full state admits an ergodic invariant probability measure
$\nu_R$. Its parameter-space occupation measure is the pushforward
\begin{equation}\label{eq:occupation}
  \mu_R=(\pi_c)_\#\nu_R,
  \qquad
  \mu_R(B)=\nu_R\!\left(\pi_c^{-1}(B)\right),
\end{equation}
where $\pi_c$ projects the full optimiser state to $\vc$. Equivalently, for a typical deterministic
trajectory and every set for which the limit exists,
\begin{equation}
  \mu_R(B)
  =
  \lim_{T\to\infty}\frac{1}{T}
  \sum_{t=0}^{T-1}\mathbf 1\{\vc_t\in B\}.
\end{equation}
The parameter-space set occupied by the regime is
\begin{equation}\label{eq:active-set}
  \gA_R
  :=
  \operatorname{supp}\mu_R
  =
  \left\{
  \vc\in\R^k:
  \mu_R(B(\vc,r))>0\ \text{for every }r>0
  \right\}.
\end{equation}
Thus $\gA_R$ contains exactly those parameter states whose every neighbourhood is visited with
positive asymptotic frequency in regime $R$. For a typical recurrent trajectory under the assumptions
above, $\gA_R$ is the closure of that trajectory after its initial transient.

Suppose that $\gA_R$ is a smooth $d$-dimensional manifold at $\mu_R$-almost every point and that
$\mu_R$ has a positive smooth density on it. A local chart then writes
\begin{equation}
  \vc=\Psi_R(z_1,\ldots,z_d),
  \qquad
  \operatorname{rank}D\Psi_R=d.
\end{equation}
We call $z_1,\ldots,z_d$ \emph{active components} of the regime and define their number, the
\emph{active dimension}, by
\begin{equation}\label{eq:act}
  d_{\mathrm{act}}(R)
  :=
  \dim\gA_R
  =d.
\end{equation}
The active coordinates themselves are not unique: a smooth invertible reparameterisation produces
another valid set of components. Their number is invariant. In the more general exact-dimensional
case, equation~\eqref{eq:act} is equivalently defined by the local mass-scaling exponent
\begin{equation}\label{eq:localdim}
  d_{\mathrm{act}}(R)
  =
  \lim_{r\downarrow0}
  \frac{\log\mu_R(B(\vc,r))}{\log r}
  \quad\text{for }\mu_R\text{-almost every }\vc.
\end{equation}

A window $\gW=\{t_0,\ldots,t_0+L-1\}$ supplies only the empirical measure
\begin{equation}\label{eq:empirical-occupation}
  \widehat\mu_{\gW}
  =
  \frac{1}{L}\sum_{t\in\gW}\delta_{\vc_t}.
\end{equation}
Its support is finite and has classical box-counting dimension zero. We therefore do not define the
active dimension as the dimension of the finite set $\{\vc_t:t\in\gW\}$. Instead, when $\gW$ lies
inside one approximately stationary regime $R$, it is used to estimate $d_{\mathrm{act}}(R)$. A window
that crosses a transition need not sample a single occupation measure, in which case a windowed
statistic is descriptive and is not automatically the dimension of one invariant set.

Finite data resolve only a range of radii. For $0<r_1<r_2$, define the finite-scale local exponent
\begin{equation}\label{eq:finitescale}
  d_R(\vc;r_1,r_2)
  :=
  \frac{
  \log\mu_R(B(\vc,r_2))-\log\mu_R(B(\vc,r_1))
  }{
  \log r_2-\log r_1
  }.
\end{equation}
The nearest-neighbour estimator used below estimates this exponent at data-dependent radii. Its
output may be read as an estimate of $d_{\mathrm{act}}(R)$ only when the neighbour radii lie in a
stable scaling range, the window covers the regime sufficiently densely, and the delay map is a
valid embedding. Otherwise we report it only as the statistic $\widehat d_{\mathrm{MG}}(\gW)$.

In our constructed recurrent systems,
\begin{equation}
  \vc_t=F(\varphi_1(t),\ldots,\varphi_q(t)),
\end{equation}
where the phases have rationally independent frequencies and $F:\mathbb T^q\to\R^k$ is constructed
to be a smooth embedding. Hence $\gA_R=F(\mathbb T^q)$ and
$d_{\mathrm{act}}(R)=q$. Rational independence makes the orbit dense on the torus; the embedding
condition ensures that $F$ does not discard or merge a phase. The covariance participation ratio is
used only to check that all $q$ components have numerically resolvable amplitudes; it is not the
definition or ground truth of active dimension.

The \emph{functional dimension} is a separate pointwise quantity, the rank of the Jacobian of the
outputs on a fixed probe set with respect to $\vc$. The covariance participation ratio is also
separate:
\begin{equation}
  \PR(\lambda)
  =
  \frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\end{equation}
Applied to the covariance spectrum of a trajectory, it measures how evenly variance is distributed
among linear directions. It can be close to one on a highly anisotropic $d$-dimensional manifold and
therefore need not equal $d_{\mathrm{act}}(R)$.
```

### Необходимая техническая правка макроса

Если в преамбуле ещё нет $\gA$, добавить, например:

```latex
\newcommand{\gA}{\mathcal{A}}
```

Либо во всём блоке заменить `\gA_R` на `\mathcal{A}_R`.

---

## 13. Что после этого нужно изменить в остальных частях статьи

### В абстракте

Вместо утверждения, что active dimension определяется на любом stretch, написать:

> We define the active dimension of a recurrent training regime as the local dimension of its parameter-space occupation measure and estimate it from finite windows.

### В описании транзиента

Не писать одновременно, что невозвратный finite trajectory имеет active dimension one и что active dimension является размерностью support инвариантной occupation measure. Следует выбрать одну терминологию.

Рекомендуемый вариант:

> A deterministic transient traces an approximately one-dimensional finite-record curve, but it does not sample the recurrent occupation measure required by our definition and estimator. We therefore treat the returned MG value as invalid rather than assign the transient a regime active dimension of one.

### В stochastic regime

Не писать, что active dimension «не существует» вообще. Корректнее:

> A stationary stochastic optimiser may have a well-defined occupation-measure dimension, but it includes directions excited by mini-batch noise and is not the structural dimension targeted here.

### В grokking section

Если короткое окно пересекает generalisation transition, писать:

> We track the windowed statistic $\widehat d_{\mathrm{MG}}(\gW)$ across the transition. Because these windows need not sample a stationary occupation measure, we interpret the change as a transition-correlated signal, not as a direct measurement of $d_{\mathrm{act}}(R)$.

### В обозначениях результатов

Использовать:

- $d_{\mathrm{act}}(R)$ — теоретическая размерность режима;
- $d_{\mathrm{act}}(R;r_1,r_2)$ — популяционная конечномасштабная размерность;
- $\widehat d_{\mathrm{MG}}(W)$ — вычисленная оконная оценка;
- $d_{\mathrm{PR}}(W)$ или $\PR(W)$ — effective rank.

Эти четыре обозначения не следует использовать взаимозаменяемо.

---

## 14. Итоговое определение в одном абзаце

Активные компоненты режима — локальные координаты support параметрической occupation measure этого режима. Их количество равно intrinsic dimension support. Конечное окно не определяет это многообразие: оно является выборкой, по которой размерность оценивается. MG оценивает локальный mass-scaling exponent на масштабе ближайших соседей и совпадает с active dimension только при наличии одного режима, корректного delay embedding, достаточного покрытия и устойчивого scaling range. Effective rank измеряет распределение дисперсии по линейным направлениям и остаётся вспомогательной отдельной величиной.
