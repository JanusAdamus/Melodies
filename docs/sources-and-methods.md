# Fuentes y trazabilidad de métodos

Cada cambio algorítmico del pipeline, mapeado a la fuente que lo respalda y al archivo y
función donde vive. También registra los métodos que se evaluaron y se descartaron, con la
medición que motivó descartarlos.

Complementa [`computational-complexity.md`](computational-complexity.md), que tiene el
análisis de costo, y [`resultados-comparacion-3000.md`](resultados-comparacion-3000.md),
que reporta la corrida sobre 3000 obras. Este documento es la trazabilidad, y su §4 es la
bibliografía única del proyecto.

Fecha: 2026-08-22. Ampliado el 2026-08-24 con la bibliografía de corpus, modelos,
evaluación y estadística (§1.1 y §4).

---

## 1. Alcance y método de búsqueda

**Esto no es una revisión sistemática PRISMA.** Es una búsqueda dirigida de métodos, con
un objetivo de ingeniería concreto: reducir el costo de pared del blocked Gibbs FFBS y del
Baum-Welch sobre un corpus grande de secuencias simbólicas. No hubo cribado por título y
resumen, ni evaluación de calidad de estudios, ni doble revisor. Se buscó hasta encontrar
los métodos canónicos de cada familia y se paró.

**Herramienta**: `WebSearch`. El skill `literature-review` recomienda `parallel-cli`, que
no está instalado en esta máquina y requiere `PARALLEL_API_KEY`, ausente. Tampoco hay
claves para generación de figuras, así que se omiten las esquemáticas.

**Consultas ejecutadas, textuales:**

1. `beam sampling infinite hidden Markov model HDP-HMM Van Gael adaptive truncation`
2. `stochastic variational inference hidden Markov model minibatch subsequences Foti Johnson Willsky`
3. `online EM hidden Markov model Cappe incremental forward-backward large datasets convergence`
4. `pruned sparse forward-backward algorithm HMM beam pruning speedup exact likelihood threshold`
5. `Pal Sutton McCallum sparse forward-backward minimum divergence beams CRF training 2006`
6. `Maddison Tarlow Minka A* sampling Gumbel-max trick categorical exact sample argmax`

**Verificación**: los detalles de Wang & Blunsom (2015) se confirmaron obteniendo el
resumen de arXiv directamente. El resto se cita desde los resultados de búsqueda; las URL
están en §4 para que cualquiera pueda comprobarlas.

### 1.1 Segunda ronda: bibliografía del reporte de resultados (2026-08-24)

Al redactar [`resultados-comparacion-3000.md`](resultados-comparacion-3000.md) se amplió
§4 con las referencias que respaldan el corpus, los cuatro modelos comparados, la
representación simbólica, el protocolo de evaluación y la inferencia estadística. Antes se
cubría solo la parte de inferencia y optimización del pipeline.

Mismo alcance dirigido y misma limitación de herramienta: `parallel-cli` no está instalado
y `PARALLEL_API_KEY`, `OPENROUTER_API_KEY`, `gget` y `pandoc` están ausentes en esta
máquina, así que se volvió a usar `WebSearch`. No hay PRISMA, cribado ni doble revisor.

**Consultas ejecutadas, textuales:**

7. `PDMX dataset Public Domain MusicXML large-scale symbolic music MuseScore paper`
8. `Begleiter El-Yaniv Yona variable order Markov models prediction JAIR 2004 PPM`
9. `Pearce IDyOM information dynamics of music variable order Markov 2005 thesis; Conklin Witten multiple viewpoint systems music prediction 1995`
10. `Demšar 2006 statistical comparisons of classifiers over multiple data sets Wilcoxon signed-rank; Holm 1979 sequentially rejective multiple test procedure`
11. `learning curves machine learning sample size power law Hestness 2017 deep learning scaling is predictable; Kaplan 2020 scaling laws neural language models`
12. `Huang Music Transformer relative attention 2019 ICLR symbolic music generation; Shaw self-attention relative position representations 2018`
13. `music structure segmentation evaluation boundary F-measure tolerance pairwise clustering V-measure Lukashevich MIREX mir_eval`

**Qué se verificó y qué no.** Las entradas 21, 22, 28, 34, 35, 36, 58 y 63–66 se
confirmaron contra los resultados de búsqueda, con DOI o URL en §4. Las referencias
canónicas de la formulación del transformer y de la estadística clásica (32, 33, 37–47,
48–57, 59–62, 67) se citan de memoria por ser estándar; **sus DOI no se han comprobado uno
a uno y conviene hacerlo con un gestor bibliográfico antes de entregar la tesis**.

---

## 2. Métodos implementados

| Método | Fuente | Dónde vive | Efecto medido |
|---|---|---|---|
| Reescalado forward-backward en dominio lineal | Rabiner (1989), §V.A | `src/models/inference.py:scaled_forward_log_likelihood`, `ffbs_sample_batch`; `Comparacion/classical_models.py:_expectation_batch` | 100× aislado; error 1.4e-08 |
| Muestreo categórico por Gumbel-max | Gumbel (1954); Maddison, Tarlow & Minka (2014) | `src/models/inference.py:ffbs_sample_batch`, bucle hacia atrás | 23× sobre esa operación |
| Agrupación por longitud con presupuesto de celdas | Práctica estándar de *bucketing*; sin fuente única | `src/models/utils.py:length_buckets` | 2.6× adicional |
| FFBS (forward filtering, backward sampling) | Chib (1996); Scott (2002) | Ya era el método del HDP-HMM; se conservó | — |
| Truncación de límite débil del HDP | Ishwaran & James (2001); Fox et al. (2011) | `Comparacion/config.py:hdp_truncation_level` | 1.2× bajando K de 40 a 24 |

**Nota sobre el reescalado.** Es la formulación de Rabiner: se mantiene `alpha` en espacio
lineal y se renormaliza en cada paso, acumulando el logaritmo de los factores de escala.
No es una aproximación — reproduce la log-verosimilitud del dominio logarítmico a 1.4e-08.
El código anterior usaba `logsumexp` sobre un bloque `K×K` por paso, que evalúa `exp` y
`log` sobre cada elemento.

**Nota sobre Gumbel-max.** `argmax(log p + g)` con `g ~ Gumbel(0,1)` i.i.d. es una muestra
categórica **exacta**, no una aproximación. Reemplaza a `rng.choice(..., p=...)`, que
revalida y acumula la distribución en cada llamada. Maddison et al. (2014) lo generalizan a
distribuciones continuas vía A*; aquí solo se usa el caso discreto clásico.

**Nota sobre la truncación.** El HDP-HMM usa la aproximación de límite débil: un
stick-breaking truncado en `K` estados. La guía estándar es fijar `K` cómodamente por
encima de la ocupación esperada. El piloto reporta **10-15 estados ocupados con `K`=40**,
así que 40 es holgura de sobra. Ver §3.5 para lo que eso compra realmente.

---

## 3. Métodos evaluados y descartados

Se registran con su medición, para que nadie los reintente sin datos.

### 3.1 Beam sampling para el HDP-HMM — no implementado

Van Gael et al. (2008) reemplazan la truncación fija por una variable auxiliar de *slice*
por paso temporal, que limita adaptativamente cuántos estados entran en la suma del forward
sin acotar el modelo a priori. Reportan que supera al muestreador de Gibbs y es más robusto.

**Por qué no ahora**: elimina el hiperparámetro `hdp_truncation_level`, lo cual es un
argumento metodológico legítimo para una tesis sobre modelos no paramétricos. Pero es una
reimplementación del muestreador, no una optimización, y el techo que promete está acotado
por lo que se midió en §3.5: bajar `K` de 40 a 16 da solo 1.5×. El *slice* adaptativo no
puede dar mucho más que eso en este corpus.

**Cuándo sí**: si la tesis quiere defender que la truncación no sesga los resultados.
Entonces el argumento es metodológico, no de velocidad.

### 3.2 Inferencia variacional estocástica (SVI) — no implementado

Foti, Xu, Laird & Fox (2014) y Johnson & Willsky (2014) desacoplan el costo por iteración
del tamaño del corpus: se actualiza con minilotes de subsecuencias en vez de barrer todo.
Foti et al. tratan el problema central —las subsecuencias no son independientes— escalando
los gradientes de subcadena para que sigan siendo insesgados, más un paso de mensajes
aproximado que acota el error por decaimiento de memoria de la cadena. Wang & Blunsom
(2015) hacen la versión colapsada. Ma, Foti & Fox (2017) la variante MCMC por gradiente
estocástico.

**Esta es la respuesta real al escalado**: con minilotes, 254 k partituras cuestan por
iteración lo mismo que 1 k.

**Por qué no ahora**: cambia el método de inferencia de Gibbs a variacional. Para una tesis,
eso no es una optimización, es otro experimento — cambia qué se está comparando y obliga a
rehacer la justificación metodológica. Es una decisión de dirección de tesis, no de
ingeniería.

### 3.3 EM online / incremental — descartado por medición

Cappé (2011) y Neal & Hinton (1998) actualizan parámetros por bloque en vez de por barrido
completo, convergiendo en muchas menos pasadas sobre los datos.

**Medición que lo descarta**: el Baum-Welch actual **ya para solo en la iteración 31 de
100** por la regla de tolerancia, y llega a 1e-3 de su mejor NLL de validación en la
iteración 24 (77 % de la corrida). El margen que queda es ~25 %, no el 10× que el EM
incremental promete cuando el EM por lotes corre cientos de iteraciones.

```
iter 20: val_nll 2.132193
iter 25: val_nll 2.124637   <- mejor
iter 31: val_nll 2.124581   <- para aquí
```

### 3.4 Forward-backward disperso / con poda — no implementado

Pal, Sutton & McCallum (2006) podan estados de baja probabilidad por paso mediante una
mezcla aproximante de deltas de Kronecker. Reportan reducir el entrenamiento de un CRF de
más de un día a seis horas —**4×**— sin pérdida de exactitud.

**Por qué no ahora**: es aproximado, y el espacio de estados aquí es chico (`K` = 12 a 48).
La poda rinde cuando hay miles de estados, como en los sistemas de habla que motivan el
método. Con `K`=24 y 10 estados ocupados, lo que se puede podar ya es poco. Además el
reescalado de §2 ya se llevó la ganancia grande.

### 3.5 Bajar la truncación del HDP — ganancia real pero modesta

Medido sobre 150 piezas reales, 40,860 tokens, 40 iteraciones de Gibbs:

| `K` | Tiempo | vs `K`=40 | Estados ocupados | log-verosimilitud final |
|---|---|---|---|---|
| 40 | 17.0 s | 1.0× | 10 | −83,392 |
| 32 | 14.9 s | 1.1× | 11 | −84,428 |
| 24 | 14.4 s | 1.2× | 10 | −84,186 |
| 20 | 12.5 s | 1.4× | 9 | −84,676 |
| 16 | 11.3 s | 1.5× | 10 | −83,594 |

La ocupación no depende de `K`: siempre 9-11 estados. La verosimilitud tampoco se degrada
de forma sistemática. Pero la ganancia es solo 1.2× a `K`=24 y 1.5× a `K`=16, porque tras
el reescalado el costo dejó de estar dominado por `K²`.

**Recomendación**: `hdp_truncation_level = 24`. Es holgura de más del doble sobre la
ocupación observada, gratis, y solo cambia un valor de configuración.

### 3.6 Recortar iteraciones de Gibbs — descartado por medición

**Resultado negativo importante.** La cadena **no** ha convergido a las 120 iteraciones:

```
iter  51: -0.912% vs mejor
iter  81: -0.495% vs mejor
iter 101: -0.310% vs mejor
iter 111: -0.051% vs mejor
dentro de 0.1% del mejor recién en la iteración 107
```

Cortar a 60 iteraciones costaría ~0.8 % de log-verosimilitud. `hdp_n_iters = 120` es
razonable, quizá corto. **No recortar.**

### 3.7 Procesar por lotes sin reescalar — descartado por medición

| Lote | Por secuencia | Por lotes | Ganancia |
|---|---|---|---|
| 512 | 37.12 µs/token | 37.21 µs/token | **1×** |
| 2048 | 36.55 µs/token | 37.84 µs/token | **1×** |

El lote por sí solo no da nada. Su papel es hacer el producto matricial lo bastante grande
para que BLAS lo paralelice, y eso solo aplica una vez que se quitó el `logsumexp`.

### 3.8 Muestreo vectorizado de filas Dirichlet — descartado por medición

Reemplazar el bucle de `rng.dirichlet` por una sola extracción gamma normalizada:
0.59 → 0.62 ms. **1×.** `rng.dirichlet` ya es eficiente.

### 3.9 Bootstrap vectorizado — descartado por irrelevante

5× medido, pero corre una sola vez al final y tarda 122 ms. No justifica el diff.

---

## 4. Referencias

**Formulación y algoritmos base**

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286. — Reescalado del forward-backward, §V.A.
2. Chib, S. (1996). Calculating posterior distributions and modal estimates in Markov mixture models. *Journal of Econometrics*, 75(1), 79–97. — FFBS.
3. Scott, S. L. (2002). Bayesian methods for hidden Markov models: Recursive computing in the 21st century. *Journal of the American Statistical Association*, 97(457), 337–351. — FFBS.

**Modelo no paramétrico**

4. Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. *JASA*, 101(476), 1566–1581.
5. Ishwaran, H., & James, L. F. (2001). Gibbs sampling methods for stick-breaking priors. *JASA*, 96(453), 161–173. — Truncación de límite débil.
6. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). A sticky HDP-HMM with application to speaker diarization. *Annals of Applied Statistics*, 5(2A), 1020–1056. https://arxiv.org/pdf/0905.2592

**Inferencia acelerada**

7. Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008). Beam sampling for the infinite hidden Markov model. *ICML 2008*. https://mlg.eng.cam.ac.uk/pub/pdf/VanSaaTehGha08.pdf · https://dl.acm.org/doi/10.1145/1390156.1390293
8. Foti, N. J., Xu, J., Laird, D., & Fox, E. B. (2014). Stochastic variational inference for hidden Markov models. *NeurIPS 2014*. https://arxiv.org/pdf/1411.1670
9. Johnson, M. J., & Willsky, A. S. (2014). Stochastic variational inference for Bayesian time series models. *ICML 2014*, PMLR 32. https://proceedings.mlr.press/v32/johnson14.html
10. Wang, P., & Blunsom, P. (2015). Stochastic collapsed variational inference for hidden Markov models. *NIPS Workshop on Time Series*. https://arxiv.org/pdf/1512.01665
11. Ma, Y.-A., Foti, N. J., & Fox, E. B. (2017). Stochastic gradient MCMC methods for hidden Markov models. *ICML 2017*, PMLR 70. https://proceedings.mlr.press/v70/ma17a/ma17a.pdf
12. Cappé, O. (2011). Online EM algorithm for hidden Markov models. *Journal of Computational and Graphical Statistics*, 20(3), 728–749. https://arxiv.org/pdf/0908.2359
13. Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In *Learning in Graphical Models*, 355–368.
14. Pal, C., Sutton, C., & McCallum, A. (2006). Sparse forward-backward using minimum divergence beams for fast training of conditional random fields. *ICASSP 2006*. https://people.cs.umass.edu/~mccallum/papers/sparse-fb.pdf

**Muestreo**

15. Gumbel, E. J. (1954). *Statistical theory of extreme values and some practical applications*. National Bureau of Standards.
16. Maddison, C. J., Tarlow, D., & Minka, T. (2014). A* sampling. *NeurIPS 2014*, 3086–3094.
17. Kool, W., van Hoof, H., & Welling, M. (2019). Stochastic beams and where to find them: the Gumbel-top-k trick. *ICML 2019*. https://arxiv.org/pdf/1903.06059

**Herramientas del pipeline**

18. Cuthbert, M. S., & Ariza, C. (2010). music21: A toolkit for computer-aided musicology and symbolic music data. *ISMIR 2010*. — Parseo de MusicXML, `src/data/parsing.py`.
19. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
20. Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. — `gammaln`, `minimize`, `wilcoxon`.

**Corpus y representación simbólica**

21. Long, P., Novack, Z., Berg-Kirkpatrick, T., & McAuley, J. (2025). PDMX: A large-scale public domain MusicXML dataset for symbolic music processing. *ICASSP 2025*. https://arxiv.org/abs/2409.10831 — Corpus de la corrida, `external/PDMX/mxl`.
22. Conklin, D., & Witten, I. H. (1995). Multiple viewpoint systems for music prediction. *Journal of New Music Research*, 24(1), 51–73. https://doi.org/10.1080/09298219508570672 — Justifica la representación compuesta `event_pitch_duration_metrical`, `next_token_experiment/data/tokenizer.py`.
23. Pearce, M. T. (2005). *The construction and evaluation of statistical models of melodic structure in music perception and composition*. Tesis doctoral, City University London. — IDyOM; marco de referencia del control VOMM, que **no** es una implementación de IDyOM.
24. Pearce, M. T., & Wiggins, G. A. (2012). Auditory expectation: the information dynamics of music perception and cognition. *Topics in Cognitive Science*, 4(4), 625–652. https://doi.org/10.1111/j.1756-8765.2012.01214.x

**Modelos de orden variable (control `vomm`)**

25. Cleary, J. G., & Witten, I. H. (1984). Data compression using adaptive coding and partial string matching. *IEEE Transactions on Communications*, 32(4), 396–402. — PPM, del que deriva el control.
26. Moffat, A. (1990). Implementing the PPM data compression scheme. *IEEE Transactions on Communications*, 38(11), 1917–1921. — PPMC y escape.
27. Witten, I. H., & Bell, T. C. (1991). The zero-frequency problem: estimating the probabilities of novel events in adaptive text compression. *IEEE Transactions on Information Theory*, 37(4), 1085–1094. — Suavizado y backoff, `Comparacion/vomm.py:backoff_strength`.
28. Begleiter, R., El-Yaniv, R., & Yona, G. (2004). On prediction using variable order Markov models. *JAIR*, 22, 385–421. https://doi.org/10.1613/jair.1491 — Comparación de VOMM sobre secuencias, música incluida.
29. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359–394. — Interpolación y el `alpha` aditivo del control.

**HMM finito**

30. Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics*, 41(1), 164–171. — Baum-Welch, `Comparacion/classical_models.py`.
31. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1), 1–38.

**Transformer (`next_token_experiment/models/small_transformer.py`)**

32. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
33. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. — Arquitectura decoder-only con atención causal.
34. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *NAACL-HLT 2018*, 464–468. https://aclanthology.org/N18-2074/
35. Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*, 21(140), 1–67. — Sesgo posicional relativo por cubetas logarítmicas, `RelativePositionBias`.
36. Huang, C.-Z. A., et al. (2019). Music Transformer: generating music with long-term structure. *ICLR 2019*. https://arxiv.org/abs/1809.04281 — Antecedente de atención relativa en música simbólica.
37. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. https://arxiv.org/abs/1607.06450 — `nn.LayerNorm`.
38. Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). https://arxiv.org/abs/1606.08415 — `nn.GELU`.
39. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929–1958. — `dropout=0.1`.
40. Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR 2015*. https://arxiv.org/abs/1412.6980
41. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. https://arxiv.org/abs/1711.05101 — `torch.optim.AdamW`, `weight_decay=0.01`.
42. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML 2013*. — Recorte de norma del gradiente, `grad_clip_norm=1.0`.
43. Press, O., & Wolf, L. (2017). Using the output embedding to improve language models. *EACL 2017*, 157–163. https://arxiv.org/abs/1608.05859 — `tie_input_output_embeddings=True`.
44. Inan, H., Khosravi, K., & Socher, R. (2017). Tying word vectors and word classifiers: a loss framework for language modeling. *ICLR 2017*. https://arxiv.org/abs/1611.01462
45. Micikevicius, P., et al. (2018). Mixed precision training. *ICLR 2018*. https://arxiv.org/abs/1710.03740 — `torch.autocast` y `GradScaler`.
46. Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade*, 55–69. — `early_stopping_patience=5`.
47. Paszke, A., et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS 2019*. https://arxiv.org/abs/1912.01703

**Evaluación predictiva y curvas de aprendizaje**

48. Jelinek, F., Mercer, R. L., Bahl, L. R., & Baker, J. K. (1977). Perplexity — a measure of the difficulty of speech recognition tasks. *Journal of the Acoustical Society of America*, 62(S1), S63. — Definición de perplejidad.
49. Cortes, C., Jackel, L. D., Solla, S. A., Vapnik, V., & Denker, J. S. (1994). Learning curves: asymptotic values and rate of convergence. *NIPS 1993*, 327–334.
50. Perlich, C., Provost, F., & Simonoff, J. S. (2003). Tree induction vs. logistic regression: a learning-curve analysis. *JMLR*, 4, 211–255. — Metodología de curvas por fracción de datos.
51. Hestness, J., et al. (2017). Deep learning scaling is predictable, empirically. https://arxiv.org/abs/1712.00409
52. Kaplan, J., et al. (2020). Scaling laws for neural language models. https://arxiv.org/abs/2001.08361

**Inferencia estadística (`Comparacion/statistics.py`)**

53. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. — `scipy.stats.wilcoxon`, pareado por obra.
54. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. — Corrección sobre las 6 comparaciones por pares.
55. Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*, 7(1), 1–26. — `bootstrap_samples=10000`, `bootstrap_seed=17`.
56. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
57. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
58. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30. https://jmlr.org/papers/v7/demsar06a.html — Recomienda Wilcoxon pareado para comparar dos modelos.

**Métricas estructurales (`Comparacion/structural_metrics.py`, definidas y probadas; sin datos de entrada en la corrida actual)**

59. Rand, W. M. (1971). Objective criteria for the evaluation of clustering methods. *JASA*, 66(336), 846–850.
60. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218. — Índice de Rand ajustado, `adjusted_rand_index`.
61. Strehl, A., & Ghosh, J. (2002). Cluster ensembles: a knowledge reuse framework for combining multiple partitions. *JMLR*, 3, 583–617. — NMI.
62. Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance. *JMLR*, 11, 2837–2854. — Normalización de la NMI, `normalized_mutual_information`.
63. Turnbull, D., Lanckriet, G., Pampalk, E., & Goto, M. (2007). A supervised approach for detecting boundaries in music using difference features and boosting. *ISMIR 2007*. — F1 de fronteras con tolerancia, `boundary_f1`.
64. Lukashevich, H. (2008). Towards quantitative measures of evaluating song segmentation. *ISMIR 2008*, 375–380.
65. Raffel, C., et al. (2014). mir_eval: a transparent implementation of common MIR metrics. *ISMIR 2014*. — Convenciones de evaluación de segmentación.
66. Nieto, O., Mysore, G. J., Wang, C.-i., Smith, J. B. L., Schlüter, J., Grill, T., & McFee, B. (2020). Audio-based music structure analysis: current trends, open challenges, and applications. *TISMIR*, 3(1), 246–263.

**Decisión multiobjetivo**

67. Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Kluwer. — Definición de no dominancia usada en `pareto_summary.json`.

---

## 5. Verdicto

De la búsqueda salieron cinco familias de métodos. Dos ya estaban implícitas en el código
(FFBS, truncación de límite débil), una se implementó (reescalado, más Gumbel-max que es
independiente de la literatura de HMM), y dos quedan disponibles pero no se tocan:

- **Beam sampling** es un argumento metodológico, no de velocidad: lo que ahorraría está
  acotado por el 1.5× medido en §3.5.
- **SVI** es la única vía real para el corpus completo, y es un cambio de método de
  inferencia. Es una decisión de tesis.

Lo único gratis que queda es **`hdp_truncation_level` de 40 a 24**: 1.2×, un valor de
configuración, con la ocupación medida (9-11 estados) como justificación.

Y dos resultados negativos que vale la pena tener por escrito: **las iteraciones de Gibbs
no se pueden recortar** (la cadena sigue mejorando en la 107 de 120), y **el EM ya para
solo** en la 31 de 100.
