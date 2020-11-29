.. _dlanpassungen:

#################################
Deep Learning Modellanpassungen
#################################

Bias / Varianz
**************

**Trade off von Bias und Varianz**

"In statistics and machine learning, the bias–variance tradeoff is the property of a model that the variance of the
parameter estimates across samples can be reduced by increasing the bias in the estimated parameters.
The bias–variance dilemma or bias–variance problem is the conflict in trying to simultaneously minimize these two
sources of error that prevent supervised learning algorithms from generalizing beyond their training set:

* The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
* The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).

This trade-off is universal: It has been shown that a model that is asymptotically unbiased must have unbounded variance.
The bias–variance decomposition is a way of analyzing a learning algorithm's expected generalization error with respect to a particular problem as a sum of three terms, the bias, variance, and a quantity called the irreducible error, resulting from noise in the problem itself."
[ `Bias Variance TradeOff`_ ]

.. _Bias Variance TradeOff: https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff

Beispiel:
*Quelle: Andrew Ng, Coursera, Improving Deep Learning*

.. _dl_01_Bias_Varianz:

.. figure:: pic/dl_01_Bias_Varianz.png
    :scale: 100%
    :alt: Bias / Varianz
    :align: center

    :numref:`Bias Varianz (Abb. %s)  <dl_01_Bias_Varianz>`

* links: Beispiel für ein hohes Bias. Hohe Fehlerrate bei den vorhergesagten Werten (Underfitting)
* rechts: Beispiel für eine hohe Varianz. Das Modell auf Basis der Trainingsdaten ist zu spezifisch, so dass das Modell
  schlecht an den realen Daten skalieren kann. (Overfitting)
* Mitte: optimaler Beziehung zw. Fehlern auf der einen Seite und Komplexität des Modells auf der anderen Seite.


Zurück zu :ref:`dlanpassungen`

