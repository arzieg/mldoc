.. _sl:

###################
Supervised Learning
###################

Beim SL sind neben den Beobachtungen (Features) auch die daraus resultierenden Ergebnisse bekannt.

Wichtige „learning“ Algorithmen sind:

* k-Nearest Neighbors
* Linear Regression
* `Binary Classification`_
* `Logistic Regression`_
* Support Vector Maschines (SVMs)
* Decision Trees and Random Forests
* Neuronal networks
* Linear Regression (LR)
* Der Klassiker: supervised_learning_linear_regression


Binary Classification
**********************

Binary Classification ist die Aufgabe die vorhandenen Elemente in zwei Gruppen von Klassen einzuteilen. Beispiele:

* Medizin: Krebs ja / nein
* Qualitätssicherung: i.O / nich i.O.
* Bildsuche: Katzenbild ja/nein

**Beispiel bei einer Bildklassifizierung:**

Ziel ist die Entwicklung eines Klassifizierungsmodelles zur Bilderkennung, bspw. Katze ja/nein.
Als Eingangsparameter wird das Bild verwendet. Das Ergebnis ist ein Labelvektor y (1=Katze, 0=keine Katze).
Das Bild wird zu einem Feature Vector X umgewandelt, Beispiel: Bild hat die Farben (RGB): rot x grün x blau.
Pixelgröße 64 x 64. Um daraus einen Feature Vektor X abzuleiten, wird die Pixelintensität (Wert zw. 0-255) als Vektor
dargestellt mit der Dimension 1 Spalte x (64 x 64 x 3 (RGB)) = 12288 Zeilen

:math:`x = \begin{pmatrix} \color{Red}{255 \\ 12 \\ 128 \\ \vdots \\ }
\color{Green}{86 \\ 172 \\ 255 \\ \vdots \\ }
\color{Blue}{88 \\ 156 \\ 192 \\ \vdots}  \end{pmatrix}`

Zurück zu :ref:`sl`


Logistic Regression
********************

Bei der LR ist das Ergebnis entweder 1 oder 0. Ziel des LR ist die Minimierung des Fehlers zw. den Vorhersagedaten
und Trainingsdaten.

Die Parameter in LR sind:

===================    =====================================================
Bedeutung              Parameter oder Funktion
===================    =====================================================
Feature Inputvektor    x
Training Label         y :math:`\in \; 0,1`
Gewichtung             w
Bias (Threshold)       b
Output                 :math:`\hat y = \sigma(w^Tx+b)`
Sigmoid Funktion       :math:`\sigma(w^Tx+b)=\sigma(z)=\frac{1}{1+e^{-z}}`
===================    =====================================================

.. _001_sl_sigmoid_function:

.. figure:: pic/001_sl_sigmoid_function
    :scale: 100%
    :alt: Sigmoid Function
    :align: center

    :numref:`Sigmoid Function (Abb. %s)  <001_sl_sigmoid_function>`

:math:`(w^Tx+b)` ist die lineare Funktion von (ax+b). Da nur nach der Wahrscheinlichkeit zwischen [0,1] bewertet wird,
wird die Sigmoid Funktion verwendet, die die Werte auf einen Wertebereich zw. [0,1] normiert. Sie hat folgende
Eigenschaften:

* wenn z sehr gross ist, dann ist :math:`\sigma(z) = 1`
* wenn z sehr klein ist oder kleiner Null, dann ist :math:`\sigma(z)=0`
* wenn z = 0 ist, dann ist :math:`\sigma(z) = 0.5`


Zurück zu :ref:`sl`


Decision Trees
**************
Bei einem Entscheidungsbaum werden die Daten in verschiedene Kategorien unterteilt. Dabei wird je Iteration ein
neues Knotenpaar erzeugt, bis alle Traings-Daten einem Knoten zugeordnet sind. Aufgrund des Algorithmus neigt
dieser zum „overfitting“, d.h. es wird ein Entscheidungsbaum in der Form aufgebaut, so dass alle Trainingsdaten
im Extremfall einem Knoten zugeordnet sind. Die Testdaten müssen dann nicht zwingend genausogut in diese Kategorien
fallen! In sklearn gibt es zwei Klassen:

    **DecisionTreeRegressor** und
    **DecisionTreeClassifier**.

DecisionTreeRegressor sind nicht in der Lage Vorhersagen außerhalb des Gültigkeitsbereichs der Trainingsdaten
zu machen!

**Wichtige Begriffe:**

    * root – Ursprungsknoten, dieser beinhaltet alle Testdaten
    * leaf – Endknoten (Blätter). Enthält der Leaf-Knoten alle den identischen Wert, wird auch von einem pure – leaf Knoten gesprochen.

In jedem Knoten  gibt es eine Testbedingung, die zum nächsten „Ast“ verzweigt.
Vermeidung von „Overfitting“ durch zwei Strategien:

    #. pre-pruning – Angabe der maximalen Ebenen eines Entscheidungsbaumes. In sklearn implementiert über

        * max_depth: maximale Anzahl der Ebenen
        * max_leaf_nodes:  maximale Anzahl der Leafs
        * min_samples_leaf: minimale Anzahl von Daten in einem Knoten, die vorhanden sein müssen.

    #. post-pruning/pruning – Die letzte Ebene wird eleminiert, um ein „overfitting“ zu vermeiden. In sklearn nicht implementiert.

feature importance: in sklearn wird beim Aufbau eines Entscheidungsbaums auch ein Array feature_importance mit Werten gefüllt. Diese geben an, welches Feature (Spalte) am Relevantesten für den Aufbau des Entscheidungsbaums ist. Die Summe alle feature_importances ist 1.

**Ziel des ML Algorithmus:**
Ziel ist der Aufbau eines Entscheidungsbaums, in der alle Daten nach einer Testentscheidung einem Knoten zugeordnet werden können.

**Vorteile von DT:**
* Ergebnisse sind leicht zu visualisieren und leicht verständlich für nicht Experten
* Daten müssen nicht erst in eine Standardnorm umgeformt werden.

**Nachteile von DT:**
* Tendenz zum „Overfitting“. Die Trainingsdaten werden – ohne (pre-)pruning – zu 100% einem Knoten zugeordnet. Der Akzeptanztest für die Testdaten fällt in der Regel schlechter aus, daher gilt
* eine geringere Generalisierungsmöglichkeiten des Modells

Um die Nachteile auszugleichen, verwendet man in der Praxis eher mehrere Decision Trees (→ siehe Random Forest) an.

Zurück zu :ref:`sl`

