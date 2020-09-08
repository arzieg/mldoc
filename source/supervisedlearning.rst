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
*Quelle: Andrew Ng, Neural Networks and Deep Learning, Coursera, 2020*

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
"Gelernter" Output     :math:`\hat y = \sigma(w^Tx+b)`
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

**Loss Funktion**

Die Loss Funktion berechnet den "Abstand" zwischen den gelernten Outputwert :math:`\hat y^{(i)}` und dem Traininglabel
:math:`y^{(i)}`. Diesen gilt es anhand der Loss Funktion zu minimieren.

:math:`L(\hat y^{(i)},y^{(i)}) = 1/2 (\hat y^{(i)} - y^{(i)})^2`

:math:`L(\hat y^{(i)},y^{(i)}) = -(y^{(i)}log(\hat y^{(i)})+(1-y^{(i)})log(1-\hat y^{(i)})`

* wenn :math:`y^{(i)} = 1 \;:\;L=-log(\hat y^{(i)})` wobei :math:`log(\hat y^{(i)}) \; und \; \hat y^{(i)}` nahe bei 1 liegen sollen.

* wenn :math:`y^{(i)} = 0 \;:\;L=-log(1-\hat y^{(i)})` wobei :math:`log(1-\hat y^{(i)}) \; und \; \hat y^{(i)}` nahe bei 0 liegen sollen.

Die Kostenfunktion aller Trainingselemente ist der Durchschnitt der Loss Funktion eines jeden Trainingswertes. Ziel ist es diese Funktion J(w,b) zu
minimieren:

:math:`J(w,b)=\frac{1}{m} \sum^{m}_{i=1} L(\hat y^{(i)},y^{(i)})=
-\frac{1}{m} \sum^{m}_{i=1}[(y^{(i)}log(\hat y^{(i)})+(1-y^{(i)})log(1-\hat y^{(i)})]`


**Beispiel: Foreward and Backward - Propagation in a Network**

Die Lösung eines LR Problems kann über das Gradient Descence Verfahren angegangen werden. Hierbei wird ein lokales Minimum
gesucht durch Änderung der unabhängigen Variablen in einer Kostenfunktion.

Bsp.:
Gegeben sei die Kostenfunktion J(a,b,c)=3(a+bc). u=bc, v=a+u und J=3v

Als Berechnungsgraph kann man das wie folgt aufschreiben:

.. graphviz::

    digraph {
        rankdir=LR;
        "a=5" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "b=3" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "c=2" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "u=3*2=6" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "v=a+u" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "J=3v" [shape=circle  , regular=1,style=filled,fillcolor=white   ] ;
        "a=5" -> "v=a+u";
        "b=3","c=2" -> "u=3*2=6";
        "u=3*2=6" -> "v=a+u";
        "v=a+u" -> "J=3v";
        { rank=same; "a=5", "b=3", "c=2" }
    }

Es wird nun die Änderung einer Variable in Abhängigkeit einer anderen Variable bestimmt, d.h. der
Berechnungsgraph wird von rechts nach links berechnet. Im Beispiel: Änderungsrate von J, wenn v sich marginal ändert?
Mathematisch :math:`\frac{dJ}{dv}`. In diesem Beispiel ist v=11 und J=33. Wenn sich v um 0.001 ändert, ändert sich
J um 3 * 0.001 auf 33.003, d.h. :math:`\frac{dJ}{dv}=3`.
J ist von v abhängig, während v von a und u abhängig ist. Wie ändert sich J, wenn a sich ändert :math:`\frac{dJ}{da}`?
a=5, wenn a=5.001, dann ist v=11.001 und J=33.003. Somit ist :math:`\frac{dJ}{da}=3`.
Oder in anderen Worten: Wenn sich a ändert, ändert sich v, ändert sich J. Das ist die Chain Rule:
:math:`\frac{dJ}{da}=\frac{dJ}{dv}\frac{dv}{da}`. Am Beispiel: a=5.001 => v=11.001 dv/da=1 und J=33.003 bzw. dJ/dv=3
und somit dJ/da=1 x 3 = 3.

Analog bei :math:`\frac{dJ}{du}`. u=6, wenn u=6.001, dann ist v=11.001 und J=33.003.
:math:`\frac{dJ}{du}=\frac{dJ}{dv}\frac{dv}{du}=3 * 1 = 3`

Für :math:`\frac{dJ}{db}` gilt: b=3, b=3.001, u=6.002, v=11.002, J=33.006 oder
:math:`\frac{dJ}{db}=\frac{dJ}{dv}\frac{dv}{du}\frac{du}{db}=3*1*2=6`

Für :math:`\frac{dJ}{dc}=\frac{dJ}{dv}\frac{dv}{du}\frac{du}{dc}=3*1*3=9`


**Foreward and Backward - Propagation im LR Network**

Im LR Netzwerk haben wir

* die lineare Funktion: :math:`z=w^Tx+b`
* den gelernten Output: :math:`\hat y=a=\sigma(z)`
* die Kostenfunktion: :math:`L(a,y) = -(y log(a) + (1-y)(log(1-a))`

Als Berechnungsgraph:

:math:`Input: \; x_1,w_1,x_2,w_2,b \rightarrow z=w_1x_1+w_2x_2+b \rightarrow \hat y=a=\sigma(z) \rightarrow L(a,y)`

Für die Backpropagation gilt dann:
:math:`\frac{dL}{dz}=\frac{dL}{da}\frac{da}{dz}`

Schritt 1: :math:`\frac{dL}{da}`

:math:`L= -(y log(a) + (1-y)log(1-a))`

:math:`\frac{dL}{da}=-y \times \frac{1}{a} - (1-y) \times \frac{1}{1-a}\times -1`

Achtung: -1 am Ende, da für f' von ln(1-a) die Chain-Rule gilt!

:math:`\frac{dL}{da}=\frac{-y}{a} + \frac{1-y}{1-a}`

:math:`\frac{dL}{da}=\frac{-y\times(1-a)}{a\times(1-a)} + \frac{a\times(1-y)}{a\times(1-a)}`

:math:`\frac{dL}{da}=\frac{-y+ay+a-ay}{a(1-a)}`

:math:`\frac{dL}{da}=\frac{a-y}{a(1-a)}`

Schritt 2: :math:`\frac{da}{dz}`

:math:`\frac{da}{dz}=\frac{d}{dz}\sigma(z)=\sigma(z)\times(1-\sigma(z))`

Wir haben :math:`\sigma(z)=a` definiert. So kann die Formel vereinfacht werden zu

:math:`\frac{da}{dz}=a(1-a)`

    *Exkurs: Ableitung:*

    :math:`\frac{d\sigma(z)}{dz}=\frac{d}{dz}\frac{1}{1+e^{-z}}`

    Hier ist wieder die Chain Rule anzuwenden. Wir definieren :math:`u=1+e^{-z}`. Die Sigmoid Funktion kann nun
    als :math:`\sigma(u)=\frac{1}{u}` geschrieben werden.

    :math:`\frac{d\sigma(z)}{dz}=\frac{d\sigma(u)}{du}\frac{u}{dz}`

    *Schritt 1:*

    :math:`\frac{d\sigma(u)}{du}=\frac{d}{du}\frac{1}{u}=-\frac{1}{u^2}=-\frac{1}{(1+e^{-z})^2}`

    *Schritt 2:*

    :math:`\frac{du}{dz}=\frac{d}{dz}(1+e^{-z})=-e^{-z}`

    *Schritt 3 zusammenbringen:*

    :math:`\frac{d\sigma(z)}{dz}=\frac{d\sigma(u)}{du}\frac{u}{dz}=-\frac{1}{(1+e^{-z})^2} \times (-e^{-z})`

    *Schritt 4 vereinfachen:*

    Es ist :math:`\sigma(z)=\sigma=\frac{1}{(1+e^{-z})}`, daher gilt:

    :math:`\frac{1}{(1+e^{-z})^2}=\sigma^2`

    Für :math:`e^{-z}` gilt:

    :math:`\sigma=\frac{1}{(1+e^{-z})} \Rightarrow \sigma(1+e^{-z})=1 \Rightarrow 1+e^{-z} = \frac{1}{\sigma}
    \Rightarrow e^{-z} = \frac{1}{\sigma}-1=\frac{1-\sigma}{\sigma}`

    Damit kann der Term vereinfacht werden zu:

    :math:`\frac{d\sigma(z)}{dz}=\frac{1}{(1+e^{-z})^2} \times e^{-z} = \sigma^2 \times \frac{1-\sigma}{\sigma}=\sigma \times
    (1-\sigma)`




Schritt 3: :math:`\frac{dL}{dz}`

:math:`\frac{dL}{dz}=\frac{dL}{da}\times\frac{da}{dz}`

:math:`\frac{dL}{dz} = \frac{a-y}{a(1-a)} \times a(1-a) = a-y`


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

