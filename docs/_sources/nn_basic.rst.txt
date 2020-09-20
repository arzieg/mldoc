.. _nl_basic:

###################
Neuronale Netzwerke
###################
*Quelle: Andrew Ng, Neural Networks and Deep Learning, Coursera, 2020*

**Notation:**
    Wichtig ist die Unterscheidung bei den hochgestellten [] - Klammern vs. den () - Klammern.
    Die [] Klammern beziehen sich auf den Layer innerhalb eine NN. Die ()-Klammern beziehen sich auf
    ein Element z.B. aus einem Trainingsset.

NN Logistic Regression
***********************
Andrew Ng. erarbeitet die Einführung in ein NN am Beispiel des Logistic Regression Algorithmus.

Bei einem NN gibt es Inputwerte (x1, x2, ...), Hidden-Layer (hier nur a1) und das Ergebnis a2 (bzw. y als Extra eingezeichnet).

.. graphviz::

    digraph {
        rankdir=LR;
        "x1" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "x2" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "x3" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "a11" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true  ] ;
        "a12" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "a13" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "a2" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "y" [shape=circle  , regular=1,style=filled,fillcolor=white, width=.5, fixedsize=true   ] ;
        "x1" -> "a11";
        "x1" -> "a12";
        "x1" -> "a13";
        "x2" -> "a11";
        "x2" -> "a12";
        "x2" -> "a13";
        "x3" -> "a11";
        "x3" -> "a12";
        "x3" -> "a13";
        "a11" -> "a2";
        "a12" -> "a2";
        "a13" -> "a2";
        "a2" -> "y";
        { rank=same; "x1", "x2", "x3" }
    }

In jedem Hidden Layer wird z und a (Aktvierungsfunktion) berechnet wie im Modell des Logistic Regression.
Der Outputwert geht dann, je nach Definition als Inputwert in den Hidden Layer 2 usw.

.. graphviz::

    digraph NN {

        node [shape=record];
        rankdir=LR;
        "x" [shape=circle  , regular=1,style=filled,fillcolor=white, label="x" ] ;
        "W" [shape=circle  , regular=1,style=filled,fillcolor=white, label="W" ] ;
        "b" [shape=circle  , regular=1,style=filled,fillcolor=white, label="b" ] ;
        "z1" [shape=record  , regular=1,style=filled,fillcolor=white, width=1.75, height=0.5, fixedsize=true, label="z[1]=W[1]x+b[1]"] ;
        "a1" [shape=record  , regular=1,style=filled,fillcolor=white, width=1.75, height=0.5, fixedsize=true,label="a[1]=sigma(z[1])"] ;
        "z2" [shape=record  , regular=1,style=filled,fillcolor=white, width=1.75, height=0.5, fixedsize=true,label="z[2]=W[2]a[1]+b[2]"   ] ;
        "a2" [shape=record  , regular=1,style=filled,fillcolor=white, width=1.75, height=0.5, fixedsize=true,label="a[2]=sigma(z[2])"] ;
        "L" [shape=record  , regular=1,style=filled,fillcolor=white, width=1.75, height=0.5, fixedsize=true,label="L(a[2],y)"  ] ;

        "x" -> "z1";
        "W" -> "z1";
        "b" -> "z1";
        "z1" -> "a1" -> "z2" -> "a2" -> "L"

        { rank=same; "x", "W", "b" }

        subgraph cluster_R1 {
          label="Hidden Layer 1"
          z1 ;
          a1 ;
        }

        subgraph cluster_R2 {
          label="Output"
          z2 ;
          a2 ;
        }

    }

oder für alle m Trainingswerte (Achtung: []=Hidden Layer, ()=Trainingsexample):

    for i = 1 to m:
        | :math:`z^{[1](i)}=W^{[1]}x^{(i)}+b^{[1]}`
        | :math:`a^{[1](i)}=\sigma(z^{[1](i)})`
        | :math:`z^{[2](i)}=W^{[2]}a^{[1](i)}+b^{[2]}`
        | :math:`a^{[2](i)}=\sigma(z^{[2](i)})`

bzw. als Vektorschreibweise:
    | :math:`Z^{[i]}=W^{[1]}X+b^{[i]}`
    | :math:`A^{[i]}=\sigma(Z^{[i]})`
    | :math:`Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}`
    | :math:`A^{[2]}=\sigma(Z^{[2]})`

Aktivierungsfunktionen:
=======================
Bisher wurde die Sigmoid-Aktivierungsfunktion verwendet. Es macht bei NN Sinn andere Aktivierungsfunktionen zu
verwenden. Empfohlen wird alle gängigen Aktivierungsfunktion beim eigenen NN Modell auszuprobieren, welche besser
funktioniert.

Gängige Aktivierungsfunktion:

**Sigmoid** Funktion - wird häufig nicht mehr verwendet in NN (außer im Outputlayer).

| :math:`g(z)=a=\frac{1}{1+e^{-z}}`
| :math:`\frac{d}{dz}g(z)=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})=g(z)(1-g(z))=a-(1-a)`

.. _nn_001_sigmoid_graph:

.. figure:: pic/nn_001_sigmoid_graph.png
    :scale: 50%
    :alt: Sigmoid Function
    :align: center

    :numref:`Sigmoid Function (Abb. %s)  <nn_001_sigmoid_graph>`

**tanh(z)** - Funktion (arbeitet i.d.R. besser als die Sigmoid-Funktion, da er durch den Nullpunkt geht im Gegensatz zu
der Sigmoid-Funktion, die den Nullwert bei 0.5 hat). Für den Outputlayer ist es sinnvoll, die Sidmoid-Funktion zu
nutzen, für die Hidden-Layer die tanh-Aktivierungsfunktion.

| :math:`g(z)=a=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}`
| :math:`\frac{d}{dz}g(z)=1-(tanh(z))^2=1-a^2`

.. _nn_002_tanh_graph:

.. figure:: pic/nn_002_tanh_graph.png
    :scale: 50%
    :alt: tanh Function
    :align: center

    :numref:`tanh Function (Abb. %s)  <nn_002_tanh_graph>`

**RelU (Rectified Linear Unit)** Funktion. Die Ableitung ist 1, wenn z>0 bzw. 0 wenn z<0.
Vorteile: Einfache Anwendung und leichte Berechnung (im Gegensatz zu Sigmoid/tanh)
Nachteil: Beim "Lernen" kann es vorkommen, dass der Wert bei 0 verharrt und damit kein "Lernen" stattfindet.

| :math:`g(z)=a=max(0,z)`
| :math:`\begin{equation}
            \frac{d}{dz}g(z)= \begin{cases}
                        0 \text{ wenn z < 0 } \\
                        1 \text{ wenn z > 0 } \\
                        undef \text{ wenn z = 0 }
                     \end{cases}
         \end{equation}`

.. _nn_003_relu_graph:

.. figure:: pic/nn_003_relu_graph.png
    :scale: 50%
    :alt: RelU Function
    :align: center

    :numref:`RelU Function (Abb. %s)  <nn_003_relu_graph>`

**leaky RelU** - wenn z<0, dann ist die Ableitung negativ und nicht 0 wie bei RelU. Dies hilft beim
Gradient Descent Verfahren (hebt den Nachteil von RelU auf).

| :math:`a=max(0.01z,z)`
| :math:`\begin{equation}
            \frac{d}{dz}g(z)= \begin{cases}
                        0.01 \text{ wenn z < 0 } \\
                        1 \text{ wenn z } \geq \; 0
                     \end{cases}
         \end{equation}`

.. _nn_004_lrelu_graph:

.. figure:: pic/nn_004_lrelu_graph.png
    :scale: 50%
    :alt: Leaky RelU Function
    :align: center

    :numref:`Leaky RelU Function (Abb. %s)  <nn_004_lrelu_graph>`

Forward Propagation
===================
Analog zu dem Logistic Regression Algorithmus berechnet sich die Forward Propagation für ein NN
wie folgt:

    | :math:`Z^{[1]}=W^{[1]}X+b^{[1]}`
    | :math:`A^{[1]}=g^{[1]}(Z^{[1]})`
    | :math:`Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}`
    | :math:`A^{[2]}=g^{[2]}(Z^{[2]})=\sigma(Z^{[2]})`

Backward Propagation
=====================

    | :math:`dZ^{[2]}=A^{[2]}-Y`
    | :math:`dW^{[2]}=\frac{1}{m}dZ^{[2]}A^{[1]T}`
    | :math:`db^{[2]}=\frac{1}{m}np.sum(dZ^{[2]}, axis=1, keepdims=true)`
    | :math:`dZ^{[1]}=dW^{[2]T}dZ^{[2]}*g^{[1]'}(Z^{[1]})`
    | :math:`dW^{[1]}=\frac{1}{m}dZ^{[1]}X^{T}`
    | :math:`db^{[1]}=\frac{1}{m}np.sum(dZ^{[1]}, axis=1, keepdims=true)`

Initialisierung dues NN
=======================
Bei der Initialisierung des NN ist die Matrix W und der Vektor b mit Werten vorzubelegen. Man kann zeigen, dass
b ein Null-Vektor sein kann, W sollte aber mit Zufallszahlen initialisiert werden. Wenn W ebenfalls eine Null-Matrix
ist, würde bei der Backpropagation symmetrische Werte errechnet werden zw. den Hidden-Layern (so dass man eigentlich
das Modell auf ein Hidden Layer reduzieren kann.)
Bei der Initialisierung ist weiterhin darauf zu achten, dass die W-Matrix mit kleinen Werten initialisiert wird.
Dies liegt daran, dass die Aktivierungsfunktion bei > 1 oder < 1 bereits auf einen Wert limitiert wird (0 oder 1).
Um hier eine hohe Variabilität zu haben, sollten die Werte für W -1 < w < 1 sein.

| :math:`w^{[1]}=np.random.rand(2,2)*0.01` (anstelle von 0.01 kann auch ein anderer kleiner Wert genommen werden).
| :math:`w^{[2]}=np.random.rand(1,2)*0.01` (anstelle von 0.01 kann auch ein anderer kleiner Wert genommen werden).



