.. _notation:

###########
Notationen
###########

Neuronale Netzwerke
*******************

Ein hochgestelltes (i) beschreibt den i\ :sup:`th` Trainingssatz.
Ein hochgestelltes [l] beschreibt den l\ :sup:`th` layer.

Dimensionen
===========

+-------------------------------------+----------------------------------------------------+
| m                                   | Anzahl Beispiele im Datensatz                      |
+-------------------------------------+----------------------------------------------------+
| n\ :sub:`x`                         | Input-Werte                                        |
+-------------------------------------+----------------------------------------------------+
| n\ :sub:`y`                         | Ergebnisse                                         |
+-------------------------------------+----------------------------------------------------+
| :math:`n ^{\text{[l]}}_{\text{h}}`  | Anzahl der Hidden Einheiten im l-ten Layer         |
+-------------------------------------+----------------------------------------------------+
| :math:`a ^{\text{[l]}}`             | Aktivierungsfunktionen im Layer l                  |
+-------------------------------------+----------------------------------------------------+
| :math:`w ^{\text{[l]}}`             | Gewichtungen im Layer l                            |
+-------------------------------------+----------------------------------------------------+
| In einer for - Schleife ist auch folgende Schreibweise möglich                           |
| :math:`n_x = n ^{\text{[0]}}_{\text{h}}` und                                             |
| :math:`n_y = n ^{\text{[number of layers + 1]}}_{\text{h}}`                              |
+-------------------------------------+----------------------------------------------------+
| L                                   | Anzahl Layer im Netzwerk                           |
+-------------------------------------+----------------------------------------------------+
| log                                 | im Machine Learning Kontext ist damit der          |
|                                     | natürliche Logarithmus ln gemeint.                 |
+-------------------------------------+----------------------------------------------------+


Zurück zu :ref:`notation`

Objects
=======

| :math:`X \in \mathbb{R} ^{n_x \; \mathsf x \;m}`  :  Inputmatrix X
| :math:`x^{(i)} \in \mathbb{R} ^{n_x}`             :  :math:`i^{th}` Datensatz als Spaltenvektor
| :math:`Y \in \mathbb{R} ^{n_y \; \mathsf x \;m}`  :  Labelmatrix Y (Ergebnismatrix)
| :math:`y^{(i)} \in \mathbb{R} ^{n_y}`             :  :math:`i^{th}` Labelsatz als Spaltenvektor |
| :math:`W^{[l]}\in \mathbb{R} ^{Anzahl \; Einheiten \; im \; Folgelayer \; \mathsf x \; Anzahl \; Einheiten \; im \; Vorlayer}`
  : Matrix der Gewichtungen, [l] ist der Layer
| :math:`b^{[l]}\in \mathbb{R} ^{Anzahl \; Einheiten \; im \; Folgelayer}`  :  Bias-Vektor im [l]-ten Layer
| :math:`\hat y \in \mathbb{R} ^{n_y}`  :  ist der berechnete Ergebnisvektor. Es kann auch :math:`a^{[L]}` geschrieben
  werden, wobei L die Anzahl der Layers in einem Netzwerk darstellt.

Zurück zu :ref:`notation`

Beispiele für typische Forward Propagation Gleichungen
=======================================================

| :math:`a = g^{[l]}(W_x x^{(i)} + b_1) = g^{[l]}(z_1) \; wobei \; g^{[l]} \; die \; l^{[th]} \; Aktivierungsfunktion \;
   beschreibt.`
| :math:`\hat y^{(i)} = softmax (W_h h + b_2)`
| Allgemeine Aktivierungsfunktion: :math:`a ^{[l]}_{j}=g ^{[l]}(\sum _k w ^{[l]}_{jk} a^{[l-1]}_k + b^{[l]}_j) =
  g^{[l]}(z ^{[l]}_j)`
| Kostenfunktion: :math:`J(x,W,b,y) \; oder \; J(\hat y,y)`

Zurück zu :ref:`notation`

Beispiele für Kostenfunktionen
===============================

| :math:`J_{CE}(\hat y,y) = - \sum ^{m}_{i=0} \; y ^{(i)} \; log \; \hat y^{(i)}`
| :math:`J_1(\hat y,y) = \sum ^{m}_{i=0} \; | y ^{(i)} \; - \; \hat y^{(i)}|`

Zurück zu :ref:`notation`

Matrixdimensionen
=================
Bei der Berechnung eines NN vektorisiert man die einzelnen Variablen im Modell. Beim Debuggen ist es hilfreich,
wenn die Dimensionen der Matritzen je Layer bekannt sind.

Es gilt:

+--------------------------------------------+---------------------------------------------------------------+
| **Vektor**                                 | **Matrixdimension je Layer**                                  |
+--------------------------------------------+---------------------------------------------------------------+
| Gewichtungsfaktor w                        | :math:`w ^{\text{[l]}} : (n^{\text{[l]}},n^{\text{[l-1]}})`   |
+--------------------------------------------+---------------------------------------------------------------+
| Änderung von Gewichtungsfaktor w           | :math:`dw ^{\text{[l]}} : (n^{\text{[l]}},n^{\text{[l-1]}})`  |
+--------------------------------------------+---------------------------------------------------------------+
| Bias b                                     | :math:`b^{\text{[l]}} : (b^{\text{[l]}},1)`                   |
+--------------------------------------------+---------------------------------------------------------------+
| Änderung von bias b                        | :math:`db^{\text{[l]}} : (b^{\text{[l]}},1)`                  |
+--------------------------------------------+---------------------------------------------------------------+
| Aktivierungsfkt. A und Funktion Z          | :math:`Z^{\text{[l]}}, A^{\text{[l]}} : (n^{\text{[l]}},m)`   |
+--------------------------------------------+---------------------------------------------------------------+
| Änderung Aktivierungsfkt. A und Funktion Z | :math:`dZ^{\text{[l]}}, dA^{\text{[l]}} : (n^{\text{[l]}},m)` |
+--------------------------------------------+---------------------------------------------------------------+