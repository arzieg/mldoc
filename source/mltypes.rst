.. _mltypes:

######################
Maschinelearning Typen
######################

Es gibt verschiedene Typen von ML – Systemen. Géron gliedert diese in

* Systeme die autonom oder unter menschlicher Anleitung trainiert werden:

    * supervised learning
    * unsupervised learning
    * semisupervised learning
    * reinforcement learning

* Systeme die online oder nur offline trainiert werden

    * online vs.
    * offline (batch) learning

* Systeme die neue Punkte mit vorhandenen vergleichen oder Muster erkennen:

    * instance-based vs.
    * model-based learning.

Supervised Learning
===================
Beim supervised learning beinhalten die Trainingsdaten auch die Lösung. Typische supervised learning Aufgaben sind:

* Klassifikation von Daten (ein Datum gehört zur Klasse A, B, C, …)
* Analyse von Daten in Abhängigkeit von mehreren unabhängigen Daten (Regressionsanalysen).

Wichtige „learning“ Algorithmen sind:

* k-Nearest Neighbors
* Linear Regression
* Logistic Regression
* Support Vector Maschines (SVMs)
* Decision Trees and Random Forests
* Neuronal networks

Zurück zu :ref:`mltypes`

Unsupervised Learning
=====================
Beim unsupervised learning sind die Trainingsdaten ohne Label (Wert). Das ML System „lernt“ ohne menschliche Anleitung.
Beispiel ist die Ermittlung von Clustern innerhalb einer Datenmenge anhand von n-Featurs. Die Featureabhängigkeiten
(bzw. Clusterzugehörigkeiten) werden vom angewandten Algorithmus ermittelt.

Weitere Anwendungsgebiete sind:

* Datenvisualisierung
* Dimensionsreduktion (feature extraction – also welche Features beeinflussen das Ergebnis)
* Anomaly detection (Fraud detection)
* association rule learning (Identifizierung von Korrelationen – z.B. Chips + Ketchup korrelieren mit Steaks)

Typische ML Algorithmen:

* Clustering
    * k-means
    * hierarchical cluster analysis (HCA)
    * expectation maximization
* visualization and dimensionality reduction
    * principal component analysis (PCA)
    * Kernel PCA
    * Locally-Linear Embedding (LLE)
    * t-distributed Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
    * Apriori
    * Eclat

Zurück zu :ref:`mltypes`

Semisupervised learning
=======================
Eine Mischform von supervised und unsupervised learning. Beispiel sind Algorithmen, die auf Fotos eine Person
identifizieren und Clustern können (unsupervised), werden diesen Personen dann Namen gegebene (Label), kann nach diesen
gesucht und für supervised learning verwendet werden.

Zurück zu :ref:`mltypes`

Reinforcement Learning
=======================
Hierbei handelt es sich um ein System, welches auf Basis eines „gelernten“ Ergebnisses eine bessere zukünftige Strategie
erarbeitet.

*…“The learning system, called an agent in this context, can observe the environment, select and perform actions and get
rewards in return (or penalties in the form of negative rewards). It must then learn by itself what is the best strategy,
called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a
given situation“ [Géron, S.32]*

Zurück zu :ref:`mltypes`

Batch / Offline Learning
========================
Beim Batch Learning kann das ML System nicht auf aktuelle (online) Daten zugreifen und die Aussage/Vorhersage verbessern.
Die vom ML System ermittelten Funktionen werden einmal auf Basis der vorhandenen Daten berechnet, in der Produktion
werden diese Funktionen angewandt. Die Prozedur ist CPU / Disk intensiv. Aktualisierungen können zwar automatisiert
eingespielt werden, verlangen aber immer wieder die gleiche ressourcenintensive Berechnung (sind somit ungeeignet für
Systeme mit limitierten Ressourcen)

Zurück zu :ref:`mltypes`

Online / Incremental Learning
=============================
Online Learning ML Systeme sind in der Lage, zusätzliche Daten in den Algorithmus aufzunehmen (also inkrementelles
lernen) und in die Vorhersage einzubinden. Es kann aber auch verwendet werden bei sehr großen Systemen, bei dem die
Daten nicht mehr in den Hauptspeicher passen (=out of core learning). Hierbei werden die Daten sukzessive in den RAM
geladen und verarbeitet. Das ganze passiert damit im Batch, insofern ist der Begriff Online Learning hier irreführend.

Ein wichtiger Parameter ist die „learning rate“. Wie schnell werden gelernte Daten in die Vorhersagen implementiert.
Bei einer hohen Lernrate werden „alte“ Daten schneller vergessen. Dies kann aber auch problematisch sein, wenn nur noch
„schlechte“ Daten gelernt werden und die „guten“ Daten dann vergessen werden (also shit in – shit out).

Zurück zu :ref:`mltypes`

Instance Based vs. Model-Based Learning
========================================
Frage der Generalisierung des Modells.

Beim Instance based Learning wird anhand von positiv (negativ) Beispielen neue Werte gelernt, die ähnlich sind, wie die
vorgegebenen Kategorien.

Beim Model-based Learning wird ein Modell mit Hilfe von Algorithmen errechnet und Schätzungen erfolgen auf Basis der
berechneten Funktionen.

Zurück zu :ref:`mltypes`

Scikit-Learn cheat sheet
========================
Welchen Algorithmus wende ich wann an: `Scikit-Learn cheat sheet`_

.. _Scikit-Learn cheat sheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Zurück zu :ref:`mltypes`

Quellen:
    [Géron]: Hands-On Machine Learning with Scikit-Learn & TensorFlow, Aurélien Géron, o’reilly, 2017, S. 26ff.

