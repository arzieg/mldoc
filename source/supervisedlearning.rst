.. _sl:

###################
Supervised Learning
###################

Beim SL sind neben den Beobachtungen (Features) auch die daraus resultierenden Ergebnisse bekannt.

Wichtige „learning“ Algorithmen sind:

* k-Nearest Neighbors
* Linear Regression
* Logistic Regression
* Support Vector Maschines (SVMs)
* Decision Trees and Random Forests
* Neuronal networks
* Linear Regression (LR)
* Der Klassiker: supervised_learning_linear_regression

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

