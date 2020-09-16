.. _nl_basic:

###################
Neuronale Netzwerke
###################

**Notation:**
    Wichtig ist die Unterscheidung bei den hochgestellten [] - Klammern vs. den () - Klammern.
    Die [] Klammern beziehen sich auf den Layer innerhalb eine NN. Die ()-Klammern beziehen sich auf
    ein Element z.B. aus einem Trainingsset.

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

In jedem Hidden Layer wird z und a berechnet wie im Modell des Logistic Regression. Der Outputwert geht dann,
je nach Definition als Inputwert in den Hidden Layer 2 usw.

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