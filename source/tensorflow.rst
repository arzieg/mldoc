.. _tf:

###################
Tensorflow
###################




Convolution
****************
* Beim Imagevergleich eine Art, bestimmte Werte bei einem Bild zu akzentuieren (und zu komprimieren)
* "but the ultimate concept is that they narrow down the content of the image to focus on specific parts and this will likely improve the model accuracy."
* https://en.wikipedia.org/wiki/Kernel_(image_processing)
* Idee: 
  - definiere eine Matrix (z.B. 3x3 Pixel) und verdichte diese auf einen Wert
    
    Pixel in der Mitte 192, Verdichtung und Berechnung eines neuen Pixelwerts durch Betrachtung der Nachbarpixel 

    0  |  64 | 128
    ---------------
    48 | 192 | 144
    ---------------
    142 | 226 | 168

    Filtermatrix

    -1 |  0 | -2
    ---------------
    .5 | 4.5| -1.5
    ----------------
    1.5 | 2 | -3

    Neuer-Pixelwert: (-1 * 0) + (0 * 64) + (-2 * 128) + (.5 * 48) + (4.5 * 192) + (-1.5 * 144) + (1.5 * 142) + (2 * 226) + (-3 * 168)

  - Durch den Filter wird auf gewisse Elemente fokussiert. Bsp. wenn der Filter in der mittleren Zeile nur 0 enthält, werden die 
    horizontalen Linien ausgeblendet, wenn die mittlere Spalte nur 0 enthält, dann die vertikalen Spalten eines Bildes. 

Pooling
========
* Pooling ist eine Art von Kompression
* Bsp.: Matrix von 2x2 wird definiert. Der maximale Wert wird hieraus genommen und ist der "neue" Pixelwert. Dann springe zwei Elemente weiter
*       nehme aus der dortigen 2x2 Matrix wieder den Maximalwert usw. Also aus 4x4 Bildpunkten wird dann auf eine 2x2 Matrix kondensiert. 
*       