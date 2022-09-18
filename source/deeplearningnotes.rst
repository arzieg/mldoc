.. _deeplearning:

#################################
Deep Learning Notes
#################################

Tensorboard starten
=====================
Quelle: Aurelien Geron, "Praxiseinstieg Machine Learning", OReilly, 2. Auflage, S318ff.

1. Erzeuge ein logdir unter Deinem Pfad 

.. code-block:: python

    import os
    root_logdir = os.path.join(os.curdir, 'my_logs')

    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()

2. Callback in keras integrieren

.. code-block:: python
    
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])

3. Tensorboard aufrufen auf cmd-line
.. code-block:: 

    tensorboard --logdir=./my_logs --port=6006


Tensorboard anpassen
======================
In diesem Beispiel kann man die Webseite von tensorboard anpassen und erweitern um Histogramme, Images, Text, audio

.. code-block:: python
    
    test_logdir = get_run_logdir()
    writer = tf.summary.create_file_writer(test_logdir)
    with writer.as_default():
    for step in range(1,1000+1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
        data = (np.random.randn(100)+2)* step/100 # Zufallsdaten
        tf.summary.histogram("my_hist", data, buckets=50, step=step)
        images = np.random.rand(2,32,32,3) # rand 32x32 rgb images
        tf.summary.image("my_images", images * step /1000, step=step)
        texts = ["Schritt: " + str(step), " Quadrat: " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 *np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1,-1,1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

Feinabstimmung der Hyperparameter eines NN
===========================================

1. Anzahl hidden layers
   Tiefere Netze besitzen eines höhere Parametereffizienz als flache. Sie können komplexe Funktionen mit exponentiell weniger Neuronen
   als flache Netze modellieren, wodurch sie mit gleicher Menge an Trainingsdaten eine deutlich bessere Performance erzielen. 
   Bei einfachen Aufgaben reichen 1-2 layer. Bei komplexeren Aufgaben kann die Anzahl der Layer erhöht werden bis die Daten "Overfitten".
   Bei komplexen Aufgaben hat man dann dutzende und mehr Layer im Einsatz, wobei bereits trainierte Schichten eingesetzt werden können. 

2. Anzahl Neuronen pro hidden layer:
   Die Anzahl der Neuronen für die Ein- und Ausgabeschicht wird durch die Art von Ein- und Ausgabe bestimmt. Bsp. MNIST mit Bildern:
   28 x 28 = 784 Eingabeneuronen, 10 Kategorien als Ausgabe = 10 Neuronen
   In der Praxis ist es oft einfacher ein Modell mit mehr Schichten und Neuronen zu wählen, als sie tatsächlich benötigen, und durch 
   Early Stopping ein Overfitting zu verhindern. 
   Im Allg. entwickelt das erhöhen der Anzahl der Schichten mehr Durchschlagskraft als das Erhöhen der Anzahl der Neuronen pro Schicht.

3. Lernrate 
   im Allg. beträgt die optimale Lernrate die Hälfte der maximalen Lernrate. Wie findet man diese: 
   Modell für ein paar 100 Iterationen trainieren. 
   
   * Starte mit einer niedrigen Lernrate (bsp. 10e-5) und steigere diese bis zu einer sehr hohen Lernrate (bspw. 10)
   * bei jeder Iteration wird die Lernrate um einen konstanten Wert erhöht (bsp. exp(log(10^6)/500)) um in 500er Schritten zu 10 zu gelangen
   * der Verlust wird in einer log.Scale abgebildet, d.h. sie sollte erst fallen und ab einem bestimmten Punkt wieder steigen (U-Kurve)
   * die optimale Lernrate wird liegt dann ein wenig unter dem minimalen Punkt(ca. 10 x niedriger als der Wendepunkt)

4. Batchgröße
   Die Batchgröße kann einen erheblichen Einfluss auf die Leistung und die Trainingsdauer haben. 
   Eine Strategie besteht also darin, eine große Batchgröße zu verwenden und die Lernrate zu steigern. Wird das Training dadurch instabil oder
   ist die resultierende Performance unbefriedigend, versuchen sie es mit einer kleineren Lernrate (und Batchgröße?)

5. Aktivierungsfunktion
   für die hidden layer ist RELU gut, für die Ausgabeschicht hängt es von der Aufgabe ab. 

6. Anzahl Iterationen
   Verwenden Sie early stopping