import tensorflow as tf
import libri as ls
import numpy as np
import time

# Dossier ou sera stocke les donnees du superviseur
path_log = 'C:/Users/mkhalem/IdeaProjects/speechrecognizer/python/log_am2/'

class AcousticModel:
    
    def __init__(self):
        
        self.batch_size_awaited = 1 # Nombre d'exemples a prendre par batch
        self.batch_s = 1 # Nombre d'exemples a prendre par mini-batch
        n_lstm = 768 # Nombre de neurones par couche LSTM
        n_input = 123 # Nombre d'entrees dans notre reseau (40 log mel-filterbank feature et energie + delta + double delta)
        n_classes = 30 # Nombre de sorties de notre reseau (26 lettres, 1 espace, 1 apostrophe, 1 EOS, 1 blank label pour le ctc)
        
        learning_rate = 1e-5 # Vitesse d'apprentissage
        
        # On cree un placeholder 3D [ max_length x nombre_exemples x nombre_features ]
        # Il recevra les entrees pour notre reseau
        self._inputs = tf.placeholder(tf.float32, shape=(None, self.batch_s, n_input), name="inputs")
        
        # On cree un placeholder 1D [ nombre_exemples ]
        # Cette entree permet de stocker la longueur de chaque exemple
        self._seq_length = tf.placeholder(tf.int32, shape=(self.batch_s), name="sequence_Length")
        
        # On cree un sparse_placeholder requis pour calculer le ctc_loss
        # Il permet de recevoir les sorties attendues par notre reseau
        self._targets = tf.sparse_placeholder(tf.int32, name="targets")
        
        # Variable self.learning_rate qui n'est pas entrainable. Elle est initialisee a 1e-5 ici (voir ci-dessus)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
        
        # Operation permettant de faire decroitre le learning_rate en le divisant par 10 == multipliant par 0.1
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.1)
        
        with tf.name_scope('RNN'):
            # On cree nos 2 couches LSTM de 768 cellules chacune 
            hidden_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_lstm), input_keep_prob=0.8, output_keep_prob=0.5) for _ in range(2)])
            
            # On cree un etat initial de zero
            initial_state = hidden_layers.zero_state(self.batch_s, tf.float32)
            
            # Operation permettant de calculer la sortie des couches LSTM depuis les entrees
            rnn_output, state = tf.nn.dynamic_rnn(hidden_layers, self._inputs, sequence_length=self._seq_length,  initial_state=initial_state, time_major=True)
            
            # On change la disposition de notre sortie
            # Elle se presentait de la maniere suivante [ max_length x nombre_exemples x nombre_lstm ]
            # On etale notre matrice 3D en 2D
            # On cherche a avoir [ (nombre_exemples x max_length) x nombre_lstm ] pour decrire la sortie de chaque cellule pour toutes les entrees a la suite
            rnn_output = tf.reshape(tf.transpose(rnn_output, [1,0,2]), [-1, n_lstm])
        
        
        with tf.name_scope('Logits'):
            # Variable W pour les poids et b pour le biais
            W = tf.get_variable("weights", shape=[n_lstm, n_classes])
            b = tf.get_variable("biases", shape=[n_classes])
            
            # On multiplie la sortie avec sa forme remanie avec la matrice de poids W et en additionnant le biais b
            rnn_output = tf.matmul(rnn_output, W) + b
            
            # Apres ce calcul, notre rnn_output a un format [ (nombre_exemples x max_length) x n_classes ]
            # On recree une matrice 3D depuis la matrice 2D. On reforme une matrice [ max_length x nombre_exemples x n_classes ]
            logits = tf.transpose(tf.reshape(rnn_output, [self.batch_s, -1, n_classes]), [1,0,2])
        
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(self._targets, logits, self._seq_length)

            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(self._seq_length)))

            # Set an accumulator to sum the loss between mini-batchs
            self.accumulated_mean_loss = tf.Variable(0.0, trainable=False)
            self.acc_mean_loss_op = self.accumulated_mean_loss.assign_add(mean_loss)
            self.acc_mean_loss_zero_op = self.accumulated_mean_loss.assign(tf.zeros_like(self.accumulated_mean_loss))
            
            # Operation permettant de faire la prediction depuis les entrees. On convertit le chemin vers les entiers pour qu'il puisse etre converti par la suite par notre parseur
            self.prediction = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits, self._seq_length, top_paths=5)[0])            
    
        with tf.name_scope('Summary'):
            # On cree les summary des variables ctc_loss et learning_rate
            tf.summary.scalar("ctc_loss", self.accumulated_mean_loss)
            tf.summary.scalar("learning_rate", self.learning_rate)
            
            # On a aussi une variable global_step qui permet a notre superviseur de connaitre l'etat d'avancement de notre apprentissage
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Compute the gradients
        trainable_variables = tf.trainable_variables()
        with tf.name_scope('Gradients'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = opt.compute_gradients(ctc_loss, trainable_variables)

            # Define a list of variables to store the accumulated gradients between batchs
            accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                                     for tv in trainable_variables]

            # Define an op to reset the accumulated gradient
            self.acc_gradients_zero_op = [tv.assign(tf.zeros_like(tv)) for tv in accumulated_gradients]

            # Define an op to accumulate the gradients calculated by the current batch with
            # the accumulated gradients variable
            self.accumulate_gradients_op = [accumulated_gradients[i].assign_add(gv[0])
                                            for i, gv in enumerate(gradients)]

            # Define an op to apply the result of the accumulated gradients
            self.train_step_op = opt.apply_gradients([(accumulated_gradients[i], gv[1]) for i, gv in enumerate(gradients)],
                                                     global_step=self.global_step)        
        
        # Variable permettant de conserver le superviseur
        self.sv = tf.train.Supervisor(logdir=path_log)
        
    # Fonction d'entrainement de notre reseau
    def train(self):
        # On initialise previous_loss et no_improvement_since a 0
        previous_loss = 0
        no_improvement_since = 0
        # On demarre la session depuis le superviseur. Celui-ci se chargera de restaurer les variables si elles sont disponibles depuis une sauvegarde
        # Sinon ils les initialisera
        
        # On stocke un pattern de phrase pour printer a chaque tour
        log = "Step {}, batch_cost = {:.3f}, time = {:.3f}"
        with self.sv.managed_session() as sess:
            while True:
                # Si le superviseur demande un arret, on coupe la boucle d'apprentissage
                if self.sv.should_stop():
                    break
                # On prend une mesure du temps pour savoir combien de temps dure l'etape
                start = time.time()
                sess.run([self.acc_gradients_zero_op, self.acc_mean_loss_zero_op])
                for _ in range(0, self.batch_size_awaited, self.batch_s):
                    
                    # On recupere les exemples que l'on souhaite entrainer
                    (inputs, seq_len), targets = ls.getAMExamples(self.batch_s, from_beginning=True) 
                    
                    # On calcule les gradients pour ce batch, on l'accumule dans les variables maintenant les gradients
                    # On calcule le loss du batch aussi
                    sess.run([self.accumulate_gradients_op, self.acc_mean_loss_op], feed_dict={self._inputs: inputs, self._seq_length: seq_len, self._targets: targets})
                
                # Apres la boucle, on execute l'entrainement sur les gradients accumules
                # On recupere le loss accumule pour pouvoir l'afficher a l'utilisateur
                _,loss = sess.run([self.train_step_op, self.accumulated_mean_loss])
                print(log.format(self.global_step.eval(session=sess), loss, time.time() - start))
                # Si la valeur du loss n'a pas evolue par rapport a l'ancienne
                if loss >= previous_loss:
                    # On incremente no_improvement_since de 1
                    no_improvement_since += 1
                    # Si on arrive a 2
                    if no_improvement_since == 2:
                        # On decremente notre vitesse d'apprentissage en appelant l'operation learning_rate_decay_op
                        print('Decrement learning rate')
                        sess.run(self.learning_rate_decay_op)
                        # On remet la variable no_improvement_since a 0
                        no_improvement_since = 0
                        # Si la vitesse d'apprentissage est en dessous de 1e-7, le modele arrete de s'entrainer et on brise la boucle d'apprentissage
                        if self.learning_rate.eval(session=sess) < 1e-7:
                            break
                else:
                    no_improvement_since = 0
                # On change la valeur de previous_loss par loss
                previous_loss = loss
    
    # Fonction de prediction
    def predict(self, inputs, seq_len):
        with self.sv.managed_session() as sess:
            # On lance l'operation prediction avec les donnees d'entrees
            # On recupere le vecteur values qui represente la sequence decryptee de notre son par notre reseau de neurones
            # On convertit enfin afin d'obtenir la sequence strng 
            outputs = sess.run([self.prediction], feed_dict={self._inputs: inputs, self._seq_length: seq_len})
            return ls.int_to_string(outputs[0].values)

# Si on demarre le fichier directement
if __name__ == '__main__':
    # On cree un AcousticModel
    m = AcousticModel()
    # On l'entraine 
    #m.train()
    # On essaie de predire le premier exemple de notre dataset
    (inputs, seq_len), targets = ls.getAMExamples(1, from_beginning=True)
    print(m.predict(inputs, seq_len))