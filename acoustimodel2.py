import os
import tensorflow as tf
import libri as ls
import numpy as np
import time

# Dossier ou sera stocke les donnees du superviseur
path_log = 'C:/Users/mkhalem/Desktop/model/log_am2'
#path_log = 'E:/Projet/PC/SpeechRecognizer/model/log_am'

class AcousticModel:
    
    def __init__(self):
        
        self.batch_s = 1 # Nombre d'exemples a prendre par mini-batch
        n_lstm = 320 # Nombre de neurones par couche LSTM
        n_input = 123 # Nombre d'entrees dans notre reseau (40 log mel-filterbank feature et energie + delta + double delta)
        n_classes = 30 # Nombre de sorties de notre reseau (26 lettres, 1 espace, 1 apostrophe, 1 EOS, 1 blank label pour le ctc)
        
        learning_rate = 0.00004 # Vitesse d'apprentissage
        
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
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.5)
        
        with tf.name_scope('RNN'):
            
            # On cree nos 4 couches forward LSTM de 320 cellules chacune
            hidden_layers_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_lstm) for _ in range(4)])
            
            # On cree nos 4 couches backward LSTM de 320 cellules chacune
            hidden_layers_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_lstm) for _ in range(4)])
            
            
            initial_state_fw = hidden_layers_fw.zero_state(self.batch_s, tf.float32)
            initial_state_bw = hidden_layers_bw.zero_state(self.batch_s, tf.float32)
            # Operation permettant de calculer la sortie des couches LSTM depuis les entrees
            self.outputs, state = tf.nn.bidirectional_dynamic_rnn(hidden_layers_fw, hidden_layers_bw, self._inputs, sequence_length=self._seq_length, time_major=True, initial_state_bw=initial_state_bw, initial_state_fw=initial_state_fw)
            self.outputs = tf.concat(self.outputs, 2)
            # On change la disposition de notre sortie
            # Elle se presentait de la maniere suivante [ max_length x nombre_exemples x nombre_lstm ]
            # On etale notre matrice 3D en 2D
            # On cherche a avoir [ (nombre_exemples x max_length) x nombre_lstm ] pour decrire la sortie de chaque cellule pour toutes les entrees a la suite
            rnn_output = tf.reshape(tf.transpose(self.outputs, [1,0,2]), [-1, n_lstm])
        
        
        with tf.name_scope('Logits'):
            # Variable W pour les poids et b pour le biais
            W = tf.get_variable("weights", shape=[n_lstm, n_classes], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            b = tf.get_variable("biases", shape=[n_classes])
            
            # On multiplie la sortie avec sa forme remanie avec la matrice de poids W et en additionnant le biais b
            rnn_output = tf.matmul(rnn_output, W) + b
            
            # Apres ce calcul, notre rnn_output a un format [ (nombre_exemples x max_length) x n_classes ]
            # On recree une matrice 3D depuis la matrice 2D. On reforme une matrice [ max_length x nombre_exemples x n_classes ]
            logits = tf.transpose(tf.reshape(rnn_output, [self.batch_s, -1, n_classes]), [1,0,2])
        
        with tf.name_scope('CTC'):
            self.ctc_loss = tf.nn.ctc_loss(self._targets, logits, self._seq_length)
            
            # Operation permettant de faire la prediction depuis les entrees. On convertit le chemin vers les entiers pour qu'il puisse etre converti par la suite par notre parseur
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self._seq_length, beam_width=512)
            
            self.accuracy = tf.edit_distance(tf.cast(self.decoded[0], dtype=tf.int32), self._targets, normalize=False)
    
        with tf.name_scope('Summary'):
            # On cree les summary des variables ctc_loss et learning_rate
            tf.summary.scalar("ctc_loss_training", self.ctc_loss)
            tf.summary.scalar("learning_rate", self.learning_rate)
            
            # On a aussi une variable global_step qui permet a notre superviseur de connaitre l'etat d'avancement de notre apprentissage
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Optimize the network
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ctc_loss)
        
        # Variable permettant de conserver le superviseur
        self.sv = tf.train.Supervisor(logdir=path_log)
        
    # Fonction d'entrainement de notre reseau
    def train(self):
        # On initialise previous_loss et no_improvement_since a 0
        epoch = 1
        previous_lar = 100.0
        drop_below_05 = 0
        drop_below_01 = 0
        
        # On demarre la session depuis le superviseur. Celui-ci se chargera de restaurer les variables si elles sont disponibles depuis une sauvegarde
        # Sinon ils les initialisera
        with self.sv.managed_session() as sess:
            finishedLearning = True
            while finishedLearning:
                # Si le superviseur demande un arret, on coupe la boucle d'apprentissage
                if self.sv.should_stop():
                    break
                # On prend une mesure du temps pour savoir combien de temps dure l'etape
                start = time.time()
            
                print('Training data...')
                for step, ((inputs, seq_len), targets) in enumerate(ls.getAll_AMData(self.batch_s), start=1):
                    
                    # On calcule le cout du batch et on accumule le gradient
                    _,loss_training = sess.run([self.optimizer, self.ctc_loss], feed_dict={self._inputs: inputs, self._seq_length: seq_len, self._targets: targets})
                    
                    print('Epoch {} - Etape {} - Current loss : {}'.format(epoch, step, loss_training))
                    
                print('Validation data...')
                label_errors = []
                for step, ((inputs, seq_len), targets) in enumerate(ls.getAll_AMData(self.batch_s, data_type='evaluation'), start=1):
                    
                    # On calcule le cout du batch et on accumule le gradient
                    nbr_error, pred = sess.run([self.accuracy, self.decoded], feed_dict={self._inputs: inputs, self._seq_length: seq_len, self._targets: targets})
                    error = nbr_error/len(pred[0].values) * 100
                    label_errors += [error]
                    
                    print('Etape {} - Error : {}%'.format(step, error))
                
                label_error = np.mean(label_errors)
                diff_target = previous_lar - label_error
                if diff_target < 0.5:
                    drop_below_05 += 1
                else:
                    drop_below_05 = 0
                    
                if diff_target < 0.1:
                    drop_below_01 += 1
                else:
                    drop_below_01 = 0
                
                if drop_below_05 == 2:
                    sess.run(self.learning_rate_decay_op)                
                if(drop_below_01 == 2):
                    finishedLearning = False
                
                # On incremente le compteur d'epoch
                epoch += 1 
    
    # Fonction de prediction
    def predict(self, inputs, seq_len):
        with self.sv.managed_session() as sess:
            # On lance l'operation prediction avec les donnees d'entrees
            # On recupere le vecteur values qui represente la sequence decryptee de notre son par notre reseau de neurones
            # On convertit enfin afin d'obtenir la sequence strng
            (decoded, log_probabilities) = sess.run(self.prediction, feed_dict={self._inputs: inputs, self._seq_length: seq_len})
            sentence = ls.int_to_string(decoded[0].values)
        return sentence, log_probabilities[0][0]

# Si on demarre le fichier directement
if __name__ == '__main__':
    # On lit le fichier audio de test
    #inputs = ls.__convertSoundToInputValues('english.wav')
    #seq_len = [len(inputs)]
    #inputs = np.transpose([inputs], [1,0,2])
    # On cree un AcousticModel
    m = AcousticModel()
    # On l'entraine 
    m.train()
    # On essaie de predire le premier exemple de notre dataset
        


    #print('"{}" - Probabilite : {}%'.format(sentence, prob))