import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import libri as ls
import time

# Dossier ou sera stocke les donnees du superviseur
path_log = 'C:/Users/mkhalem/IdeaProjects/speechrecognizer/python/log_lm/'

class LanguageModel:
    
    def __init__(self):
        
        self.batch_s = 8 # Nombre d'exemples a prendre par batch
        n_lstm = 512 # Nombre de neurones par couche LSTM
        n_input = n_classes = 29 # Nombre d'entrees dans notre reseau (28 caracteres pour celui en cours)
        
        learning_rate = 0.1 # Vitesse d'apprentissage
        
        # On cree un placeholder 3D [ nombre_exemples x max_length ]
        # Il recevra les entrees pour notre reseau
        self._inputs = tf.placeholder(tf.int32, shape=(None, 1), name="inputs")
        
        # On cree un sparse_placeholder requis pour calculer le ctc_loss
        # Il permet de recevoir les sorties attendues par notre reseau
        self._targets = tf.placeholder(tf.int32, shape=(None, 1), name="targets")
        _targets = tf.squeeze(self._targets)
        
        # Variable self.learning_rate qui n'est pas entrainable. Elle est initialisee a 1e-5 ici (voir ci-dessus)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
        
        # Operation permettant de faire decroitre le learning_rate en le divisant par 10 == multipliant par 0.1
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.5)
        
        # Variable self.loss_variable qui n'est pas entrainable. Elle est initialise a 0 ici
        self.loss_variable = tf.Variable(0.0, trainable=False, name="loss_variable")
        
        # Conversion des donnees d'entree vers un encodage one-hot
        self.one_hot_inputs = tf.one_hot(self._inputs, n_input)
        
        with tf.name_scope('RNN'):
            # On cree nos 2 couches LSTM de 512 cellules chacune 
            hidden_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_lstm) for _ in range(2)])
            
            # On cree un etat initial de zero
            shape = tf.shape(self.one_hot_inputs)
            initial_state = hidden_layers.zero_state(shape[1], tf.float32)
            
            # Operation permettant de calculer la sortie des couches LSTM depuis les entrees
            rnn_output, state = tf.nn.dynamic_rnn(hidden_layers, self.one_hot_inputs, initial_state=initial_state, time_major=True)
        
        # On change la disposition de notre sortie
        # Elle se presentait de la maniere suivante [ nombre_exemples x max_length x nombre_lstm ]
        # On etale notre matrice 3D en 2D
        # On cherche a avoir [ (nombre_exemples x max_length) x nombre_lstm ] pour decrire la sortie de chaque cellule pour toutes les entrees a la suite
        rnn_output = tf.reshape(rnn_output, [-1, n_lstm])
        
        with tf.name_scope('Logits'):
            # Variable W pour les poids et b pour le biais
            W = tf.get_variable("weights", shape=[n_lstm, n_classes])
            b = tf.get_variable("biases", shape=[n_classes])
            
            # On multiplie la sortie avec sa forme remanie avec la matrice de poids W et en additionnant le biais b : @ == tf.matmul
            logits = (rnn_output @ W) + b
        
        # Operation permettant de calculer le softmax_cross_entropy entre les targets et les labels
        with tf.name_scope('Loss'):
            loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=_targets)
            self.loss = tf.reduce_mean(loss_batch)
            # Operation permettant d'assigner le loss a la variable loss
            self.loss_variable_assign_op = self.loss_variable.assign(self.loss)
        
        # Operation permettant de predire le caractere suivant depuis l'entree soumise
        self.logits_pred = logits[-1,:]
        self.prediction = tf.argmax(self.logits_pred)
        
        with tf.name_scope('Summary'):
            # On cree les summary des variables ctc_loss et learning_rate
            tf.summary.scalar("loss", self.loss_variable)
            tf.summary.scalar("learning_rate", self.learning_rate)    
        
            # On a aussi une variable global_step qui permet a notre superviseur de connaitre l'etat d'avancement de notre apprentissage
            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        with tf.name_scope('Optimizer'):
            # Operation permettant de mettre a jour les poids des variables accordement au loss
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        
        # Variable permettant de conserver le superviseur    
        self.sv = tf.train.Supervisor(logdir=path_log, save_model_secs=300)
        
    # Fonction d'entrainement de notre reseau
    def train(self):
        # On initialise previous_loss et no_improvement_since a 0
        epoch = 1
        previous_loss = 0.0
        number_decrement = 0
        # On demarre la session depuis le superviseur. Celui-ci se chargera de restaurer les variables si elles sont disponibles depuis une sauvegarde
        # Sinon ils les initialisera
        
        # On stocke un pattern de phrase pour printer a chaque tour
        log = "Epoch {} / Step {}, batch_cost = {:.6f}, time = {:.3f}"
        log_validation = "Epoch {}, cost_validation = {:.6f}"
        with self.sv.managed_session() as sess:
            while True:
                # Si le superviseur demande un arret, on coupe la boucle d'apprentissage
                if self.sv.should_stop():
                    break
                for inputs, targets in ls.getAll_LMData(self.batch_s):
                
                    # On prend une mesure du temps pour savoir combien de temps dure l'etape
                    start = time.time()
                
                    # On fait le loss afin de recuperer d'en recuperer la valeur, opt pour optimiser le reseau  
                    loss,_,_ = sess.run([self.loss, self.opt, self.loss_variable_assign_op], feed_dict={self._inputs: inputs, self._targets: targets})
                
                    # On affiche la phrase d'etape
                    print(log.format(epoch, self.global_step.eval(session=sess), loss, time.time() - start))
                
                epoch += 1
                loss_validation = 0.0
                for inputs, targets in ls.getAll_LMData(self.batch_s, data_type='evaluation'):
                
                    # On prend une mesure du temps pour savoir combien de temps dure l'etape
                    start = time.time()
                
                    # On fait le loss afin de recuperer d'en recuperer la valeur, opt pour optimiser le reseau  
                    loss_validation += sess.run([self.loss], feed_dict={self._inputs: inputs, self._targets: targets}) / len(inputs)
                
                # On affiche la phrase d'etape
                print(log_validation.format(epoch, loss_validation))                
                # Si la valeur du loss n'a pas evolue par rapport a l'ancienne
                if previous_loss > 0.0 and loss_validation >= previous_loss:
                    # Si on a deja decremente notre learning rate, on s'arrete
                    if number_decrement > 0:
                        break
                    else:
                        # Sinon on decremente notre vitesse d'apprentissage en appelant l'operation learning_rate_decay_op
                        print('Decrement learning rate')
                        sess.run(self.learning_rate_decay_op)
                        # On incremente number_decrement de 1
                        number_decrement += 1
                # On change la valeur de previous_loss par loss_validation
                previous_loss = loss_validation
                
    def predict(self, n, from_=[]):
        with self.sv.managed_session() as sess:
            predict = np.array(from_)
            for _ in range(n):
                prediction = sess.run([self.prediction], feed_dict={self._inputs: predict})
                predict = np.expand_dims(np.append(predict, [prediction[0]]), axis=1)
            return predict
        
# Si on demarre le fichier directement
if __name__ == '__main__':
    # On cree un LanguageModel
    m = LanguageModel()
    # On l'entraine
    m.train()