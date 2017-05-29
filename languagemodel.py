import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import libri as ls
import time

# Dossier ou sera stocke les donnees du superviseur
path_log = 'C:/Users/mkhalem/IdeaProjects/speechrecognizer/python/log_lm/'
#path_log = 'E:/Projet/PC/SpeechRecognizer/model/log_lm'

class LanguageModel:
    
    def __init__(self):
        
        self.batch_s = 8 # Nombre d'exemples a prendre par batch
        n_lstm = 512 # Nombre de neurones par couche LSTM
        n_input = n_classes = 29 # Nombre d'entrees dans notre reseau (29 caracteres pour celui en cours)
        
        learning_rate = 0.1 # Vitesse d'apprentissage
        
        # On cree un placeholder 2D [ nombre_exemples x max_length ]
        # On presente la phrase comme une multitude d'exemples specifies comme une suite de caracteres
        # Il recevra les entrees pour notre reseau
        self._inputs = tf.placeholder(tf.int32, shape=(None, 1), name="inputs")
        
        # On cree un placeholder 2D [ nombre_exemples x max_length ]
        # Les caracteres suivants sont notes comme cible
        # Il permet de recevoir les sorties attendues par notre reseau
        self._targets = tf.placeholder(tf.int32, shape=(None, 1), name="targets")
        # On ecrase la dimension de 1 qui ne sert pas dans la fonction de loss
        _targets = tf.squeeze(self._targets)
        
        # Variable self.learning_rate qui n'est pas entrainable. Elle est initialisee a 0.1 ici (voir ci-dessus)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
        
        # Operation permettant de faire decroitre le learning_rate en le divisant par 2 == multipliant par 0.5
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.5)
        
        # Variable self.loss_training qui n'est pas entrainable. Elle est initialise a 0 ici
        # Elle permet de mesurer le loss sur la base de training
        self.loss_training = tf.Variable(0.0, trainable=False, name="loss_training")
        
        # Variable self.loss_validation qui n'est pas entrainable. Elle est initialise a 0 ici
        # Elle permet de mesurer le loss sur la base de validation
        self.loss_validation = tf.Variable(0.0, trainable=False, name="loss_validation")
        
        
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
            W = tf.Variable(tf.random_normal([n_lstm, n_classes], stddev=0.35), name="weights")
            b = tf.Variable(tf.zeros([n_classes]), name="biases")
            
            # On multiplie la sortie avec sa forme remanie avec la matrice de poids W et en additionnant le biais b : @ == tf.matmul
            logits = tf.matmul(rnn_output,W) + b
        
        # Operation permettant de calculer le softmax_cross_entropy entre les targets et les labels
        with tf.name_scope('Loss'):
            loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=_targets)
            self.loss = tf.reduce_mean(loss_batch)
        
        # Operation permettant de predire le caractere suivant depuis l'entree soumise
        # On ne calcule que depuis la derniere sortie du rnn_output
        self.logits_pred = logits[-1,:]
        # On recupere l'argmax qui represente le caractere le plus probable du dernier caractere
        self.prediction = tf.argmax(self.logits_pred, axis=0)
        
        with tf.name_scope('Summary'):
            # On cree les summary des variables loss et learning_rate
            tf.summary.scalar("loss_training", self.loss_training)
            tf.summary.scalar("loss_validation", self.loss_validation)
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
        # On initialise previous_loss et number_decrement a 0 et epoch a 1
        epoch = 1
        previous_loss = 0.0
        number_decrement = 0
        
        # On stocke un pattern de phrase pour printer a chaque tour
        log = "Epoch {} / Step {} | {:.2f}% , batch_cost = {:.6f}, batch_eval_cost = {:.6f}, time = {:.3f}"
        # On demarre la session depuis le superviseur. Celui-ci se chargera de restaurer les variables si elles sont disponibles depuis une sauvegarde
        # Sinon ils les initialisera        
        with self.sv.managed_session() as sess:
            while True:
                # Si le superviseur demande un arret, on coupe la boucle d'apprentissage
                if self.sv.should_stop():
                    break
                
                # On parcoure l'ensemble de la base d'entrainement
                loss_training = []
                loss_validation = []
                # On parcoure l'ensemble de la base de training
                for step, (inputs, targets) in enumerate(ls.getAll_LMData(self.batch_s), start=1):
                    
                    # On prend une mesure du temps pour savoir combien de temps dure l'etape
                    start = time.time()
                
                    # On fait le loss afin de recuperer d'en recuperer la valeur, opt pour optimiser le reseau  
                    loss,_ = sess.run([self.loss, self.opt], feed_dict={self._inputs: inputs, self._targets: targets})
                    
                    # On ajoute le loss du batch a la liste de loss_training
                    loss_training += [loss]
                    
                    # On prend le meme nombre d'exemples dans la base d'evaluation
                    inputs_eval, targets_eval = ls.getN_LMData(self.batch_s, data_type='evaluation')
                    
                    # On calcule le loss sur l'exemple de la base d'evaluation
                    loss_eval = sess.run(self.loss, feed_dict={self._inputs: inputs_eval, self._targets: targets_eval}) 
                    
                    # On ajoute le loss du batch a la liste de loss_training
                    loss_validation += [loss_eval]
                    
                    # On met a jour la variable pour qu'elle puisse etre summarise 
                    sess.run(self.loss_training, feed_dict={self.loss_training: np.mean(loss_training)})                
                
                    # On met a jour la variable pour qu'elle puisse etre summarise 
                    sess.run(self.loss_validation, feed_dict={self.loss_validation: np.mean(loss_validation)})                    
                
                    # On affiche la phrase d'etape
                    print(log.format(epoch, self.global_step.eval(session=sess), min(100, (100 * (step*self.batch_s)/float(ls.nbr_data_training))), loss, loss_eval, time.time() - start))
                
                validation_loss = np.mean(loss_validation) 
                # Si la valeur du loss n'a pas evolue par rapport a l'ancienne
                if previous_loss > 0.0 and validation_loss >= previous_loss:
                    # Si on a deja decremente notre learning rate, on s'arrete
                    if number_decrement > 0:
                        break
                    else:
                        # Sinon on decremente notre vitesse d'apprentissage en appelant l'operation learning_rate_decay_op
                        print('Decrement learning rate')
                        sess.run(self.learning_rate_decay_op)
                        # On incremente number_decrement de 1
                        number_decrement += 1
                # On incremente le compteur d'epoch
                epoch += 1
                # On change la valeur de previous_loss par loss_val
                previous_loss = validation_loss
                
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