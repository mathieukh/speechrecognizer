import os
import random
import numpy as np
import pickle as pkl
import scipy.io.wavfile as wav
from python_speech_features import fbank
from python_speech_features import delta

# Mettre les chemins relatifs a nos donnees ici
#path = "V:/Projets/Machine Learning/Data/"
path = "E:/Projet/PC/SpeechRecognizer/Data/"

path_training = path + "training/"
path_evaluation = path + "evaluation/"
data_file_training = path + 'data_training.pkl'
data_file_evaluation = path + 'data_evaluation.pkl'

data_training = []
nbr_data_training = len(data_training)
data_evaluation = []
nbr_data_evaluation = len(data_evaluation)


# Permettra de faire l'indexage de notre alphabet depuis ici
FIRST_INDEX = ord('A') - 1

def __convertSoundToInputValues(soundname):
    """
    Fonction permettant de convertir un son au format wav
    en une liste de vecteur de log-mel filterbank feature de taille 40
    On utilise les variables de notre modele afin de retourner la matrice adequate 
    """    
    # On lit notre fichier wav, on retourne la frequence et le signal 
    (rate,sig) = wav.read(soundname)
    # On calcule les 40 filterbank du signal a une frequence de 8000 hz, sur une fenetre de 25 ms et toutes les 10 ms. On extrait aussi l'energie
    fbank_feat, energy = fbank(sig,8000, winlen=0.025, winstep=0.01, nfilt=40, nfft=256)
    # On passe le vecteur energy de 1 dimension a 2 dimensions dans le but de la concatener avec le filterbank apres 
    energy = np.reshape(energy, (len(energy),1))
    
    # On calcule le log des filter bank precedemment retournees
    fbank_feat = np.log(fbank_feat)
    # On concatene les filterbank et l'energie
    f_e = np.append(fbank_feat, energy, axis=1)
    
    # On calcule le delta et le double delta
    d_fbank_feat = delta(f_e, 2)
    d_d_fbank_feat = delta(d_fbank_feat, 2)
    
    # On finit par concatener l'ensemble afin de retourner la matrice souhaitee
    return np.append(f_e, np.append(d_fbank_feat, d_d_fbank_feat, axis=1), axis=1)

# 0 espace / 1 - 26 A - Z / 27 apostrophe / 28 end of line
def string_to_int(data):
    da = []
    for y in data:
        if y == ' ':
            da.append(0)
        elif y == '\'':
            da.append(27)
        elif y == '_':
            da.append(28)
        else:
            da.append(ord(y) - FIRST_INDEX)
    return da

def int_to_string(data):
    da = []
    for y in data:
        if y == 0:
            da.append(' ')
        elif y == 27:
            da.append('\'')
        elif y == 28:
            da.append('_')
        else:
            i = y + FIRST_INDEX
            da.append(chr(i))
    return da


"""
Fonctions de recuperation de donnees pour le modele Acoustique
Les fichiers de sauvegarde sont indispensables. Ils se presentent sous la forme d'un fichier pickle.
Ils permettent de retenir une liste de couple avec l'id de chaque fichier ainsi que la phrase correspondante.
Cette etape permet d'avoir une lecture beaucoup plus rapide de la base de donnees au lieu de faire des requetes
systemes a repetition et tres couteuse en temps.
"""

def init_data():
    """
    Si les donnees ne sont pas pretes, on va les ouvrir depuis les fichiers de sauvegarde
    """
    global data_training
    global data_evaluation
    global nbr_data_training
    global nbr_data_evaluation
    if not data_training:
        with open(data_file_training, 'rb') as f:
            data_training = pkl.load(f)
            nbr_data_training = len(data_training)
    if not data_evaluation:
        with open(data_file_evaluation, 'rb') as f:
            data_evaluation = pkl.load(f)
            nbr_data_evaluation = len(data_evaluation)
            
def __targetsAM_from_data(data):
    """
    La fonction prend en parametre data
    C'est une liste de couple de son et de label
    labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the id for (batch b, time t). 
    labels.values[i] must take on values in [0, num_labels). See core/ops/ctc_ops.cc for more details.
    """
    sequence_length = []
    num_features = len(data[0][0][0])
    indices_label = []
    values_label = []
    max_length_label = 0
    # On admet que les donnees seront recus dans le format prevu a la sortie des fonctions d'obtention des donnees ci-dessus
    # On a donc une liste de tuple contenants dans chaque tuple le son (ne nous interesse pas ici) et la liste des labels associee a ce son
    
    # On enumere cette liste de tuple, en ignorant la premiere donnee qui ne nous interesse pas
    # i correspond a l'iterateur d'exemple, t correspond a la liste de labels
    for i,(inputs,targets) in enumerate(data):    
        # Sequence Length:
        sequence_length.append(len(inputs))
        
        # On itere sur la liste de labels
        # j correspond a l'iterateur de label, tj correspond au label 
        for j,targets_j in enumerate(targets):
            indices_label.append([i,j])
            values_label.append(targets_j)
        # Au dernier label, j = nombre_label - 1
        # on recupere le maximum du nombres de labels entre tous les exemples pour obtenir le max_length
        max_length_label = max(max_length_label, j+1)
    # Au dernier exemple, i = nombre_exemple - 1 car l'iteration commence a 0. On additionne 1 pour obtenir le nombre d'exemples
    shape_label = [i+1,max_length_label]
    
    # On recupere la longueur max des sequences d'entre tous les exemples
    max_length = max(sequence_length)
    # Pour notre reseau, il nous faut des entrees de type [max_time x batch_size x num_features] 
    # On cree la matrice d'entrees de notre reseau de neurones avec des zeros qui serviront de padding pour le reste de la matrice
    inputs_pla = np.zeros((max_length,i+1,num_features))
    for i, (inputs,_) in enumerate(data):
        # On vient remplir la matrice avec les donnees que l'on a
        inputs_pla[:sequence_length[i],i,:] = inputs
    return (inputs_pla,np.asarray(sequence_length)), (np.asarray(indices_label), np.asarray(values_label), np.asarray(shape_label))

def __get_AMData(data_type, id_data, sentence):
    """
    La fonction prend en entree 3 parametres
    data_type: training ou evaluation
    id_data: string au format id1-id2-id3 attendu
    sentence: phrase qui correspond au son id_data
    """
    # On initialise les donnees 
    init_data()
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        path = path_training
    elif data_type == 'evaluation':
        path = path_evaluation
    # Si on ne reconnait pas la donnee, on souleve une exception
    else:
        raise Exception('Les donnees ' + str(data_type) + 'ne sont pas disponibles')
    # On separe le string recu et on recupere les id correspondants
    id1,id2,id3 = id_data.split('-')
    # On constitue le chemin de notre son correspondant
    path_soundname = path + id1 + '/' + id2 + '/' + id_data + '.wav'
    # On retourne afin un couple du son converti et de la phrase a laquelle on aura ajoute le symbole de EOS
    return __convertSoundToInputValues(path_soundname), string_to_int(list(sentence.strip()) + ['_'])

def getN_AMData(n, data_type='training'):
    """
    n: nombre d'exemples a prendre
    data_type: training ou evaluation
    Retourne une liste de donnees prete a etre utilisee dans le reseau de neurones
    """
    # On initialise les donnees
    init_data()
    d = []
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        data = data_training
    elif data_type == 'evaluation':
        data = data_evaluation
    # On itere n fois
    for _ in  range(n):
        # On recupere dans la donnee de facon aleatoire
        data_id, sentence = random.choice(data)
        # On recupere le couple grace a la fonction __get_AMData et des donnees trouvees dans la liste
        d.append(__get_AMData(data_type, data_id, sentence))
    # On retourne enfin les donnees au format attendu par notre reseau de neurones
    return __targetsAM_from_data(d)


def getAll_AMData(n, data_type='training'):
    """
    n: nombre d'exemples a prendre par batch
    data_type: training ou evaluation
    Retourne un generateur de donnees prete a etre utilisees dans le reseau de neurones
    L'ensemble des donnees sont parcourues
    """
    # On initialise les donnees
    init_data()
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        data = data_training
    elif data_type == 'evaluation':
        data = data_evaluation
    # On melange notre dossier afin de pouvoir parcourir la liste des donnees dans un ordre distinct a chaque rappel de la fonction
    random.shuffle(data)
    # On initialise i a 0 et d a vide
    i = 0
    d = []
    # On parcourt l'ensemble des donnees disponibles
    for data_id, sentence in data:
        # Si i est inferieur a n, cela veut dire qu'il faut encore remplir le batch
        if i < n:
            # on ajoute la donnee au batch et on incremente i de 1
            d.append(__get_AMData(data_type, data_id, sentence))
            i += 1
        else:
            # Si i >= n, on renvoie d car le batch est a la taille souhaitee
            yield __targetsAM_from_data(d)
            # On remet la liste a vide et le compteur i a 0
            d = []
            i = 0
    # A la fin du parcours, si la liste d n'est pas vide, cela veut dire qu'un batch n'etait pas complet mais ne peut etre rempli pour le completer
    if d:
        # On renvoie donc ce dernier batch
        yield __targetsAM_from_data(d)



#### Fonction de recuperation de donnees pour le modele linguistique ####
        
def __get_LMData(data_type, id_data, sentence):
    """
    La fonction prend en entree 3 parametres
    data_type: training ou evaluation
    id_data: string au format id1-id2-id3 attendu
    sentence: phrase qui correspond au son id_data
    ready_to_use: permet de couper la phrase pour servir d'entree et de sortie directement par le RN
    """
    # On initialise les donnees 
    init_data()
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        path = path_training
    elif data_type == 'evaluation':
        path = path_evaluation
    # Si on ne reconnait pas la donnee, on souleve une exception
    else:
        raise Exception('Les donnees ' + str(data_type) + 'ne sont pas disponibles')
    # On ajoute un symbole EOS a la fin de la phrase
    sentence = string_to_int(list(sentence.strip()) + ['_'])
    # On retourne la phrase deux fois pour designer l'entree et la cible
    return sentence, sentence

def getN_LMData(n, data_type='training'):
    """
    n: nombre d'exemples a prendre
    data_type: training ou evaluation
    Retourne une liste de donnees prete a etre utilisee dans le reseau de neurones
    """
    # On initialise les donnees
    init_data()
    d = []
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        data = data_training
    elif data_type == 'evaluation':
        data = data_evaluation
    # On itere n fois
    for _ in  range(n):
        # On recupere dans la donnee de facon aleatoire
        data_id, sentence = random.choice(data)
        # On recupere le couple grace a la fonction __get_LMData et des donnees trouvees dans la liste
        d.append(__get_LMData(data_type, data_id, sentence))
    # On retourne enfin les donnees au format attendu par notre reseau de neurones
    d = np.concatenate(d, axis=1)
    inputs = d[0][:-1]
    targets = d[1][1:]
    return np.expand_dims(np.array([inputs,targets]), axis=2)

def getAll_LMData(n, data_type='training'):
    """
    n: nombre d'exemples a prendre par batch
    data_type: training ou evaluation
    Retourne un generateur de donnees prete a etre utilisees dans le reseau de neurones
    L'ensemble des donnees sont parcourues
    """
    # On initialise les donnees
    init_data()
    # On choisit le dossier correspond aux donnees qu'on souhaite ouvrir
    if data_type == 'training':
        data = data_training
    elif data_type == 'evaluation':
        data = data_evaluation
    # On melange notre dossier afin de pouvoir parcourir la liste des donnees dans un ordre distinct a chaque rappel de la fonction
    random.shuffle(data)
    # On initialise i a 0 et d a vide
    i = 0
    d = []
    # On parcourt l'ensemble des donnees disponibles
    for data_id, sentence in data:
        # Si i est inferieur a n, cela veut dire qu'il faut encore remplir le batch
        if i < n:
            # on ajoute la donnee au batch et on incremente i de 1
            d.append(__get_LMData(data_type, data_id, sentence))
            i += 1
        else:
            # Si i >= n, on renvoie d car le batch est a la taille souhaitee
            d = np.concatenate(d, axis=1)
            inputs = d[0][:-1]
            targets = d[1][1:]
            yield np.expand_dims(np.array([inputs,targets]), axis=2)
            # On remet la liste a vide et le compteur i a 0
            d = []
            i = 0
    # A la fin du parcours, si la liste d n'est pas vide, cela veut dire qu'un batch n'etait pas complet mais ne peut etre rempli pour le completer
    if d:
        # On renvoie donc ce dernier batch
        d = np.concatenate(d, axis=1)
        inputs = d[0][:-1]
        targets = d[1][1:]
        yield np.expand_dims(np.array([inputs,targets]), axis=2)


# Si on demarre le fichier directement
if __name__ == '__main__':
    (inputs, targets) = getN_LMData(2)
    #print(int_to_string(get_NAMData(2)[1][1]))