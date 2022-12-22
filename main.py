#
# Ogni utente deve avere una propria cartella dentro dataset
#  - per ognuno dei 10 gesti devono essere fatti due video: uno con la destra e uno con la sinistra
#    dalla durata di almeno cinque secondi per un totale di 20 video dentro ogni cartella
#  - ogni video deve essere rinominato come segue
#       {gesto}_{mano}.mp4
#    ad esempio
#       thumbUp_right.mp4

from dataset.scripts.create_dataset import *
from dataset.scripts.live_camera import *

record_live('daniele')

#create_dataset('damiano', 'marianna', 'daniele', 'daniele_zio', 'donatella', flip_horizontally=True)
#create_dataset('anna')
