Le fichier config.yaml règle les paramètres, l'agent DQN est dans le fichier DQN_Agent.py. 

DQN_check_solution est le fichier donné dans le TP

DQN_Test permet de visualiser une simulation étant donné un modèle

DQN_problem permet de lancer un apprentissage (paramètre donnés via config.yaml), il sauvegarde le modele a la fin de la periode d'apprentissage et il peut sauvegarder un modele pendant l'apprentissage si la moyenne des 50 dernieres rewards est >= 50. Les modèles sont sauvés dans models/underway

DQN_check_solution_in_folder_undrway permet d'aller chercher un modèle non testé dans models/underway de la tester et de la mettre dans le dossier approprié (models/solvers ou models/failers selon s'il passe le test ou pas) marche plus depuis que j'ai mis le dueling... Pour le moment tous les modèles dans solvers sont des modèles qui utilisent pas dueling et pas cer.

DQN_check_solution_random_agent sert juste pour avoir le -180 de la question a.