version de base :

- ecran de droite: fonctionnalites: -boutons de lancement (audio/video, application partagee, tableau blanc,
covise(actuellement inactif: pour lancer covide, il faut clicquer sur l'icone covise situee dans le panneau
deroulant a la droite de l'ecran))
 						-liste des personnes presentes dans la reunion avec la possibilite de
lancer un chat avec une des personnes en double clicquant sur son nom.
						-l'audio/video s'incruste au-dessus de la liste des personnes. Les boutons
d'options permettent de changer la camera courante, de mettere en pause ou de stopper le son en entree ou en
sortie.
						-une fenetre de controle affiche les commandes permettant de controler
l'application lancee sur l'ecran de gauche.

- ecran de gauche: application partagee ou tableau blanc. La personne qui a lance une application partagee ne peut
plus lancer de tableau blanc jusqu'a ce qu'il ferme son application partagee.

- bug video corrige: son flux est supprime lorsque quelqu'un lance l'application partagee. Il est remis en service
quand l'application partagee est fermee. La video est remplacee par la derniere image generee avant la fermeture
du flux.

- Problemes de passages en premier/ arriere plan pour le lanceur de l'application partagee: corrige en
deplacant la frame java d'un ecran vers la droite et en inversant les 2 ecrans pour garder l'ecran de controle
disponible. D'ou un autre bug: lorsque l'application partagee est stoppee, il y a des problemes de
raffraichissement qui cause l'apparition de l'ecran de controle sur l'ecran de gauche pendant quelques dixiemes de
secondes...

- Qualite du son mediocre du au micro utilise qui est celui de la camera. Il capte toute source sonore se trouvant
aux alentours ce qui cree un bourdonnement en fond sonore.

- Problemes de rafraichissement constate lorqu'on ouvre et que l'on ferme plusieurs fois l'application partagee.
Solution: fermer et relancer l'application partagee.

- La fenetre internet explorer de lancement de l'applet reste en arriere plan lorque l'applet est lancee. Si elle
se retrouve en premier plan, il faut la minimiser et surtout pas la fermer ce qui causerait la fermeture de
l'applet java.

- Actuellement, une fenetre indique l'arrivee d'un nouvel utilisateur dans le meeting mais il n'est pas possible
de le refuser.
