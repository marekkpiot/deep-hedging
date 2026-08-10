# Deep Hedging

Projet pédagogique de couverture d'une option européenne avec un réseau de neurones entraîné sous PyTorch.

L'objectif est de comparer deux méthodes de couverture :

- le delta-hedging classique de Black-Scholes ;
- une stratégie de deep hedging apprise par un réseau de neurones.

Le réseau n'apprend pas directement la formule du delta Black-Scholes. Il apprend uniquement à choisir une position de couverture afin de réduire le risque du PnL final.

---

## Objectif du projet

On considère une banque ayant vendu un call européen.

À l'échéance, elle doit verser au détenteur du call :

```text
payoff = max(S_T - K, 0)
```

où :

```text
S_T = prix de l'action à l'échéance
K   = strike de l'option
```

Si l'action monte fortement, le payoff peut devenir important.

La banque cherche donc à réduire ce risque en achetant ou vendant régulièrement des actions.

Le problème consiste à déterminer :

```text
Combien d'actions faut-il détenir
à chaque date pour couvrir le call ?
```

Deux méthodes sont étudiées :

```text
Black-Scholes
→ utilise une formule analytique pour calculer le delta

Deep Hedging
→ utilise un réseau de neurones entraîné sur des trajectoires simulées
```

---

# Structure du projet

```text
deep-hedging/
├── figures/
│   ├── gbm_paths.png
│   ├── untrained_network_pnl.png
│   ├── training_loss.png
│   └── pnl_comparison.png
│
├── models/
│   └── hedging_network.pt
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── delta_hedging.py
│   ├── gbm.py
│   ├── hedging.py
│   ├── hedging_network.py
│   └── losses.py
│
├── torch_check.py
├── gbm_demo.py
├── network_demo.py
├── hedging_pnl_demo.py
├── train_network.py
├── compare_hedging.py
├── requirements.txt
├── .gitignore
└── README.md
```

Le fichier `hedging_network.pt` contient les paramètres appris du réseau. Il peut être exclu du dépôt GitHub si les modèles entraînés sont ignorés dans `.gitignore`.

---

# Installation

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

Sous PowerShell, il peut être nécessaire d'autoriser temporairement l'exécution des scripts :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Activer ensuite l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

Les principales bibliothèques utilisées sont :

```text
PyTorch
NumPy
Matplotlib
SciPy
```

---

# Ordre d'exécution

Les différents scripts peuvent être exécutés dans l'ordre suivant :

```powershell
python torch_check.py
python gbm_demo.py
python network_demo.py
python hedging_pnl_demo.py
python train_network.py
python compare_hedging.py
```

---

# 1. Simulation du prix de l'action

Les trajectoires de l'action sont simulées avec un mouvement brownien géométrique.

La discrétisation utilisée est :

```text
S_(t+dt)
=
S_t
×
exp(
    (r - 0.5 × sigma²) × dt
    +
    sigma × sqrt(dt) × Z
)
```

avec :

```text
S_t   = prix actuel
r     = taux sans risque
sigma = volatilité
dt    = durée d'un pas temporel
Z     = variable normale N(0,1)
```

Le code correspondant se trouve dans :

```text
src/gbm.py
```

PyTorch permet de simuler simultanément plusieurs milliers de trajectoires.

Un tenseur contenant :

```text
5000 trajectoires
31 dates
```

possède par exemple la forme :

```text
torch.Size([5000, 31])
```

Chaque ligne correspond à une trajectoire et chaque colonne à une date.

Le graphique des trajectoires simulées est enregistré dans :

```text
figures/gbm_paths.png
```

![Trajectoires GBM](figures/gbm_paths.png)

---

# 2. Prix du call Black-Scholes

La banque vend initialement le call et reçoit sa prime.

Le prix théorique du call est calculé avec la formule de Black-Scholes dans :

```text
src/black_scholes.py
```

Sous la probabilité risque-neutre :

```text
prix du call aujourd'hui
=
payoff moyen risque-neutre actualisé
```

Autrement dit :

```text
prix du call
=
exp(-rT)
×
espérance risque-neutre du payoff
```

Les trajectoires utilisées dans le projet sont donc simulées avec une dérive égale au taux sans risque.

---

# 3. Principe du hedging

Une banque ayant vendu un call perd lorsque la valeur du call augmente.

Pour réduire ce risque, elle achète des actions.

Par exemple :

```text
Delta du call = 0,60

L'action monte de 1 €
→ le call augmente approximativement de 0,60 €

La banque possède 0,60 action
→ les actions gagnent approximativement 0,60 €
```

Les deux mouvements se compensent en partie.

La position doit cependant être régulièrement réajustée car le delta évolue avec :

```text
le prix de l'action
le temps restant
la volatilité
le strike
```

---

# 4. Portefeuille de couverture

Le portefeuille de la banque contient :

```text
un compte en espèces
+
une position en actions
-
le call vendu
```

Lorsqu'elle modifie sa couverture :

```text
nombre d'actions achetées
=
nouvelle position - ancienne position
```

Le coût du rééquilibrage est :

```text
coût
=
nombre d'actions achetées × prix actuel
```

Le compte en espèces est mis à jour à chaque date.

S'il est positif, il produit des intérêts.

S'il est négatif, il représente une dette et génère un coût d'emprunt.

---

# 5. PnL final

Le PnL utilisé dans l'entraînement est calculé uniquement à l'échéance.

Pendant la trajectoire, le programme met à jour :

```text
la position en actions
le cash
les achats et ventes
les intérêts
```

À l'échéance :

```text
richesse finale
=
cash final
+
position finale × prix terminal
```

Le call doit ensuite être payé :

```text
PnL final
=
richesse finale
-
payoff du call
```

Chaque trajectoire produit donc un seul PnL final.

Avec :

```text
10 000 trajectoires
```

on obtient :

```text
10 000 PnL finaux
```

---

# 6. Réseau de neurones

Le réseau est défini dans :

```text
src/hedging_network.py
```

Il reçoit trois informations :

```text
1. prix de l'action / strike
2. proportion de temps restant
3. position de couverture actuelle
```

La première variable correspond au moneyness :

```text
moneyness = S / K
```

Exemples :

```text
S / K < 1
→ action sous le strike

S / K = 1
→ action au strike

S / K > 1
→ action au-dessus du strike
```

Le réseau produit une seule sortie :

```text
nouvelle position de couverture
```

Une fonction sigmoid limite cette position entre 0 et 1.

---

# 7. Architecture du réseau

L'architecture utilisée est :

```text
3 entrées
    ↓
Linear : 3 → 16
    ↓
ReLU
    ↓
Linear : 16 → 16
    ↓
ReLU
    ↓
Linear : 16 → 1
    ↓
Sigmoid
    ↓
position de couverture
```

Le réseau contient 353 paramètres ajustables.

Ces paramètres sont constitués des poids et des biais des différentes couches.

Au début de l'entraînement, ils sont initialisés automatiquement.

---

# 8. Fonction ReLU

La fonction ReLU est définie par :

```text
ReLU(x) = max(0, x)
```

Exemples :

```text
ReLU(-2) = 0
ReLU(0)  = 0
ReLU(3)  = 3
```

Elle introduit une non-linéarité dans le réseau.

Sans fonction d'activation non linéaire, plusieurs couches linéaires successives resteraient équivalentes à une seule transformation linéaire.

---

# 9. Fonction sigmoid

La dernière couche utilise une sigmoid.

Elle transforme n'importe quelle valeur en un nombre compris entre 0 et 1.

```text
entrée très négative
→ sortie proche de 0

entrée proche de 0
→ sortie proche de 0,5

entrée très positive
→ sortie proche de 1
```

La sortie peut donc être directement interprétée comme une quantité d'actions à détenir.

---

# 10. Deep Hedging

À chaque date, le réseau reçoit :

```text
[S_t / K, temps restant, position actuelle]
```

et renvoie :

```text
nouvelle position
```

Le même réseau est utilisé à toutes les dates.

Il n'existe donc pas un réseau différent pour chaque étape temporelle.

Le processus est :

```text
état actuel
      ↓
réseau
      ↓
nouvelle position
      ↓
achat ou vente d'actions
      ↓
mise à jour du cash
      ↓
date suivante
```

Ce processus est répété jusqu'à l'échéance.

---

# 11. Fonction de perte

Le réseau est entraîné pour réduire la dispersion du PnL final.

La loss utilisée est la variance du PnL :

```text
PnL moyen
=
moyenne des PnL

PnL centré
=
PnL - PnL moyen

loss
=
moyenne des PnL centrés au carré
```

Une faible variance signifie que les résultats du portefeuille sont plus concentrés.

Le PnL moyen est également surveillé afin d'éviter qu'une faible variance masque un biais important.

La fonction de perte se trouve dans :

```text
src/losses.py
```

---

# 12. Descente de gradient

L'entraînement suit le processus :

```text
poids du réseau
       ↓
positions de couverture
       ↓
PnL final
       ↓
loss
       ↓
calcul des gradients
       ↓
modification des poids
```

Le cœur de l'entraînement PyTorch est :

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

## `zero_grad`

Efface les gradients calculés lors de l'étape précédente.

## `backward`

Effectue la rétropropagation.

PyTorch calcule automatiquement l'influence de chaque paramètre du réseau sur la loss.

## `step`

L'optimizer modifie les paramètres à partir des gradients obtenus.

---

# 13. Optimizer Adam

Le projet utilise Adam :

```text
learning rate = 0,001
```

Adam est une méthode d'optimisation basée sur la descente de gradient.

Il adapte les mises à jour en tenant compte de l'historique récent des gradients.

Le principe général reste :

```text
paramètre
=
paramètre
-
correction déterminée à partir du gradient
```

---

# 14. Entraînement

Le réseau est entraîné avec de nouveaux scénarios simulés à chaque itération.

Une configuration utilisée est par exemple :

```text
batch size = 2048 trajectoires
nombre d'itérations = 3000
```

Cela représente plus de six millions de trajectoires présentées au réseau au cours de l'entraînement.

Elles ne sont pas toutes stockées simultanément :

```text
génération d'un batch
→ entraînement
→ suppression du batch
→ nouveau batch
```

Le graphique de convergence est enregistré dans :

```text
figures/training_loss.png
```

![Loss d'entraînement](figures/training_loss.png)

On cherche à observer une diminution puis une stabilisation de la loss de validation.

---

# 15. Training, validation et test

Trois types de trajectoires sont distingués.

## Training

Les trajectoires d'entraînement servent à :

```text
calculer la loss
calculer les gradients
modifier les paramètres
```

## Validation

Des trajectoires fixes servent à suivre les performances pendant l'entraînement.

Elles ne sont jamais utilisées pour modifier directement les poids.

Elles permettent de vérifier que le réseau améliore réellement sa capacité de généralisation.

## Test

Un troisième ensemble de trajectoires est utilisé uniquement pour la comparaison finale.

```text
train
→ apprentissage

validation
→ contrôle pendant l'apprentissage

test
→ évaluation finale
```

---

# 16. Sauvegarde du réseau

Après l'entraînement, les paramètres appris sont sauvegardés dans :

```text
models/hedging_network.pt
```

Le checkpoint contient notamment :

```text
les poids et biais du réseau
la taille des couches
les paramètres du modèle
certaines statistiques de validation
```

Il peut ensuite être chargé sans réentraîner le réseau.

---

# 17. Réseau non entraîné

Avant entraînement, les paramètres du réseau sont essentiellement aléatoires.

Les positions produites ne possèdent donc pas encore de véritable signification financière.

Le projet compare notamment :

```text
PnL sans couverture
PnL avec un réseau non entraîné
```

dans :

```text
figures/untrained_network_pnl.png
```

![Réseau non entraîné](figures/untrained_network_pnl.png)

Cette étape sert à montrer que l'architecture seule ne suffit pas : le réseau doit apprendre à partir de la fonction de perte.

---

# 18. Delta-hedging Black-Scholes

Une stratégie classique est implémentée dans :

```text
src/delta_hedging.py
```

Le delta d'un call Black-Scholes est :

```text
delta = N(d1)
```

où `N` représente la fonction de répartition de la loi normale standard.

À chaque date :

```text
1. le prix actuel est observé
2. le delta Black-Scholes est recalculé
3. la position en actions est ajustée
4. le compte en espèces est mis à jour
```

Le delta donne directement le nombre d'actions à détenir par call vendu.

---

# 19. Pourquoi le delta-hedging n'est-il pas parfait ?

Dans la théorie Black-Scholes, la réplication parfaite suppose une couverture continuellement réajustée.

Dans ce projet, seulement 30 rééquilibrages sont réalisés sur une année.

```text
couverture théorique :
rééquilibrage continu

simulation :
30 rééquilibrages
```

Le prix de l'action peut évoluer entre deux dates alors que la position reste inchangée.

Cela crée une erreur de couverture appelée :

```text
hedging error
```

Le PnL final n'est donc pas exactement nul, même pour le delta Black-Scholes.

---

# 20. Comparaison finale

Le script :

```text
compare_hedging.py
```

compare trois stratégies sur exactement les mêmes trajectoires de test :

```text
1. aucune couverture
2. delta Black-Scholes
3. deep hedging
```

Utiliser les mêmes trajectoires est essentiel pour rendre la comparaison équitable.

Le graphique final est enregistré dans :

```text
figures/pnl_comparison.png
```

![Comparaison des PnL](figures/pnl_comparison.png)

---

# 21. Mesures de performance

Plusieurs mesures sont utilisées pour comparer les distributions de PnL.

## PnL moyen

```text
PnL moyen
=
moyenne des PnL finaux
```

Un PnL moyen proche de zéro est cohérent avec une valorisation risque-neutre correcte.

---

## Écart-type

L'écart-type mesure la dispersion des PnL autour de leur moyenne.

```text
écart-type faible
→ résultats plus concentrés
→ risque de couverture plus faible
```

---

## Variance

La variance est le carré de l'écart-type.

```text
variance
=
moyenne des écarts au PnL moyen au carré
```

C'est également la fonction de perte utilisée pendant l'entraînement.

---

## RMSE

Le RMSE mesure la distance moyenne du PnL par rapport à zéro.

```text
RMSE
=
racine de la moyenne des PnL au carré
```

Contrairement à la variance, le RMSE pénalise également un éventuel PnL moyen éloigné de zéro.

---

# 22. Value at Risk

La VaR permet d'étudier les scénarios défavorables.

Pour une VaR à 95 %, on regarde le 5e percentile de la distribution du PnL.

Exemple :

```text
VaR 95 % = -2,50 €
```

signifie qu'environ 5 % des scénarios produisent un PnL inférieur ou égal à environ `-2,50 €`.

---

# 23. CVaR

La CVaR étudie directement la queue gauche de la distribution.

Elle correspond au PnL moyen parmi les scénarios situés au-delà de la VaR.

```text
CVaR 95 %
=
PnL moyen parmi les 5 % pires scénarios
```

Exemple :

```text
VaR 95 %  = -2,50 €
CVaR 95 % = -3,40 €
```

Cela signifie que, parmi les 5 % de scénarios les plus défavorables, la perte moyenne est d'environ 3,40 €.

Pour notre convention de PnL :

```text
CVaR proche de zéro
→ meilleur résultat

CVaR très négative
→ pertes extrêmes plus importantes
```

La CVaR permet donc d'étudier le risque extrême que la variance seule ne décrit pas complètement.

---

# 24. Résultat principal

Le premier résultat important est que :

```text
aucune couverture
→ PnL très dispersé
```

alors que :

```text
delta Black-Scholes
→ dispersion beaucoup plus faible
```

Cela montre quantitativement l'intérêt d'une couverture dynamique.

Après un entraînement suffisamment long, le réseau de neurones apprend également à réduire fortement la dispersion du PnL.

Sa stratégie se rapproche de la couverture Black-Scholes sans avoir reçu directement la formule du delta.

Le réseau apprend uniquement à partir :

```text
des trajectoires simulées
des positions qu'il choisit
du PnL final
de la fonction de perte
```

---

# 25. Pourquoi le réseau ne bat-il pas nécessairement Black-Scholes ?

Dans ce projet, les trajectoires sont simulées précisément selon les hypothèses de Black-Scholes :

```text
mouvement brownien géométrique
volatilité constante
taux constant
absence de coûts de transaction
marché simplifié
```

Black-Scholes possède déjà une solution analytique particulièrement adaptée à cet environnement.

Il n'y a donc aucune raison de supposer qu'un réseau de neurones doit être meilleur.

Le résultat intéressant est plutôt :

```text
le réseau apprend une stratégie proche
de la solution analytique
sans connaître explicitement cette solution
```

---

# 26. Quand le deep hedging devient-il plus intéressant ?

L'intérêt du deep hedging augmente lorsque le problème devient trop complexe pour disposer d'une solution analytique simple.

Par exemple :

```text
frais de transaction
contraintes sur les positions
liquidité limitée
volatilité stochastique
plusieurs actifs
fonctions de risque asymétriques
instruments complexes
```

Dans ces situations, une politique de couverture peut être directement apprise à partir d'une fonction objectif.

---

# 27. Limites du projet

Ce projet constitue une introduction au deep hedging.

Plusieurs hypothèses sont simplificatrices.

## Modèle Black-Scholes

Les trajectoires sont générées avec un mouvement brownien géométrique.

Dans les marchés réels :

```text
la volatilité n'est pas constante
les rendements ne sont pas parfaitement log-normaux
des sauts peuvent apparaître
les paramètres changent dans le temps
```

---

## Absence de frais de transaction

Les achats et ventes d'actions sont considérés comme gratuits.

Dans la réalité, un rééquilibrage fréquent entraîne des coûts.

Les frais de transaction constituent justement une situation dans laquelle le deep hedging peut devenir particulièrement intéressant.

---

## Liquidité parfaite

Le modèle suppose que toute quantité d'action peut être achetée ou vendue instantanément au prix observé.

Il n'existe ni slippage ni impact de marché.

---

## Même taux d'emprunt et de placement

Le cash positif et la dette sont rémunérés au même taux sans risque.

Cette hypothèse est simplificatrice.

---

## Une seule option

Le projet étudie seulement :

```text
un call européen
sur un seul actif
```

Les portefeuilles réels peuvent contenir de nombreux produits et plusieurs facteurs de risque.

---

## Architecture simple

Le réseau possède seulement :

```text
deux couches cachées
16 neurones par couche
```

L'objectif est pédagogique plutôt que de rechercher une architecture optimale.

---

## Fonction de perte

Le réseau minimise principalement la variance du PnL.

D'autres objectifs pourraient être utilisés :

```text
CVaR
utilité exponentielle
pertes asymétriques
contraintes réglementaires
```

Cela pourrait produire des politiques de couverture différentes.

---

# 28. Améliorations possibles

Plusieurs extensions naturelles sont possibles :

```text
ajouter des frais de transaction
entraîner directement sur la CVaR
utiliser un modèle de Heston
ajouter des contraintes de position
tester plusieurs fréquences de rebalancement
comparer différentes architectures
étudier différentes maturités
étudier différents strikes
ajouter plusieurs actifs
tester le modèle sur des données de marché
```

---

# 29. Compétences travaillées

Ce projet permet de pratiquer :

```text
Python
PyTorch
tenseurs
vectorisation
réseaux de neurones
couches linéaires
ReLU
sigmoid
forward pass
backpropagation
autograd
descente de gradient
optimizer Adam
training / validation / test
simulation Monte Carlo
mouvement brownien géométrique
pricing Black-Scholes
delta hedging
gestion d'un portefeuille de couverture
PnL
variance
RMSE
VaR
CVaR
analyse des distributions
Git
GitHub
```

---

# 30. Conclusion

Ce projet montre qu'un réseau de neurones peut apprendre une stratégie de couverture dynamique à partir d'un objectif financier.

Le réseau ne reçoit pas directement le delta Black-Scholes.

Il observe :

```text
le prix relatif de l'action
le temps restant
la position actuellement détenue
```

et choisit une nouvelle position.

L'entraînement repose uniquement sur le PnL final :

```text
positions
→ portefeuille final
→ PnL
→ loss
→ gradients
→ amélioration des poids
```

Dans un environnement construit selon Black-Scholes, la couverture analytique constitue naturellement une référence très performante.

Après entraînement, le réseau parvient néanmoins à produire une couverture proche de cette référence.

Le projet constitue ainsi une introduction simple au principe du deep hedging et à l'utilisation du deep learning pour des problèmes de finance quantitative.

---

# Avertissement

Ce projet a une vocation exclusivement éducative.

Il ne constitue pas un conseil financier ni une stratégie destinée à être utilisée directement sur les marchés réels.
