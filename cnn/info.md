# info over onze ai

## nutige links

* dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data
* dataset source: http://vision.stanford.edu/aditya86/ImageNetDogs/
* dataset met leeftijden categorieën: https://www.kaggle.com/competitions/petfinder-adoption-prediction/data
* dataset met geluiden van honden in verschillende scenario's: https://drive.google.com/drive/folders/1OGSTJEPEYiWBsKTquPRth1H_ul6HKH9G -> Bark, Yip, Howl, Bow-wow, Growling, Whimper (dog)
* dataset uitleg over hondengeluiden: https://research.google.com/audioset///////ontology/dog_1.html
* cnn tutorial: https://www.tensorflow.org/tutorials/images/cnn

## algemene verduidelijkingen

De cnn is de python script voor het trainen en testen van de cnn. Zorg dat je elke lijn snapt, er staat wat commentaar bij. In app.py vindt men het python script dat het model inlaadt en een voorspelling doet op de afbeelding. De index.html is het html bestand waar men een foto kan uploaden en de voorspelling te zien krijgt. Dit werkt met ```flask run```. Dat is een python web framework, dat men dan ook runt in de environment.

Kies zelf welke honden je wil herkennen, je kan voor alles kiezen maar vraagt meer tijd van je pc.

Het gokken van de leeftijdscategorie werkt. Een extra is geluiden van honden herkennen, dat wekrt ook.