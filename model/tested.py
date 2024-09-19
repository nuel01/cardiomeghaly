# -*- coding: utf-8 -*-

#Spyder Editor

from sklearn.externals import joblib

loaded_model=joblib.load('piped.pkl')

print(loaded_model.predict(['cardiac silhouette is normal in size and contour. CTR of 0.47']))

