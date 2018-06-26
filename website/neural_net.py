# neural_net.py

### neural network processing modules for Northern Flights



import json
import os
import numpy as np
import csv
import geopy
from geopy.distance import great_circle
import matplotlib.pylab as plt
import pandas as pd

import pickle




def load_neural_network_data(prefix):
	model = pickle.load(open('../'+prefix+'_network.sav', 'rb'))
	encoder = pickle.load(open('../'+prefix+'_encoder.sav', 'rb'))
	scaler = pickle.load(open('../'+prefix+'_scaler.sav', 'rb'))
	kmeans = pickle.load(open('../'+prefix+'_kmeans.sav', 'rb'))
	df = pickle.load(open('../'+prefix+'_df.sav', 'rb'))
	return model,encoder,scaler,kmeans,df
	


 


def predict_airfares_kmeans(origin_lat,origin_lon,destination_lat,destination_lon,encoder,scaler,reg,kmeans,N_days=30,day_max=450):
    #### route = 'MUC-YVR', for example
    #origin = route[:3]
    #destination = route[-3:]
    #print(origin)
    #print(destination)
    #with open('airports.csv') as csvfile:
    #    reader = csv.DictReader(csvfile)
    #    origin_lat,origin_lon,_ = match_iata_code(origin,reader)
    #with open('airports.csv') as csvfile:
    #    reader = csv.DictReader(csvfile)
    #    destination_lat,destination_lon,_ = match_iata_code(destination,reader)
    #print(origin_lat)
    c1 = (origin_lat,origin_lon)
    c2 = (destination_lat,destination_lon)
    gcd = great_circle(c1,c2).miles
    dts = np.linspace(0,day_max,N_days)
    prices = np.zeros(N_days)
    month_num = np.zeros(N_days)
    year = np.zeros(N_days)
    date_code = np.zeros(N_days)
    from datetime import datetime, timedelta, date, time
    for i in range(N_days):
    	date_today = date.today() +timedelta(days=dts[i])
    	month_num[i] = date_today.month
    	year[i] = date_today.year
    	date_code[i] = date_today.year+date_today.month/12.0 +date_today.day/365.0
    X_extrapolate = pd.DataFrame({'Origin lat' : [np.float(origin_lat) for i in range(N_days)],        #0
                       'Origin lon' : [np.float(origin_lon) for i in range(N_days)],
                       'Destination lat' : [np.float(destination_lat) for i in range(N_days)],
                       'Destination lon' : [np.float(destination_lon) for i in range(N_days)],
                       'distance' : [gcd for i in range(N_days)],
                       'year' : [year[i] for i in range(N_days)],
                       'month' : [month_num[i] for i in range(N_days)],
                        'date' : [date_code[i] for i in range(N_days)]})
    X_extrapolate['Origin lon'] = X_extrapolate['Origin lon']+180.0
    X_extrapolate['Destination lon'] = X_extrapolate['Destination lon']+180.0
    X_extrapolate['Origin lat'] = X_extrapolate['Origin lat']+90.0
    X_extrapolate['Destination lat'] = X_extrapolate['Destination lat']+90.0
    #print(X_extrapolate[['Origin lon','Origin lat']])
    #print(kmeans.cluster_centers_)
    kmeans_origin = kmeans.predict(X_extrapolate[['Origin lon','Origin lat']])    
    kmeans_destination = kmeans.predict(X_extrapolate[['Destination lon','Destination lat']])
    d = {'orgs': kmeans_origin, 'dests': kmeans_destination}
    dt = pd.DataFrame(data=d)
    #print(dt)
    X_onehot = encoder.transform(dt)
    X_onehot_pd = pd.DataFrame(X_onehot.todense())

    nothot_X = pd.concat([X_extrapolate['month'],X_extrapolate['distance'],X_extrapolate['year'],X_extrapolate['date']],axis=1)
    X_nothot = scaler.transform(nothot_X)
    X_nothot_pd = pd.DataFrame(X_nothot)

    X_new = pd.concat([X_onehot_pd,X_nothot_pd],axis=1)
    #print(X_new)
    Y_new = reg.predict(X_new)
    return Y_new,dts
    

