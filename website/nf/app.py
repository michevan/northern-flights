    
from flask import Flask,render_template, request, redirect
from datetime import datetime, timedelta, date, time
import time

import json
import numpy as np
import pandas as pd


import matplotlib
matplotlib.use('Agg')


import northern_flights as nf
import great_circle_routines as gc
import neural_net as nn
from scipy.special import expit    


from flask import abort







app = Flask(__name__)

@app.route('/')
def main():
   return render_template('input.html')
#def main():
#	return "<form action='/results' method='post'><input type='text' name='data'><input type='submit' value='Airport Code'>"

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST' :
    	start_time = time.time()
    	arctic_circle_lat = 64.0   #it's actually 66.3 degrees, but give a bit of wiggle room
    	route_dl = 50   #km, the spatial resolution of the great circle geometry
    	airport_code = request.form['airport']   
    	slider_value = request.form['value_slider']
    	print(slider_value)
    	current_date = time.strftime('%m/%d/%Y')    #make date format a constant
    	sp_api = nf.SkyPickerApi()
    	sp_results = sp_api.search_flights_from_airport(airport_code, datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=7),datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=14))
    	dest_iata_codes = []
    	for i in range(len(sp_results)):
    		dest_iata_code = sp_results[i]['legs'][0]['to'][-4:-1]
    		dest_iata_codes.append(dest_iata_code)
    	
    	
    	unique_dest_iata_codes = set(dest_iata_codes)
    	sp_loc_api = nf.SkyPickerApi()
    	unique_dest_lons = []
    	unique_dest_lats = []
    	elapsed_time = time.time() - start_time
    	print('time to search flights from airport: '+str(elapsed_time))
    	start_time = time.time()
    	lat_dict = nf.load_airport_lat_dictionary()
    	lon_dict = nf.load_airport_lon_dictionary()
    	for iata_code in unique_dest_iata_codes:
    		unique_dest_lats.append(lat_dict[iata_code])
    		unique_dest_lons.append(lon_dict[iata_code])
    	origin_lat = lat_dict[airport_code]
    	origin_lon = lon_dict[airport_code]	
    	elapsed_time = time.time() - start_time
    	print('time to get airport coordinates: '+str(elapsed_time))
    	start_time = time.time()
    	dest_airports = pd.DataFrame(data=np.transpose([list(unique_dest_iata_codes),unique_dest_lons,unique_dest_lats]),columns=['code','lon','lat'])
    	if origin_lat > arctic_circle_lat:
    		does_this_route_pass_through_arctic = np.ones(len(dest_airports))
    	else:
    		does_this_route_pass_through_arctic = np.zeros(len(dest_airports))
    		for i in range(len(dest_airports)):
    			folderpath = 'gis_temp/'
    			layername = airport_code+'-'+dest_airports.loc[i].code
    			gtg = gc.GeodesicLine2Gisfile()
    			lons_lats = (origin_lon, origin_lat, np.float(dest_airports.loc[i].lon), np.float(dest_airports.loc[i].lat))
    			cd = gtg.gdlComp(lons_lats, km_pts=route_dl)
    			gtg.gdlToGisFile(cd, folderpath, layername, fmt="GeoJSON")  # geojson output
    			gc_lat = []; gc_lon = []
    			with open(folderpath+'/'+layername+'.geojson') as json_file:  
    				data = json.load(json_file)
    				if len(data['features'][0]['geometry']['coordinates'][0]) == 2:   #if everything is working as expected
    					for j in range(len(data['features'][0]['geometry']['coordinates'][:])):
    						lon,lat = data['features'][0]['geometry']['coordinates'][j]
    						gc_lat.append(lat)
    						gc_lon.append(lon)
    				else:    #anti-meridian crossing, they split the results into two groups so the len will be a lot longer
    					for j in range(len(data['features'][0]['geometry']['coordinates'][0][:])):
    						lon,lat = data['features'][0]['geometry']['coordinates'][0][j]
    						gc_lat.append(lat)
    						gc_lon.append(lon)
    					for j in range(len(data['features'][0]['geometry']['coordinates'][1][:])):
    						lon,lat = data['features'][0]['geometry']['coordinates'][1][j]
    						gc_lat.append(lat)
    						gc_lon.append(lon)
    			if np.max(gc_lat) > arctic_circle_lat:
    				does_this_route_pass_through_arctic[i] = 1
    	elapsed_time = time.time() - start_time
    	
    	if does_this_route_pass_through_arctic.sum() == 0:
    		abort(400, 'No direct flights found from this airport. Please try another airport.')

    		
    	print('time to get flight routes: '+str(elapsed_time))
    	start_time = time.time()
		
    	figdata_path = nf.make_plot(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code)
    	    	
    	elapsed_time = time.time() - start_time
    	print('time to make the globe plot: '+str(elapsed_time))
    	start_time = time.time()
	
    	#plot_html = make_plot2(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code)

    	final_airport_codes = []
    	final_airport_lats = []
    	final_airport_lons = []
    	for i in range(len(does_this_route_pass_through_arctic)):
    		if does_this_route_pass_through_arctic[i]:
    			final_airport_codes.append(airport_code+'-'+dest_airports.loc[i].code)
    			final_airport_lats.append(dest_airports.loc[i].lat)
    			final_airport_lons.append(dest_airports.loc[i].lon)
    	
    			
    	### make a giant dataframe that contains all the routes which pass through the arctic
		### and it also contains the flight path, sampled every 10 km
		### in both (lat,lon) and in magnetic (lat,lon)
    	
    	N_routes = 0     #Number of routes, this will get filled in by the code
    	routes = pd.DataFrame(data=None)
    	for i in range(len(does_this_route_pass_through_arctic)):
    		df = nf.get_flight_route(i,does_this_route_pass_through_arctic,airport_code,dest_airports)
    		if df is not None:
    			routes = routes.append(df)
    			N_routes += 1
    	elapsed_time = time.time() - start_time
    	print('time to get the final flight routes prepped: '+str(elapsed_time))
    	start_time = time.time()

    	### load aurora model fit to NOAA data, defined in make_aurora_model.ipynb
    	### strictly valid until Jan 4, 2019   (end of current Solar cycle)
    	### and extrapolated out to Jan 4, 2021 assuming the next Solar cycle begins like the current one
    	x_model,bf_model = np.loadtxt('../../aurora_model.dat',skiprows=1,usecols=[0,1],unpack=True)  
    	
    	### aurora prob lookup table (from https://www.ngdc.noaa.gov/stp/geomag/kp_ap.html)
    	### to convert ap values into kp values
    	kp_lookup = np.array([0,0.33,0.67,1.0,1.33,1.67,2.0,2.33,2.67,3.0,3.33,3.67,4.0,4.33,4.67,5.0,5.33,5.67,6.0,6.33,6.67])
    	ap_lookup = np.array([0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,48,56,67,80,94,111])
    	kp_model = np.interp(bf_model,ap_lookup,kp_lookup)
    	
    	elapsed_time = time.time() - start_time
    	print('time to load precomputed aurora data '+str(elapsed_time))
    	start_time = time.time()

    	N_days = 26
    	aurora_p = np.zeros([N_days,N_routes])
    	for i in range(N_routes):
    		p,dts = nf.get_aurora_prob_over_time(routes.iloc[4*i][1:].values,routes.iloc[4*i+1][1:].values,routes.iloc[4*i+2][1:].values,routes.iloc[4*i+3][1:].values,x_model,kp_model,route_dl,N_days=N_days,day_max=500)
    		aurora_p[:,i] = p
    	elapsed_time = time.time() - start_time
    	print('time to estimate aurora probabilities: '+str(elapsed_time))
    	start_time = time.time()


    	aurora_plot_path = nf.make_aurora_plot(N_routes,dts,aurora_p,final_airport_codes)
    	elapsed_time = time.time() - start_time
    	print('time to make aurora plot: '+str(elapsed_time))
    	start_time = time.time()
    	
    	nn_model,encoder,scaler,kmeans,nn_df = nn.load_neural_network_data('k_interp')
    	prices = np.zeros([N_days,N_routes])
    	for i in range(N_routes):
    		pr,dts = nn.predict_airfares_kmeans(origin_lat,origin_lon,final_airport_lats[i],final_airport_lons[i],encoder,scaler,nn_model,kmeans,N_days=N_days,day_max=500)
    		prices[:,i] = pr
    	#print(prices)	
    	
    	elapsed_time = time.time() - start_time
    	print('time to predict flight prices with neural network: '+str(elapsed_time))
    	start_time = time.time()
		
    	price_plot_path = nf.make_prices_plot(N_routes,dts,prices,final_airport_codes,nn_df)	

    	elapsed_time = time.time() - start_time
    	print('time to make flight prices plot: '+str(elapsed_time))
    	start_time = time.time()
    	
    	best_aurora = np.where(aurora_p == aurora_p.max())
    	#prices_temp = prices[:,best_aurora[1]]
    	#prices_reshape = np.hstack([prices_temp[i,0] for i in range(prices_temp.shape[0])])
    	#best_aurora_price = np.interp(dts[best_aurora[0]][0], dates_dt, prices_reshape)
    	best_aurora_price = prices[best_aurora][0]
    	best_aurora_month = (datetime.today() + timedelta(days=dts[best_aurora[0]][0])).month
    	best_aurora_year = (datetime.today() + timedelta(days=dts[best_aurora[0]][0])).year
    	best_aurora_text = final_airport_codes[best_aurora[1][0]]
    	best_aurora_text +=' in '+str(nf.NumtoMonth(np.int(best_aurora_month)))+' '+str(best_aurora_year)
    	best_aurora_text +='. Estimated price ~  $%.f' % (best_aurora_price)
    	best_aurora_text +=', and estimated aurora probability = %0.2f'  % (aurora_p[best_aurora][0])
    	
    	best_price = np.where(prices == prices.min())
    	#auroras_temp = aurora_p[:,best_price[1]]
    	#auroras_reshape =  np.hstack([auroras_temp[i,0] for i in range(auroras_temp.shape[0])])
		
    	#best_price_aurora = np.interp(dates_dt[best_price[0]][0], dts , auroras_reshape)
    	best_price_aurora = aurora_p[best_price][0]
    	best_price_month = (datetime.today() + timedelta(days=dts[best_price[0]][0])).month
    	best_price_year = (datetime.today() + timedelta(days=dts[best_price[0]][0])).year
    	best_price_text = final_airport_codes[best_price[1][0]]
    	best_price_text +=' in '+str(nf.NumtoMonth(np.int(best_price_month)))+' '+str(best_price_year)
    	best_price_text +='. Estimated price ~  $%.f' % (prices[best_price][0])
    	best_price_text +=', and estimated aurora probability = %0.2f'  % (best_price_aurora)
    	
    	### use sigmoid function to translate slider value to weights
    	aurora_weight = expit(-1.0*np.float(slider_value))
    	price_weight = expit(np.float(slider_value))
    	
    	figure_of_merit =   (aurora_p * aurora_weight) +    (prices.min() / prices)*price_weight * aurora_p.max()
    	#figure_of_merit =   (aurora_p * aurora_weight) +    ((prices.max() - prices)/(prices.max()-prices.min()))*price_weight
    	best_overall = np.where(figure_of_merit == figure_of_merit.max())
    	best_aurora_overall = aurora_p[best_overall][0]
    	best_price_overall = prices[best_overall][0]
    	best_overall_month = (datetime.today() + timedelta(days=dts[best_overall[0]][0])).month
    	best_overall_year = (datetime.today() + timedelta(days=dts[best_overall[0]][0])).year
    	best_overall_text = final_airport_codes[best_overall[1][0]]
    	best_overall_text +=' in '+str(nf.NumtoMonth(np.int(best_overall_month)))+' '+str(best_overall_year)
    	best_overall_text +='. Estimated price ~  $%.f' % (best_price_overall)
    	best_overall_text +=', and estimated aurora probability = %0.2f'  % (best_aurora_overall)
    	
    	#best_total = np.where(figure_of_merit == figure_of_merit.max())

    	
    	elapsed_time = time.time() - start_time
 
    	print('time to compute best values: '+str(elapsed_time))
    	start_time = time.time()

    	print('time to finish code: '+str(elapsed_time))
    	#print(best_aurora_text)
    	#print(best_price_text)
    	#print(best_overall_text)
    	#print(figure_of_merit.max())

    return render_template('result.html',plot_code=figdata_path,aurora_plot_code=aurora_plot_path,price_plot_code=price_plot_path,best_aurora_text=best_aurora_text,best_price_text=best_price_text,best_overall_text=best_overall_text)


    
    

if __name__ == '__main__':
        app.run(debug=True)