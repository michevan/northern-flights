    
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
    










app = Flask(__name__)

@app.route('/')
def main():
   return render_template('input.html')
#def main():
#	return "<form action='/results' method='post'><input type='text' name='data'><input type='submit' value='Airport Code'>"

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST' :
    	arctic_circle_lat = 64.0   #it's actually 66.3 degrees, but give a bit of wiggle room
    	route_dl = 50   #km, the spatial resolution of the great circle geometry
    	airport_code = request.form['airport']   
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
    	for iata_code in unique_dest_iata_codes:
    		lat,lon = nf.get_airport_coords(sp_api,iata_code)    #use a local file instead
    		unique_dest_lats.append(lat)
    		unique_dest_lons.append(lon)
    		
    	origin_lat,origin_lon = nf.get_airport_coords(sp_api,airport_code)
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
    				
    	figdata_path = nf.make_plot(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code)
    	    	
    	#plot_html = make_plot2(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code)

    	final_airport_codes = []
    	for i in range(len(does_this_route_pass_through_arctic)):
    		if does_this_route_pass_through_arctic[i]:
    			final_airport_codes.append(airport_code+'-'+dest_airports.loc[i].code)
    			
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
    			
        ### put the aurora data into pandas dataframes
        ### I will actually only use the ap data
        ### and I think there is something wrong with how I am parsing the kp data		
    	kp,ap = nf.get_aurora_data()        
    	
    	### smooth the ap data
    	ap_smooth = ap.rolling(240).mean()  #3-day smoothing
    	
    	### fit a 4th-order function to the data for the current Solar cycle
    	cycle_begin = 2008.01   #Jan 4, 2008
    	cycle_end = 11.0 + cycle_begin
    	x = np.array(ap.index)
    	y = ap_smooth.values
    	filter = np.where((x > cycle_begin)*(x<cycle_end))
    	y = y[filter]
    	x = x[filter]
    	z = np.polyfit(x-2013., y, 4)
    	p = np.poly1d(z)
    	x_model = np.linspace(cycle_begin,cycle_end+0.6,1001)
    	bf_model = p(x_model-2013.)
    	
    	
    	### aurora prob lookup table (from https://www.ngdc.noaa.gov/stp/geomag/kp_ap.html)
    	### to convert ap values into kp values

    	
    	kp_lookup = np.array([0,0.33,0.67,1.0,1.33,1.67,2.0,2.33,2.67,3.0,3.33,3.67,4.0,4.33,4.67,5.0,5.33,5.67,6.0,6.33,6.67])
    	ap_lookup = np.array([0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,48,56,67,80,94,111])
    	kp_model = np.interp(bf_model,ap_lookup,kp_lookup)
    	
    	N_days = 31
    	aurora_p = np.zeros([N_days,N_routes])
    	for i in range(N_routes):
    		p,dts = nf.get_aurora_prob_over_time(routes.iloc[4*i][1:].values,routes.iloc[4*i+1][1:].values,routes.iloc[4*i+2][1:].values,routes.iloc[4*i+3][1:].values,x_model,kp_model,route_dl)
    		aurora_p[:,i] = p


    	aurora_plot_path = nf.make_aurora_plot(N_routes,dts,aurora_p,final_airport_codes)
    	
    	t_max = 75
    	dates = pd.date_range(pd.datetime.today(), periods=t_max)
    	dates_dt = np.linspace(0,t_max-1,t_max)
    	prices = np.zeros([t_max,N_routes])
    	for i in range(N_routes):
    		#print(airport_code)
    		print(final_airport_codes[i][4:])
    		best_prices_skypicker,dates_skypicker = nf.scrape_skypicker(airport_code,final_airport_codes[i][4:],delta_t = 7,t_max = t_max)
    		prices[:,i] = best_prices_skypicker
    		
    	price_plot_path = nf.make_prices_plot(N_routes,dates_dt,prices,final_airport_codes,dts)	

    	best_aurora = np.where(aurora_p == aurora_p.max())
    	best_price = np.where(prices == prices.min())
    	aurora_value = np.float(request.form['aurora_value'])
    	price_value = np.float(request.form['price_value']   )
    	#figure_of_merit =   (aurora_p / aurora_p.max()) * aurora_value  + (prices.min()/prices) * price_value
    	#best_total = np.where(figure_of_merit == figure_of_merit.max())
    	print(best_aurora)
    	#best_aurora_text = final_airport_codes[best_aurora[1]][4:]+' in '+str(dates[best_aurora[0]])+';  price = '+str(prices[best_aurora])+' p='+str(aurora_p[best_aurora])




    return render_template('result.html',dest_codes=final_airport_codes,plot_code=figdata_path,aurora_plot_code=aurora_plot_path,price_plot_code=price_plot_path)#,best_aurora_text=best_aurora_text)


    
    

if __name__ == '__main__':
        app.run(debug=True)