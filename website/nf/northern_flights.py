# northern_flights.py

### support modules for Northern Flights

from datetime import datetime, timedelta, date, time

import time
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ephem
import aacgmv2

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

import io
import base64
import urllib.parse
import requests
import json
import csv


class SkyPickerApi(object):
    """ SkyPicker API. """
    def __init__(self):
        """ Initializes the API object with URL attributes. """
        self.base_url = 'https://api.skypicker.com/'
        self.path = ''
        self.param_str = ''

    @property
    def full_url(self):
        """ Returns the full URL for requesting the data. """
        return '{}{}{}'.format(self.base_url, self.path, self.param_str)

    def get_request(self):
        """ Requests the API endpoint and returns the response """
        headers = {'content-type': 'application/json'}
        resp = requests.get(self.full_url, headers=headers)
        return resp.json()
    

    def search_places(self, place_name, locale=None):
        """ Finds matching place API ids to use for searches.
        :param place_name: string of the place name to search for
        :kwarg locale: two letter lowercase locale string

        returns JSON response
        """
        self.path = 'places'
        self.param_str = '?term={}&partner=picky'.format(place_name)
        if locale:
            self.param_str += '&locale={}'.format(locale)
        return self.get_request()
    
    def search_airports(self, iata_code, locale=None):
        """ gets lon and lat of given airport
        returns JSON response
        """
        self.path = 'locations'
        self.param_str = '?term={}&partner=picky'.format(iata_code)
        if locale:
            self.param_str += '&locale={}'.format(locale)
        return self.get_request()


    def search_flights_from_airport(self, origin, start_date, end_date):
        """ Searches for direct flights given a time range and origin airport code.
        :param origin: string representing the ID or IATA
        :param start_date: datetime representing first possible travel date
        :param end_date: datetime representing last possible travel date

        returns JSON response
        """
        self.path = 'flights'
        self.param_str = '?flyFrom={}&dateFrom={}&dateTo={}&directFlights={}&partner=picky'.format(
                origin, start_date.strftime('%d/%m/%Y'),
                end_date.strftime('%d/%m/%Y'),1)

        resp = self.get_request()
        flights = []
        for flight in resp.get('data'):
            flight_info = {
                'departure': datetime.utcfromtimestamp(flight.get('dTimeUTC')),
                'arrival': datetime.utcfromtimestamp(flight.get('aTimeUTC')),
                'price': flight.get('price'),
                'currency': resp.get('currency'),
                'legs': []
            }
            flight_info['duration'] = flight_info['arrival'] - \
                flight_info['departure']
            flight_info['duration_hours'] = (flight_info[
                'duration'].total_seconds() / 60.0) / 60.0
            for route in flight['route']:
                flight_info['legs'].append({
                    'carrier': route['airline'],
                    'departure': datetime.utcfromtimestamp(
                        route.get('dTimeUTC')),
                    'arrival': datetime.utcfromtimestamp(
                        route.get('aTimeUTC')),
                    'from': '{} ({})'.format(route['cityFrom'],
                                             route['flyFrom']),
                    'to': '{} ({})'.format(route['cityTo'], route['flyTo']),
                })
            flight_info['carrier'] = ', '.join(set([c.get('carrier') for c
                                                    in flight_info['legs']]))
            flights.append(flight_info)
        return flights
        
        
    def search_flights_all(self, origin, destination, start_date, end_date,
                       num_passengers):
        """ Searches for all flights given a time range and origin and destination.
        :param origin: string representing the ID or IATA
        :param destination: string representing the ID or IATA
        :param start_date: datetime representing first possible travel date
        :param end_date: datetime representing last possible travel date
        :param num_passengers: integer

        returns JSON response
        """
        self.path = 'flights'
        self.param_str = '?flyFrom=' + \
            '{}&to={}&dateFrom={}&dateTo={}&passengers={}&curr={}&partner=picky'.format(
                origin, destination, start_date.strftime('%d/%m/%Y'),
                end_date.strftime('%d/%m/%Y'), num_passengers,'USD')
        resp = self.get_request()
        flights = []
        for flight in resp.get('data'):
            flight_info = {
                'departure': datetime.utcfromtimestamp(flight.get('dTimeUTC')),
                'arrival': datetime.utcfromtimestamp(flight.get('aTimeUTC')),
                'price': flight.get('price'),
                'currency': resp.get('currency'),
                'legs': []
            }
            flight_info['duration'] = flight_info['arrival'] - \
                flight_info['departure']
            flight_info['duration_hours'] = (flight_info[
                'duration'].total_seconds() / 60.0) / 60.0
            for route in flight['route']:
                flight_info['legs'].append({
                    'carrier': route['airline'],
                    'departure': datetime.utcfromtimestamp(
                        route.get('dTimeUTC')),
                    'arrival': datetime.utcfromtimestamp(
                        route.get('aTimeUTC')),
                    'from': '{} ({})'.format(route['cityFrom'],
                                             route['flyFrom']),
                    'to': '{} ({})'.format(route['cityTo'], route['flyTo']),
                })
            flight_info['carrier'] = ', '.join(set([c.get('carrier') for c
                                                    in flight_info['legs']]))
            flights.append(flight_info)
        return flights    






def get_airport_coords(sp_api,iata_code):
    ### returns the latitude and longitude of a given airport,
    ### identified by its IATA code
    ### sp_api - an instance of the SkyPicker API
    ### iata_code - an IATA code for an airport
    lat = None; lon = None;  #reset values
    airport_results = sp_api.search_airports(iata_code)
    for airport_result in airport_results['locations']:
        if airport_result['id'] != iata_code:
            continue
        lat = airport_result['location']['lat']
        lon = airport_result['location']['lon']
    return lat,lon
    
    
    
   
def load_airport_lat_dictionary():
	with open('../../airports.csv', 'r') as infile:
		reader = csv.DictReader(infile)
		lat_dict = {rows['iata_code']:np.float(rows['latitude_deg']) for rows in reader}
	return lat_dict
	
def load_airport_lon_dictionary():
	with open('../../airports.csv', 'r') as infile:
		reader = csv.DictReader(infile)
		lon_dict = {rows['iata_code']:np.float(rows['longitude_deg']) for rows in reader}
	return lon_dict
	
def load_airport_name_dictionary():
	with open('../../airports.csv', 'r') as infile:
		reader = csv.DictReader(infile)
		name_dict = {rows['iata_code']:rows['name'] for rows in reader}
	return name_dict
	
        



def make_plot(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code):
	fig=Figure()
	ax=fig.add_subplot(111)
	ax = plt.axes(projection=ccrs.NearsidePerspective(central_latitude=80.0,central_longitude=origin_lon))
	ax.stock_img()
	plt.text(0, 90, '+',horizontalalignment='right',transform=ccrs.Geodetic())
	plt.plot(np.linspace(0,359,360), np.ones(360)*arctic_circle_lat,color='black', linewidth=2, ls='--',transform=ccrs.Geodetic())
	plt.text(origin_lon, origin_lat, airport_code, transform=ccrs.Geodetic())
	N_routes = does_this_route_pass_through_arctic.sum()
	color_list = plt.cm.Set1(np.linspace(0, 1, N_routes))
	route_count = 0
	for i in range(len(dest_airports)):
		if does_this_route_pass_through_arctic[i]:
			plt.plot([origin_lon,np.float(dest_airports.loc[i].lon)],[origin_lat,np.float(dest_airports.loc[i].lat)],color=color_list[route_count],linewidth=2,marker='o',transform=ccrs.Geodetic())
			plt.text(np.float(dest_airports.loc[i].lon),np.float(dest_airports.loc[i].lat),dest_airports.loc[i].code,transform=ccrs.Geodetic())
			route_count += 1
	img = io.BytesIO()  # create the buffer
	plt.savefig(img, format='png')  # save figure to the buffer
	img.seek(0)  # rewind your buffer
	plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
	return plot_data
	

def make_plot2(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code):
	import matplotlib.pyplot as plt, mpld3
	fig, ax = plt.subplots()
	#ax = plt.plot([4,5,6],[6,4,2],'ok')
	ax = plt.axes(projection=ccrs.NearsidePerspective(central_latitude=80.0,central_longitude=origin_lon))
	ax.stock_img()
	ax.text(0, 90, '+',horizontalalignment='right',transform=ccrs.Geodetic())
	ax.plot(np.linspace(0,359,360), np.ones(360)*arctic_circle_lat,color='black', linewidth=2, ls='--',transform=ccrs.Geodetic())
	ax.text(origin_lon, origin_lat, airport_code, transform=ccrs.Geodetic())
	for i in range(len(dest_airports)):
		if does_this_route_pass_through_arctic[i]:
			ax.plot([origin_lon,np.float(dest_airports.loc[i].lon)],[origin_lat,np.float(dest_airports.loc[i].lat)],color=color_list[route_count],linewidth=2,marker='o',transform=ccrs.Geodetic())
			ax.text(np.float(dest_airports.loc[i].lon),np.float(dest_airports.loc[i].lat),dest_airports.loc[i].code,transform=ccrs.Geodetic())
			route_count += 1
	html_text = mpld3.fig_to_html(fig)
	return html_text
	
	
	
	


def get_flight_route(i,does_this_route_pass_through_arctic,airport_code,dest_airports):
    ### returns a pandas dataframe with the flight route info
    ### or returns None if the flight does not pass through the Arctic
    if not does_this_route_pass_through_arctic[i]:
        return None
    folderpath = 'gis_temp/'
    layername = airport_code+'-'+dest_airports.loc[i].code
    mag_lat = [dest_airports.loc[i].code+' mag lat']; mag_lon = [dest_airports.loc[i].code+' mag lon']
    gc_lat = [dest_airports.loc[i].code+' lat']; gc_lon = [dest_airports.loc[i].code+' lon']
    with open(folderpath+'/'+layername+'.geojson') as json_file:  
        data = json.load(json_file)
        if len(data['features'][0]['geometry']['coordinates'][0]) == 2:   #if everything is working as expected
            for j in range(len(data['features'][0]['geometry']['coordinates'][:])):
                lon,lat = data['features'][0]['geometry']['coordinates'][j]
                mg_lat,mg_lon = aacgmv2.convert(lat, lon, 10, date=date.today(), 
                                                  a2g=False, trace=False, allowtrace=False,
                                                  badidea=False, geocentric=False)
                mag_lat.append(mg_lat[0])
                mag_lon.append(mg_lon[0])
                gc_lat.append(lat)
                gc_lon.append(lon)
        else:    #anti-meridian crossing, they split the results into two groups so the len will be a lot longer
            for j in range(len(data['features'][0]['geometry']['coordinates'][0][:])):
                lon,lat = data['features'][0]['geometry']['coordinates'][0][j]
                mg_lat,mg_lon = aacgmv2.convert(lat, lon, 10, date=date.today(), 
                                                  a2g=False, trace=False, allowtrace=False,
                                                  badidea=False, geocentric=False)
                mag_lat.append(mg_lat[0])
                mag_lon.append(mg_lon[0])
                gc_lat.append(lat)
                gc_lon.append(lon)
            for j in range(len(data['features'][0]['geometry']['coordinates'][1][:])):
                lon,lat = data['features'][0]['geometry']['coordinates'][1][j]
                mg_lat,mg_lon = aacgmv2.convert(lat, lon, 10, date=date.today(), 
                                                  a2g=False, trace=False, allowtrace=False,
                                                  badidea=False, geocentric=False)
                mag_lat.append(mg_lat[0])
                mag_lon.append(mg_lon[0])
                gc_lat.append(lat)
                gc_lon.append(lon)
    df = pd.DataFrame.from_records([gc_lat,gc_lon,mag_lat,mag_lon])
    return df
    
    

    

def get_length_of_daytime(lat,lon,dt):
    ### returns the fraction of a 24-hour period which is daytime for a given lat,lon and a given date
    ### lat and lon should be in the form '+85.5','90'
    ### dt is the number of days in the future from today
    from datetime import time
    obs = ephem.Observer()
    obs.pressure = 0
    obs.horizon = '-12'
    obs.lat = lat
    obs.lon = lon
    obs.elevation = 10000
    d = date.today()
    #print(lat)
    #print(lon)
    h = np.int(np.floor(12.0 + np.float(lon)/15.0))
    m = np.int(np.floor(np.mod(np.float(lon)/15.0,60)))
    if h == 24:
    	h = 23
    	m = 59
    t = time(hour=h, minute=m)
    obs.date = datetime.combine(d+timedelta(days=dt), t)
    try:
        t1 = obs.next_setting(ephem.Sun(), use_center=True)
        t2 = obs.previous_rising(ephem.Sun(), use_center=True)
        day_frac = (t1-t2)
        if day_frac > 1:
            day_frac = day_frac - 1
    except ephem.NeverUpError:
        #print('Never Up Error')
        day_frac = 0
    except ephem.AlwaysUpError:
        #print('Always Up Error')
        day_frac = 1
    return day_frac


def get_yearly_daytime_fracs(lat,lon):
    ### returns the dayfracs over a one-year period for a location
    ### lat and lon should be in the form '+85.5','90'
    N = 365
    day_fracs = np.zeros(N)
    for i in range(N):
        day_fracs[i] = get_length_of_daytime(lat,lon,i)
    return day_fracs



#### get the highest magnetic latitude for each route
def get_mag_lat_maxes(routes,N_routes):
    ilocs = 2+np.arange(N_routes)*4
    maxes = np.zeros(N_routes)
    for i in range(N_routes):
        maxes[i] = routes.iloc[ilocs[i]][1:].max()
    return maxes
    

def get_one_day_aurora_prob(lat,lon,mag_lat,mag_lon,dt,mag_lats,aurora_frac_over_route,dl):
    ### for a given flight route (lat,lon,mag_lat,mag_lon) output the probability of seeing 
    ### the aurora. 
    ### mag_lats and kpX_aurora_prob are matching arrays giving the model
    ### probability of seeing the aurora over a 4-hour period at a given mag latitude 
    ### (see get_aurora_prob_over_time for details)
    ### dt = number of days from today
    v_airplane = 800.0  #km/hr
    N_pts = len(lat)
    day_fracs = np.zeros(N_pts)
    for i in range(N_pts):
        if np.isnan(lat[i]):
            continue
        day_fracs[i] = get_length_of_daytime(str(lat[i]),str(lon[i]),dt)
    probs = (1.0-day_fracs) * aurora_frac_over_route * (dl / v_airplane / 4.0)   #4 hour windw
    return 1.0 - np.prod(1.0 - probs)
    



"""
### load up the aurora data which I downloaded from NOAA
### FTP link available at https://www.ngdc.noaa.gov/stp/geomag/kp_ap.html
def get_aurora_data():
    fnames = np.linspace(1932,2018,87).astype(int)
    time_list = []
    kp_list = []
    ap_list = []
    for i in range(87):
        with open('/Users/michevan/Desktop/insight/noaa/'+str(fnames[i]),'r') as file:
            for line in file: 
                year = np.int(fnames[i])
                month = np.int(line[2:4])
                day = np.int(line[4:6])
                mydate = date(year,month,day)
                day_count = mydate.strftime("%j")
                time_list.append(year+np.float(day_count)/365.+0.0625/365)
                kp_list.append(np.int(line[12:14]))
                ap_list.append(np.int(line[31:34]))
                time_list.append(year+np.float(day_count)/365.+0.1875/365)
                kp_list.append(np.int(line[14:16]))
                ap_list.append(np.int(line[34:37]))
                time_list.append(year+np.float(day_count)/365.+0.3125/365)
                kp_list.append(np.int(line[16:18]))
                ap_list.append(np.int(line[37:40]))
                time_list.append(year+np.float(day_count)/365.+0.4375/365)
                kp_list.append(np.int(line[18:20]))
                ap_list.append(np.int(line[40:43]))
                time_list.append(year+np.float(day_count)/365.+0.5625/365)
                kp_list.append(np.int(line[20:22]))
                ap_list.append(np.int(line[43:46]))
                time_list.append(year+np.float(day_count)/365.+0.6875/365)
                kp_list.append(np.int(line[22:24]))
                ap_list.append(np.int(line[46:49]))
                time_list.append(year+np.float(day_count)/365.+0.8125/365)
                kp_list.append(np.int(line[24:26]))
                ap_list.append(np.int(line[49:52]))
                time_list.append(year+np.float(day_count)/365.+0.9375/365)
                kp_list.append(np.int(line[26:28]))            
                ap_list.append(np.int(line[52:55]))
    kp = pd.Series(kp_list,index=time_list)
    ap = pd.Series(ap_list,index=time_list)
    return kp,ap
"""





def get_aurora_prob_over_route(lat,lon,mag_lat,mag_lon,mag_lats,kpX_aurora_prob,dl):
    ### for a given flight route (lat,lon,mag_lat,mag_lon) output the probability of seeing 
    ### the aurora. 
    ### mag_lats and kpX_aurora_prob are matching arrays giving the model
    ### probability of seeing the aurora over a 4-hour period at a given mag latitude 
    ### (see get_aurora_prob_over_time for details)
    N_pts = len(lat)
    aur_frac = np.zeros(N_pts)
    for i in range(N_pts):
        if np.isnan(lat[i]):
            continue
        aur_frac[i] = np.interp(mag_lat[i],mag_lats,kpX_aurora_prob)
    return aur_frac
	

def get_aurora_prob_over_time(lat,lon,mag_lat,mag_lon,x_model,kp_model,route_dl,N_days=30,day_max=450):
    ### get probability of seeing aurora for a given flight path, over the next day_max days
    ### N_days is the number of days to sample within the day_max-day window
    ### lat,lon,mag_lat,mag_lon are the flight path info
    ### x_model and kp_model are matching arrays showing my model prediction for the average 
    ### Kp for the rest of the aurora cycle
    ###
    ###
    ### here is where I put my aurora model itself
    mag_lats = np.linspace(50,90,401)
    kp0_aurora_prob = (mag_lats>66.5)*0.4     # over 4 hours
    kp0_frac_over_route = get_aurora_prob_over_route(lat,lon,mag_lat,mag_lon,mag_lats,kp0_aurora_prob,route_dl)
    kp1_aurora_prob = (mag_lats>64.5)*0.5     # over 4 hours
    kp1_frac_over_route = get_aurora_prob_over_route(lat,lon,mag_lat,mag_lon,mag_lats,kp1_aurora_prob,route_dl)
    kp2_aurora_prob = (mag_lats>62.4)*0.6     # over 4 hours
    kp2_frac_over_route = get_aurora_prob_over_route(lat,lon,mag_lat,mag_lon,mag_lats,kp2_aurora_prob,route_dl)
    kp3_aurora_prob = (mag_lats>60.4)*0.7     # over 4 hours
    kp3_frac_over_route = get_aurora_prob_over_route(lat,lon,mag_lat,mag_lon,mag_lats,kp3_aurora_prob,route_dl)
    #kp4_aurora_prob = (mag_lats>58.3)*0.8     # over 4 hours
    #kp5_aurora_prob = (mag_lats>56.3)*0.9     # over 4 hours
    ### based on https://www.swpc.noaa.gov/content/tips-viewing-aurora
    ### and fig 12-20b of http://www.cnofs.org/Handbook_of_Geophysics_1985/Chptr12.pdf
    dts = np.linspace(0,day_max,N_days)
    p_aurora = np.zeros(N_days)
    for i in range(N_days):
    	date_today = date.today() +timedelta(days=dts[i])
    	year_code = date_today.year+date_today.month/12.0 +date_today.day/365.0
    	kp_prediction = np.interp(year_code,x_model,kp_model)
    	if kp_prediction < 0.5:
    		aurora_frac_over_route = kp0_frac_over_route
    	elif kp_prediction < 1.5:
    		aurora_frac_over_route = kp1_frac_over_route
    	elif kp_prediction < 2.5:
    		aurora_frac_over_route = kp2_frac_over_route
    	elif kp_prediction < 3.5:
    		aurora_frac_over_route = kp3_frac_over_route
    	else:
    		print('model kp is too high')
    		raise
    	p_aurora[i] = get_one_day_aurora_prob(lat,lon,mag_lat,mag_lon,dts[i],mag_lats,aurora_frac_over_route,route_dl)
    return p_aurora,dts
    
    



def make_aurora_plot(N_routes,dts,aurora_p,final_airport_codes):
	fig=Figure()
	ax=fig.add_subplot(111)
	
	plt.figure()
	plt.clf()
	color_list = plt.cm.Set1(np.linspace(0, 1, N_routes))
	for i in range(N_routes):
		plt.plot(dts,100.0*aurora_p[:,i],'-', label=final_airport_codes[i][4:],color=color_list[i])
	#plt.xlabel('days from now')
	plt.title('Probability of Seeing the Aurora on Different Routes')
	plt.ylabel('Probability')
	plt.ylim(0,100)
	dts_new = np.linspace(0,dts.max(),dts.max()+1)
	dts_to_dates = np.array([date.today() + timedelta(days=dts_new[i]) for i in range(len(dts_new))])
	month_starts = np.array([dts_to_dates[i].day == 1 for i in range(len(dts_new))])
	x_tick_locs = dts_new[np.where(month_starts)]
	month_nums = dts_to_dates[np.where(month_starts)]
	x_tick_names = np.array([NumtoMonth2(month_nums[i].month)+'\n '+str(month_nums[i].year) for i in range(len(month_nums))],dtype='str')
	#print(x_tick_locs)
	#print(x_tick_names)
	plt.xticks(x_tick_locs,x_tick_names)
	plt.yticks([0,20,40,60,80,100],['0%','20%','40%','60%','80%','100%'])
	#plt.xticks(rotation=90)
	plt.xticks(fontsize=7)#, rotation=90)
	plt.xlim([0,dts.max()])
	plt.legend(loc='best')
	img = io.BytesIO()  # create the buffer
	plt.savefig(img, format='png')  # save figure to the buffer
	img.seek(0)  # rewind your buffer
	plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
	return plot_data






def match_iata_code(code,reader):
    for row in reader:
        if(row['iata_code'] == code):
            if row['id']!='326459':   #CDG weirdness
                return row['latitude_deg'],row['longitude_deg'],row['name']
                
                
def get_scraped_data(df,origin,destination):
    with open('../../airports.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        origin_lat,origin_lon,_ = match_iata_code(origin,reader)
    with open('../../airports.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        destination_lat,destination_lon,_ = match_iata_code(destination,reader)
    matches = df.loc[(df['Origin lon'] == np.float(origin_lon)) *
                         (df['Destination lat'] == np.float(destination_lat))]
    date_code_today = date.today().year + date.today().month/12.0 + date.today().day/365.0
    date_from_now = matches['date'] - date_code_today
    return date_from_now*365.0,matches['airfare']

	



def make_prices_plot(N_routes,dts,prices,final_airport_codes,df):
	fig=Figure()
	ax=fig.add_subplot(111)
	
	plt.figure()
	plt.clf()
	color_list = plt.cm.Set1(np.linspace(0, 1, N_routes))
	for i in range(N_routes):
		plt.plot(dts,prices[:,i],'-',color=color_list[i],label=final_airport_codes[i][4:])
		dts_data,price_data = get_scraped_data(df,final_airport_codes[i][:3],final_airport_codes[i][4:])
		plt.plot(dts_data,price_data,'o',color=color_list[i],label='_nolegend_')
	#plt.xlabel('days from now')
	#plt.ylabel('one-way price (USD)')
	plt.legend(loc='best')
	
	
	plt.title('Approximate Expected Airfare on Different Routes')
	plt.ylabel('One-Way Price (USD)')
	dts_new = np.linspace(0,dts.max(),dts.max()+1)
	dts_to_dates = np.array([date.today() + timedelta(days=dts_new[i]) for i in range(len(dts_new))])
	month_starts = np.array([dts_to_dates[i].day == 1 for i in range(len(dts_new))])
	x_tick_locs = dts_new[np.where(month_starts)]
	month_nums = dts_to_dates[np.where(month_starts)]
	x_tick_names = np.array([NumtoMonth2(month_nums[i].month)+'\n '+str(month_nums[i].year) for i in range(len(month_nums))],dtype='str')
	#print(x_tick_locs)
	#print(x_tick_names)
	plt.xticks(x_tick_locs,x_tick_names)
	#plt.xticks(rotation=90)
	plt.xticks(fontsize=7)#, rotation=90)
	plt.xlim([0,dts.max()])
	plt.ylim([0,prices.max()*1.2])

	#plt.ylim(0,1)	
	#ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
	#fig.autofmt_xdate()
	#plt.xticks(rotation=90)

	img = io.BytesIO()  # create the buffer
	plt.savefig(img, format='png')  # save figure to the buffer
	img.seek(0)  # rewind your buffer
	plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
	return plot_data


def scrape_skypicker(origin_code,destination_code,delta_t = 7,t_max = 300):
    """
    scrapes skypicker API, searching for the cheapest flight between origin and destination over an
    interval of delta_t days, out to a total of t_max days from today
    returns:
        best_prices - np.array of best_prices with delta_t time resolution
        dates - pd.Series of all t_max days considered, matched to the indices in best_prices
    """
    current_date = time.strftime('%m/%d/%Y')
    num_adults = 1   #input('number of adults traveling? ')
    best_prices = np.zeros(t_max)
    dates = pd.date_range(pd.datetime.today(), periods=t_max)
    for day_idx in range(t_max):
        if np.mod(day_idx,delta_t) != 0:
            continue
        sp_api = SkyPickerApi()
        print(origin_code)
        print(destination_code)
        sp_results = sp_api.search_flights_all(origin_code, destination_code,
                                           datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=day_idx),
                                           datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=day_idx+delta_t),
                                           num_adults)
        prices = []
        for i in range(len(sp_results)):
            try:
                price = np.float(sp_results[i]['price'])
                prices.append(price)
            except:
                pass
        best_prices[day_idx:day_idx+delta_t] = np.min(prices)
    return best_prices,dates



def NumtoMonth(num):
    NumMonth = {
                    1: 'January',
                    2: 'February',
                    3: 'March',
                    4: 'April',
                    5: 'May',
                    6: 'June',
                    7: 'July',
                    8: 'August',
                    9: 'September', 
                    10: 'October',
                    11: 'November',
                    12: 'December'
            }
    return NumMonth[num]



def NumtoMonth2(num):
    NumMonth = {
                    1: 'Jan',
                    2: 'Feb',
                    3: 'Mar',
                    4: 'Apr',
                    5: 'May',
                    6: 'Jun',
                    7: 'Jul',
                    8: 'Aug',
                    9: 'Sep', 
                    10: 'Oct',
                    11: 'Nov',
                    12: 'Dec'
            }
    return NumMonth[num]



def make_kiwi_url(origin, destination, month, year):
        """ Create URL to search kiwi.com for direct flights over a one-month period
        origin  and destination are IATA codes
        month and year are integers
        """
        today_date = datetime.today()
        date1 = today_date.replace(year=year,month=month,day=1)
        if month in [1,3,5,7,8,10,12]:
            date2 = today_date.replace(year=year,month=month,day=31)
        elif (month == 2) and (year == 2020):  #leap year
            date2 = today_date.replace(year=year,month=month,day=29)
        elif (month == 2):
            date2 = today_date.replace(year=year,month=month,day=28)
        else:
            date2 = today_date.replace(year=year,month=month,day=30)
        
        base_url = 'https://www.kiwi.com/us/search/'
        loc_url = origin+'/'+destination+'/'
        date_url = date1.strftime('%Y-%m-%d')+'_'+date2.strftime('%Y-%m-%d')+'/'
        final_url = 'no-return?stopNumber=0'
        url_final = base_url + loc_url + date_url + final_url
        return url_final


