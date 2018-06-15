    
from flask import Flask,render_template, request, redirect


import requests
from datetime import datetime, timedelta, date, time

#import datetime
import time
import numpy as np
import json
import pandas as pd
import ephem

### for the geodesic module later on down
import os
import math
import logging
from pyproj import Geod
from shapely.geometry import LineString, mapping
from fiona import collection
from fiona.transform import transform_geom
from fiona.crs import from_epsg

import matplotlib
matplotlib.use('Agg')

### make the globe
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


import aacgmv2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


import io
import base64
import urllib.parse


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





    def search_flights(self, origin, start_date, end_date):
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
        
        
    
    
    def search_flights2(self, origin, destination, start_date, end_date,
                       num_passengers):
        """ Searches for flights given a time range and origin and destination.
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
    
    




class ComputeGeodesicLineError(Exception):
    pass

class ExportGeodesicLineError(Exception):
    pass

class GeodesicLine2Gisfile(object):

    def __init__(self, antimeridian=True, loglevel="INFO"):
        """
            antimeridian: solving antimeridian problem [True/False].
            prints: print output messages [True/False].
        """
        self.__antimeridian = antimeridian
        self.__logger = self.__loggerInit(loglevel)
        self.__distances = []


    def __loggerInit(self, loglevel):
        """
        Logger init...
        """
        if loglevel=="INFO":
            __log_level=logging.INFO
        elif loglevel=="DEBUG":
            __log_level=logging.DEBUG
        elif loglevel=="ERROR":
            __log_level=logging.ERROR
        else:
            __log_level=logging.NOTSET

        logfmt = "[%(asctime)s - %(levelname)s] - %(message)s"
        dtfmt = "%Y-%m-%d %I:%M:%S"

        logging.basicConfig(level=__log_level, format=logfmt, datefmt=dtfmt)

        return logging.getLogger()
    
    @property
    def distances(self):
        return self.__distances
    
    def __dest_folder(self, dest, crtfld):
        if not os.path.exists(dest):
            if not crtfld:
                self.__logger.error("Output folder does not exist. Set a valid folder path to store file.")
                return
            os.mkdir(dest)
            self.__logger.debug("New output folder {0} created.".format(dest))
        else:
            self.__logger.debug("Output folder {0} already exists.".format(dest))

    def gdlComp(self, lons_lats, km_pts=20):
        """
        Compute geodesic line
            lons_lats: input coordinates.
            (start longitude, start latitude, end longitude, end latitude)
            km_pts: compute one point each 20 km (default).
        """

        try:
            lon_1, lat_1, lon_2, lat_2 = lons_lats

            pygd = Geod(ellps='WGS84')

            res = pygd.inv(lon_1, lat_1, lon_2, lat_2)
            dist = res[2]

            pts  = int(math.ceil(dist) / (km_pts * 1000))

            coords = pygd.npts(lon_1, lat_1, lon_2, lat_2, pts)

            coords_se = [(lon_1, lat_1)] + coords
            coords_se.append((lon_2, lat_2))
            
            self.__distances.append({
                "id": len(self.__distances),
                "distance": dist,
                "coords": lons_lats
            })
            
            self.__logger.info("Geodesic line succesfully created!")
            self.__logger.info("Total points = {:,}".format(pts))
            self.__logger.info("{:,.4f} km".format(dist / 1000.))

            return coords_se

        except Exception as e:
            self.__logger.error("Error: {0}".format(e))
            raise ComputeGeodesicLineError(e)


    def gdlToGisFile(self, coords, folderpath, layername, fmt="ESRI Shapefile",
                     epsg_cd=4326, prop=None, crtfld=True):
        """
        Dump geodesic line coords to ESRI Shapefile
        and GeoJSON Linestring Feature
            coords: input coords returned by gcComp.
            folderpath: folder to store output file.
            layername: output filename.
            fmt: output format ("ESRI Shapefile" (default), "GeoJSON").
            epsg_cd: Coordinate Reference System, EPSG code (default: 4326)
            prop: property
            
            crtfld: create folder if not exists (default: True).
        """

        schema = { 'geometry': 'LineString',
                   'properties': { 'prop': 'str' }
                   }

        try:

            if fmt in ["ESRI Shapefile", "GeoJSON"]:
                ext = ".shp"
                if fmt == "GeoJSON":
                    ext = ".geojson"

                filepath = os.path.join(folderpath, "{0}{1}".format(layername, ext))

                self.__dest_folder(folderpath, crtfld)

                if fmt == "GeoJSON" and os.path.isfile(filepath):
                    os.remove(filepath)

                out_crs = from_epsg(epsg_cd)

                with collection(filepath, "w", fmt, schema, crs=out_crs) as output:

                    line = LineString(coords)

                    geom = mapping(line)

                    if self.__antimeridian:
                        line_t = self.__antiMeridianCut(geom)
                    else:
                        line_t = geom

                    output.write({
                        'properties': {
                            'prop': prop
                        },
                        'geometry': line_t
                    })

                self.__logger.info("{0} succesfully created!".format(fmt))

            else:
                self.__logger.error("No format to store output...")
                return

        except Exception as e:
            self.__logger.error("Error: {0}".format(e))
            raise ExportGeodesicLineError(e)


    def gdlToGisFileMulti(self, data, folderpath, layername, prop=[], gjs=True):
        """
        Run creation for a multi input: a list of lat/lon.
            data: a list with input coordinates.
            [
              (start longitude, start latitude, end longitude, end latitude),
              (start longitude, start latitude, end longitude, end latitude),
              (start longitude, start latitude, end longitude, end latitude),
              (start longitude, start latitude, end longitude, end latitude),
              ...
            ]
            folderpath: folder to store output files.
            layername: output base filename (an ordinal integer is added at the end).
            gfs: GeoJSON output format [True (default)|False], in addition to Shapefile.
        """

        try:
            lendata = len(data)

            for i in range(lendata):
                lyrnm = "{0}{1}".format(layername, i)
                _prop = prop[i] if prop else None
                self.__multiGeodesicLineCreation(data[i], folderpath, lyrnm, gjs, _prop)

        except Exception as e:
            self.__logger.error("Error: {0}".format(e))
            raise ExportGeodesicLineError(e)


    def __multiGeodesicLineCreation(self, lons_lats, folderpath, layername, gjs, prop):
        """
        Creating geodesic lines in batch mode
        """

        cd = self.gdlComp(lons_lats)

        self.gdlToGisFile(cd, folderpath, layername, prop=prop)

        if gjs:
            self.gdlToGisFile(cd, folderpath, layername, fmt="GeoJSON")


    def __antiMeridianCut(self, geom):
        """
        Solving antimeridian problem.
        """

        src_crs = '+proj=longlat +datum=WGS84 +no_defs'
        dst_crs = '+proj=longlat +datum=WGS84 +no_defs'

        am_offset = 360.0

        line_t = transform_geom(src_crs, dst_crs, geom,
                                antimeridian_cutting=self.__antimeridian,
                                antimeridian_offset=am_offset,
                                precision=-1)

        return line_t
        


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
                mg_lat,mg_lon = aacgmv2.convert(lat, lon, 10, date=datetime.date.today(), 
                                                  a2g=False, trace=False, allowtrace=False,
                                                  badidea=False, geocentric=False)
                mag_lat.append(mg_lat[0])
                mag_lon.append(mg_lon[0])
                gc_lat.append(lat)
                gc_lon.append(lon)
            for j in range(len(data['features'][0]['geometry']['coordinates'][1][:])):
                lon,lat = data['features'][0]['geometry']['coordinates'][1][j]
                mg_lat,mg_lon = aacgmv2.convert(lat, lon, 10, date=datetime.date.today(), 
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
    h = np.int(np.floor(12.0 + np.float(lon)/15.0))
    m = np.int(np.floor(np.mod(np.float(lon)/15.0,60)))
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
    


def get_one_day_aurora_prob(lat,lon,mag_lat,mag_lon,dt,mag_lats,kpX_aurora_prob,dl):
    ### for a given flight route (lat,lon,mag_lat,mag_lon) output the probability of seeing 
    ### the aurora. 
    ### mag_lats and kpX_aurora_prob are matching arrays giving the model
    ### probability of seeing the aurora over a 4-hour period at a given mag latitude 
    ### (see get_aurora_prob_over_time for details)
    ### dt = number of days from today
    v_airplane = 600.0  #km/hr
    N_pts = len(lat)
    probs = np.zeros(N_pts)
    for i in range(N_pts):
        if np.isnan(lat[i]):
            continue
        day_frac = get_length_of_daytime(str(lat[i]),str(lon[i]),dt)
        aur_frac = np.interp(mag_lat[i],mag_lats,kpX_aurora_prob)
        probs[i] = (1.0-day_frac) * aur_frac * (dl / v_airplane / 4.0)
    return 1.0 - np.prod(1.0 - probs)
    




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


def get_aurora_prob_over_time(lat,lon,mag_lat,mag_lon,x_model,kp_model,route_dl,N_days=31,day_max=300):
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
    kp1_aurora_prob = (mag_lats>64.5)*0.5     # over 4 hours
    kp2_aurora_prob = (mag_lats>62.4)*0.6     # over 4 hours
    kp3_aurora_prob = (mag_lats>60.4)*0.7     # over 4 hours
    kp4_aurora_prob = (mag_lats>58.3)*0.8     # over 4 hours
    kp5_aurora_prob = (mag_lats>56.3)*0.9     # over 4 hours
    ### based on https://www.swpc.noaa.gov/content/tips-viewing-aurora
    ### and fig 12-20b of http://www.cnofs.org/Handbook_of_Geophysics_1985/Chptr12.pdf
    dts = np.linspace(0,day_max,N_days)
    p_aurora = np.zeros(N_days)
    for i in range(N_days):
        print(i)
        date_today = date.today() +timedelta(days=dts[i])
        year_code = date_today.year+date_today.month/12.0 +date_today.day/365.0
        kp_prediction = np.interp(year_code,x_model,kp_model)
        if kp_prediction < 0.5:
            kpX_aurora_prob = kp0_aurora_prob
        elif kp_prediction < 1.5:
            kpX_aurora_prob = kp1_aurora_prob
        elif kp_prediction < 2.5:
            kpX_aurora_prob = kp2_aurora_prob
        elif kp_prediction < 3.5:
            kpX_aurora_prob = kp3_aurora_prob
        else:
            kpX_aurora_prob = kp4_aurora_prob
        p = get_one_day_aurora_prob(lat,lon,mag_lat,mag_lon,dts[i],mag_lats,kpX_aurora_prob,route_dl)
        p_aurora[i] = p
    return p_aurora,dts
    
    



def make_aurora_plot(N_routes,dts,aurora_p,final_airport_codes):
	fig=Figure()
	ax=fig.add_subplot(111)
	
	plt.figure()
	plt.clf()
	color_list = plt.cm.Set1(np.linspace(0, 1, N_routes))
	for i in range(N_routes):
		plt.plot(dts,aurora_p[:,i],'-', label=final_airport_codes[i][4:],color=color_list[i])
	plt.xlabel('days from now')
	plt.ylabel('aurora prob')
	plt.ylim(0,1)
	plt.legend(loc='best')
	img = io.BytesIO()  # create the buffer
	plt.savefig(img, format='png')  # save figure to the buffer
	img.seek(0)  # rewind your buffer
	plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
	return plot_data


def make_prices_plot(N_routes,dates,prices,final_airport_codes,dts):
	fig=Figure()
	ax=fig.add_subplot(111)
	
	plt.figure()
	plt.clf()
	color_list = plt.cm.Set1(np.linspace(0, 1, N_routes))
	for i in range(N_routes):
		plt.plot(dates,prices[:,i],'-',color=color_list[i],label=final_airport_codes[i][4:])
	plt.xlabel('days from now')
	plt.ylabel('one-way price (USD)')
	plt.legend(loc='best')
	plt.xlim([0,dts.max()])
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
        sp_results = sp_api.search_flights2(origin_code, destination_code,
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
    	sp_api = SkyPickerApi()
    	sp_results = sp_api.search_flights(airport_code, datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=7),datetime.strptime(current_date, '%m/%d/%Y')+timedelta(days=14))
    	dest_iata_codes = []
    	for i in range(len(sp_results)):
    		dest_iata_code = sp_results[i]['legs'][0]['to'][-4:-1]
    		dest_iata_codes.append(dest_iata_code)
    		
    	unique_dest_iata_codes = set(dest_iata_codes)
    	sp_loc_api = SkyPickerApi()
    	unique_dest_lons = []
    	unique_dest_lats = []
    	for iata_code in unique_dest_iata_codes:
    		lat,lon = get_airport_coords(sp_api,iata_code)    #use a local file instead
    		unique_dest_lats.append(lat)
    		unique_dest_lons.append(lon)
    		
    	origin_lat,origin_lon = get_airport_coords(sp_api,airport_code)
    	dest_airports = pd.DataFrame(data=np.transpose([list(unique_dest_iata_codes),unique_dest_lons,unique_dest_lats]),columns=['code','lon','lat'])
    	if origin_lat > arctic_circle_lat:
    		does_this_route_pass_through_arctic = np.ones(len(dest_airports))
    	else:
    		does_this_route_pass_through_arctic = np.zeros(len(dest_airports))
    		for i in range(len(dest_airports)):
    			folderpath = 'gis_temp/'
    			layername = airport_code+'-'+dest_airports.loc[i].code
    			gtg = GeodesicLine2Gisfile()
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
    				
    	figdata_path = make_plot(arctic_circle_lat,origin_lon,origin_lat,does_this_route_pass_through_arctic,dest_airports,airport_code)
    	    	
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
    		df = get_flight_route(i,does_this_route_pass_through_arctic,airport_code,dest_airports)
    		if df is not None:
    			routes = routes.append(df)
    			N_routes += 1
    			
        ### put the aurora data into pandas dataframes
        ### I will actually only use the ap data
        ### and I think there is something wrong with how I am parsing the kp data		
    	kp,ap = get_aurora_data()        
    	
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
    		p,dts = get_aurora_prob_over_time(routes.iloc[4*i][1:].values,routes.iloc[4*i+1][1:].values,routes.iloc[4*i+2][1:].values,routes.iloc[4*i+3][1:].values,x_model,kp_model,route_dl)
    		aurora_p[:,i] = p


    	aurora_plot_path = make_aurora_plot(N_routes,dts,aurora_p,final_airport_codes)
    	
    	t_max = 150
    	dates = pd.date_range(pd.datetime.today(), periods=t_max)
    	dates_dt = np.linspace(0,t_max-1,t_max)
    	prices = np.zeros([t_max,N_routes])
    	for i in range(N_routes):
    		#print(airport_code)
    		print(final_airport_codes[i][4:])
    		best_prices_skypicker,dates_skypicker = scrape_skypicker(airport_code,final_airport_codes[i][4:],delta_t = 7,t_max = t_max)
    		prices[:,i] = best_prices_skypicker
    		
    	price_plot_path = make_prices_plot(N_routes,dates_dt,prices,final_airport_codes,dts)	

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