import json
import datetime

import numpy as np
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
    
#-----------------------------------

#with open('./data/sample_item_list.json', 'r') as f:
#    item_list=json.load(f)
#with open('./data/sample_roi.json', 'r') as f:
#    roi=json.load(f)
#    roi_shape = shape(roi)


def geodf_geometry_area(gdf):
    """ 
    Given a geopandas data.frame return the geographic area of all polygons and
    multipolygons.
    Return value is km^2.
    This uses a pyproj method and is correct even with lat/lon coordinates.
    """
    
    geod = gdf.crs.get_geod()
    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon','Polygon']:
            return np.nan
        
        # For MultiPolygon do each separately
        if geom.geom_type=='MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[0]
    
    return gdf.geometry.apply(area_calc) / (1000**2)


def scene_list_to_gdf(scene_list):
    """Convert scene list to geopandas data.frame with various attributes
    
    colnames = 'id','acquired','date','time','satellite_id','clear_percent','cloud_percent'
    """
    property_attrs = ['satellite_id','acquired','clear_percent','cloud_percent']
    
    all_attributes = []
    all_geoms = []
    for i in scene_list:
        scene_attrs = {a:i['properties'][a] for a in property_attrs}
        scene_attrs['id'] = i['id']
        
        all_attributes.append(scene_attrs)
        all_geoms.append(shape(i['geometry']))
    
    geodf = gpd.GeoDataFrame(all_attributes, geometry=gpd.GeoSeries(all_geoms)).set_crs(4326)
    geodf['date'] = gpd.pd.DatetimeIndex(geodf.acquired).date
    geodf['time'] = gpd.pd.DatetimeIndex(geodf.acquired).strftime('%H%M')

    return geodf

def scene_geodf_to_strip_geodf(scene_geodf):
    """ 
    From the scene geodf, convert scenes into strips. Where a strip is a 
    single pass from 1 satellite. This is done faily niavely by grouping
    each satelliteXdate combination. Scenes within a strip will align 
    in a generally north/south direction.
    
    The new geopandas data.frame will have the following columns
    strip_id: a unique id from the pasting of satellite_id+date
    scenes_in_strip: a comma separated string of all scene ids
    clear_percent: average clear_percent among all scenes
    cloud_percent: average cloud_percent among all scenes
    geometry: the union of all scenes within the strip. can potentially be a multipolygon
    
    """
    strip_geodf = scene_geodf.groupby(['satellite_id','date']).agg(
        geometry = pd.NamedAgg('geometry', aggfunc=lambda x: unary_union(x)),
        scenes_in_strip = pd.NamedAgg('id', aggfunc=lambda x: ','.join(x)),
        clear_percent = pd.NamedAgg('clear_percent', aggfunc='mean'),
        cloud_percent = pd.NamedAgg('cloud_percent', aggfunc='mean'),
            ).reset_index().set_crs(scene_geodf.crs)
    strip_geodf['strip_id'] = strip_geodf['satellite_id'] + '-' + strip_geodf['date'].astype(str)
    return strip_geodf

def planet_scene_trimming(scene_list, 
                          roi,
                          potential_solutions_to_calculate = 50,
                          iterations_per_solution = 5,
                          percent_increase_beta = 0.1,
                          result_type = 'geodf', # gpd data.frame or just a list of scenes ids
                          return_stats = True,
                          ):
    
    #TODO: 
    # check roi is geom dict and convert to shape
    # to_return must be 'geodf', or 'scene_list'
    # algorithm parameters should be reasonable
    
    scene_geodf = scene_list_to_gdf(scene_list)
    strip_geodf = scene_geodf_to_strip_geodf(scene_geodf)
    roi_shape = shape(roi)
    
    scene_geodf['area_km'] = geodf_geometry_area(scene_geodf)
    strip_geodf['area_km'] = geodf_geometry_area(strip_geodf)
    
    avg_scene_size = scene_geodf.area.mean()
    avg_strip_size = strip_geodf.area.mean()
    
    best_solution = None
    best_solution_score = 1e6 
    for solution_i in range(potential_solutions_to_calculate):
        # Start with fresh copy.
        strip_geodf = strip_geodf.copy()
        
        # calculate how much of each strip is within the ROI
        strip_geodf['total_area_in_ROI'] = strip_geodf.geometry.apply(lambda x: x.intersection(roi_shape).area/x.area)
        
        # differences among proposed solutions is due to this initial shuffle weighted by strip area.
        strip_geodf = strip_geodf.sample(frac=1, replace=False, weights='total_area_in_ROI')
        
        strip_geodf['proposed'] = False
        strip_geodf['in_solution'] = False
        strip_geodf.in_solution.iat[0] = True
        
        
        for iter_i in range(iterations_per_solution):
            # calculate the neccessary increase in ROI area of this pass
            current_total_strip_shape =  unary_union(strip_geodf[strip_geodf.in_solution].geometry)
            current_roi_filled_percent = round(current_total_strip_shape.intersection(roi_shape).area / roi_shape.area,5)
            assert current_roi_filled_percent <= 1 and current_roi_filled_percent >= 0, 'current_roi_filled_percent outside 0-1'
            iteration_threshold_for_inclusion = (1-current_roi_filled_percent) * percent_increase_beta
            
            print('iteration threshold: {}'.format(round(iteration_threshold_for_inclusion,5)))
            
            for strip_i in range(1,len(strip_geodf)):
                if strip_geodf.in_solution.iloc[strip_i]:
                    print('strip already in solution')
                    continue
                
                current_total_strip_shape =  unary_union(strip_geodf[strip_geodf.in_solution].geometry)
                current_total_strip_area_in_roi = current_total_strip_shape.intersection(roi_shape).area
                #current_coverage = current_strip_shape.intersection(roi_shape).area / roi_shape.area
                
                strip_geodf.proposed.iat[strip_i] = True
                proposed_total_strip_shape = unary_union(strip_geodf[strip_geodf.proposed | strip_geodf.in_solution].geometry)
                proposed_total_strip_area_in_roi = proposed_total_strip_shape.intersection(roi_shape).area
                
                # Of the single strip for proposal, what how much does in increase ROI coverage?
                proposed_total_strip_percent_increase = (proposed_total_strip_area_in_roi - current_total_strip_area_in_roi) / roi_shape.area
                
                if proposed_total_strip_percent_increase >= iteration_threshold_for_inclusion:
                    strip_geodf.in_solution.iat[strip_i] = True
                
                # no longer proposed regardless if it ended up in solution.
                strip_geodf.proposed.iat[strip_i] = False
    
        
        # Final stats and score for this proposed solution
        proposed_solution_shape = unary_union(strip_geodf[strip_geodf.in_solution].geometry)
        proposed_solution_jaccard = roi_shape.intersection(proposed_solution_shape).area / roi_shape.union(proposed_solution_shape).area
        
        # area_efficency is the *total* scene area (that will be charged to quota on download) as a 
        # fraction of the ROI area. 
        proposed_solution_total_area = strip_geodf[strip_geodf.in_solution].area.sum()
        proposed_solution_area_efficency = proposed_solution_total_area / roi_shape.area
        # Proposed solution score is jaccard index penalized by area_efficency. When both are 1
        # to score is minimized at 0.
        proposed_solution_score = (2-(proposed_solution_jaccard + proposed_solution_area_efficency)) ** 2
        if proposed_solution_score < best_solution_score:
            best_solution_score = proposed_solution_score
            best_solution = strip_geodf


    # get scene ids within the strips of the best solution
    best_solution_scenes = []
    for scene_list_str in  best_solution[best_solution.in_solution].scenes_in_strip:
        best_solution_scenes.extend(scene_list_str.split(','))
    
    best_solution_scene_geodf = scene_geodf[scene_geodf.id.isin(best_solution_scenes)]
    
    if result_type == 'geodf':
        return_obj = best_solution_scene_geodf
    elif result_type == 'scene_list':
        return_obj = best_solution_scenes
        
    if return_stats:
        stats =  dict(original_scene_area = scene_geodf.area_km.sum(),
                      trimmed_scene_area  = best_solution_scene_geodf.area_km.sum(),
                      #original_roi_area   = total_roi_area,
                      #area_efficency      = area_efficency,
                      original_scene_count    = len(scene_geodf),
                      trimmed_scene_count     = len(best_solution_scene_geodf),
                      )
        return stats, return_obj
    else:
        return return_obj
    
    
#strip_geodf = best_solution
#strip_geodf['date'] = strip_geodf.date.astype(str)
#strip_geodf[strip_geodf.in_solution].to_file('test_strips.geojson', driver='GeoJSON')






