import json
import datetime

from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd

class PlanetCompositePrep:
    def __init__(self, scene_list, roi):
        self.scene_list = scene_list
        self.scene_geodf = self.scene_list_to_gdf(scene_list)
        self.strip_geodf = self.scene_geodf_to_strip_geodf(self.scene_geodf)
        self.roi = roi
        self.roi_shape = shape(roi)
    
    def scene_list_to_gdf(self, scene_list):
        """Convert scene list to geopandas data.frame with various attributes
        
        colnames = id,date,time,satellite_id,satellite_id
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
        
    def calculate_scene_subset_roi_jaccard(self, scene_ids):
        """ 
        Given a list of IDs, which are present in scene_geodf, what is the jaccard
        index with the roi?
        
        Jaccard = intersection(roi1,roi1) / union(roi1,roi2)
        """
        subset_geoms = self.scene_geodf[self.scene_geodf.id.isin(scene_ids)]
        subset_geom_shape = unary_union(subset_geoms.geometry)
        
        return self.roi_shape.intersection(subset_geom_shape).area / self.roi_shape.union(subset_geom_shape).area
    
    def scene_geodf_to_strip_geodf(self, scene_geodf):
        """ 
        From the scene geodf, convert scenes into strips. Where a strip is a 
        single pass from 1 satellite. Scenes within a strip will align 
        in a generally north/south direction
        """
        strip_geodf = scene_geodf.groupby(['satellite_id','date']).agg(
            geometry = pd.NamedAgg('geometry', aggfunc=lambda x: unary_union(x)),
            scenes_in_strip = pd.NamedAgg('id', aggfunc=lambda x: ','.join(x)),
            clear_percent = pd.NamedAgg('clear_percent', aggfunc='mean'),
            cloud_percent = pd.NamedAgg('cloud_percent', aggfunc='mean'),
                ).reset_index().set_crs(scene_geodf.crs)
        strip_geodf['strip_id'] = strip_geodf['satellite_id'] + '-' + strip_geodf['date'].astype(str)
        return strip_geodf
    
    def calculate_scene_subset_roi_completness(self, scene_ids):
        """ 
        Given a list of IDs, which are present in scene_geodf, how much of the roi
        footprint is covered by them?
        """
        subset_geoms = self.scene_geodf[self.scene_geodf.id.isin(scene_ids)]
        subset_geom_shape = unary_union(subset_geoms.geometry)
        
        return self.roi_shape.intersection(subset_geom_shape).area / self.roi_shape.area
    
    def calculate_scene_subset_area(self, scene_ids):
        """ 
        Given a list of IDs, which are present in scene_geodf, what is the area of
        each? 
        Done using a geographic projection, area in km^2. Can be summed with .sum()
        """
        subset_geoms = self.scene_geodf[self.scene_geodf.id.isin(scene_ids)]
        
        geod = subset_geoms.crs.get_geod()
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
        
        return subset_geoms.geometry.apply(area_calc).sum() / (1000**2)
    
    def composite_trim(self):
        """
        Given a full list of scenes, pair down to the minimum 
        needed to make a composite given the ROI
        """
        pass
    
    def daily_stats(self):
        """ 
        Get stats for each unique date in the scene list with respect to the ROI
        """
                
        date_stats = self.scene_geodf.groupby('date').agg(
            jaccard = pd.NamedAgg(column='id', aggfunc=self.calculate_scene_subset_roi_jaccard),
            completeness = pd.NamedAgg(column='id', aggfunc=self.calculate_scene_subset_roi_completness),
            total_area = pd.NamedAgg(column='id', aggfunc=self.calculate_scene_subset_area),
            n_scenes = pd.NamedAgg(column='id', aggfunc= lambda x: len(x))
            )
        
        return date_stats
        
        
    def get_outlines_shapefile(self):
        pass
    
    
#-----------------------------------

# with open('./data/sample_item_list.json', 'r') as f:
#     item_list=json.load(f)
# with open('./data/sample_roi.json', 'r') as f:
#     roi=json.load(f)
#     roi_shape = shape(roi)

focal_day =  datetime.date(year=2020,month=5,day=15)

roi_shape = shape(roi)
prepper = PlanetCompositePrep(scene_list = items, roi=roi)

percent_increase_needed_for_inclusion = 0.1

potential_solutions_to_caluclate = 50
iterations_per_solution=5
best_solution = None
best_solution_score = 1e6 
for solution_i in range(potential_solutions_to_caluclate):
    strip_geodf = prepper.strip_geodf.copy()
    
    # calculate how much of each strip is within the ROI
    strip_geodf['total_area_in_ROI'] = strip_geodf.geometry.apply(lambda x: x.intersection(roi_shape).area/x.area)
    strip_geodf = strip_geodf.sample(frac=1, replace=False, weights='total_area_in_ROI')
    #strip_geodf.sort_values('total_area_in_ROI', ascending=False, inplace=True)
    
    strip_geodf['proposed'] = False
    strip_geodf['in_solution'] = False
    strip_geodf.in_solution.iat[0] = True
    
    
    for iter_i in range(iterations_per_solution):
        current_total_strip_shape =  unary_union(strip_geodf[strip_geodf.in_solution].geometry)
        current_roi_filled_percent = current_total_strip_shape.intersection(roi_shape).area / roi_shape.area
        assert current_roi_filled_percent <= 1 and current_roi_filled_percent >= 0, 'current_roi_filled_percent outside 0-1'
        iteration_threshold_for_inclusion = (1-current_roi_filled_percent) / 2
        
        print('iteration threshold: {}'.format(round(iteration_threshold_for_inclusion,5)))
        
        for i in range(1,len(strip_geodf)):
            if strip_geodf.in_solution.iloc[i]:
                print('already in solution')
                continue
            
            current_total_strip_shape =  unary_union(strip_geodf[strip_geodf.in_solution].geometry)
            current_total_strip_area_in_roi = current_total_strip_shape.intersection(roi_shape).area
            #current_coverage = current_strip_shape.intersection(roi_shape).area / roi_shape.area
            
            strip_geodf.proposed.iat[i] = True
            proposed_total_strip_shape = unary_union(strip_geodf[strip_geodf.proposed | strip_geodf.in_solution].geometry)
            proposed_total_strip_area_in_roi = proposed_total_strip_shape.intersection(roi_shape).area
            
            # Of the single strip for proposal, what how much does in increase ROI coverage?
            proposed_total_strip_percent_increase = (proposed_total_strip_area_in_roi - current_total_strip_area_in_roi) / roi_shape.area
            
            if proposed_total_strip_percent_increase >= iteration_threshold_for_inclusion:
                strip_geodf.in_solution.iat[i] = True
            
            # no longer proposed regardless if it ended up in solution.
            strip_geodf.proposed.iat[i] = False

    
    proposed_solution_shape = unary_union(strip_geodf[strip_geodf.in_solution].geometry)
    proposed_solution_jaccard = roi_shape.intersection(proposed_solution_shape).area / roi_shape.union(proposed_solution_shape).area
    
    # area efficienty is the *total* scene area (that will be charged on download) as a 
    # fraction of the ROI area. 
    proposed_solution_total_area = strip_geodf[strip_geodf.in_solution].area.sum()
    proposed_solution_area_efficency = proposed_solution_total_area / roi_shape.area
    proposed_solution_score = (2-(proposed_solution_jaccard + proposed_solution_area_efficency)) ** 2
    if proposed_solution_score < best_solution_score:
        best_solution_score = proposed_solution_score
        best_solution = strip_geodf




strip_geodf = best_solution
strip_geodf['date'] = strip_geodf.date.astype(str)
strip_geodf[strip_geodf.in_solution].to_file('test_strips.geojson', driver='GeoJSON')






