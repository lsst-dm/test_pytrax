#!/usr/bin/env python

#TESTING PYTRAX commit [ 6b46741|https://github.com/pytrax/pytrax/commit/6b46741999635bbe991df9859178db94c907a45c]

# pytrax libraries
#%reset
import pytrax # the pytrax data management and analysis library (now a single import statement)

# helper libraries
import time
import datetime
import numpy as np
import sqlite3 as lite
import sys
import pandas as pd
import copy

import pandas.io.sql as psql
import multiprocessing
import gc
from collections import defaultdict
import pickle
import argparse

t0=time.clock()

### I/O ###

# Get local path of LSST pytrax database
parser = argparse.ArgumentParser()
parser.add_argument("local_db_path", help="Path to pytrax_LSST.db")
parser.add_argument("output_file_path", help="Path to output file")
args = parser.parse_args()
local_db_path = args.local_db_path
outpath = args.output_file_path

# View available universes
pytrax.Universe.show()

universe = pytrax.Universe.get('LSST_simulation_sqlite')
print(universe)


# Matt Payne's db path:
universe['db_file_path'] =  local_db_path

universe['LOAD_OBJECT_ID_EXPOSURES_ONLY'] = False
universe['LOAD_OBJECT_ID_DETECTIONS_ONLY'] = False

# LOAD_OBJECT_ID_INCLUDE_DETECTION universe settings (for LSST simulation only):
# 0 = load only detections for a selected object
# 1 = load only detections for real objects from exposures containing a detection of the selected object
# 2 = load all detections from exposures containing detection of selected object, including false detections and gaussian noise
universe['LOAD_OBJECT_ID_INCLUDE_DETECTIONS'] = 2

universe['RUN_TYPE'] = 0 # i.e. will load exposures and detections from db for a new run, not contents of earlier runs

universe['HEALPIX_SELECT'] = True
universe['healpix_ids'] = [753]
#universe['healpix_ids'] = [752]

pytrax.Universe.save(universe)

# 'save' changes to a selected universe
universe = pytrax.Universe.refresh() # do this after changing universe selection criteria

# select all exposures for a single night in healPix 753 within the julian date range 2452441.29 - 2452442.08,
# which was used both for the multi-night tracklet creation testing and arrow/clustering tests

universe['start_jd'] = 2452439.5 - 15.5
universe['end_jd'] = 2452439.5 + 15.5

pytrax.Universe.save(universe)
print(universe)


# Defining the run-time parameters
# ----------------------------------------------------------------

multi_night_run = pytrax.Discover(universe)

# LSST simulation dataset parameters
params = multi_night_run.get_parameters()
print (params.REFERENCE_TIME, params.REFERENCE_DIRECTION)

# Size of the linking source tracklets - i.e. the
# tracklets used to predict positions in searching for link candidate tracklets
params.MIN_LINK_SOURCE_SIZE = 2

params.RUN_TYPE = 0
params.HEALPIX_SEARCH = True

# Return tracklets with at least this number of detections ("pairs") from radius search as link candidates
params.MIN_LINK_SEARCH_SIZE = 2

params.CLUSTERING_SIZE_MIN = 3

# THIS OVERRIDES OTHER SELECTION CRITERIA

# LSST efficiency test?
params.LSST_EFFICIENCY_TEST = False

params.ORBIT_FILTER = False

# Per M. Holman, assumed distance for LSST simulation 
params.DISTANCE_ASSUMED = 1.5
multi_night_run.set_parameters(params)


# Load the run data
# ----------------------------------------------------------------
# Data Loading Option 1 -- load exposures and detections FROM SQLite db TO pytrax
#(order of magnitude faster than loading from dataframes)
multi_night_run.clear()
multi_night_run.load()


# Examine the loaed data
# ----------------------------------------------------------------
# show # of exposures and detections loaded
print ("Number of exposures: ", multi_night_run.num_exposures()  )
print ("Number of detections: ", multi_night_run.num_detections())

# show the number of nights of observations in the data universe for this run
print( "Number of nights : " , multi_night_run.num_nights() )


# Re-calculate data universe census


# initialize efficiency variables - execute this cell before each run
object_detection_counts = dict()
object_detection_counts['S0'] = 0
object_detection_counts['S1'] = 0
object_detection_counts['FD'] = 0
object_detection_counts['NS'] = 0

real_objects = dict() # raw
neo_objects = dict()
mba_objects = dict()
neo_objects_master = dict()
mba_objects_master = dict()

night_object_detections = dict() # counts of detections per object per night

should_have_tracklets = 0 # num of objects whose detections comprise one or more tracklets
should_make_tracklets = 0 # num of tracklets that should be created for objects with one or more tracklets
object_type_tracklet_total = dict()
object_type_tracklet_total['S0'] = 0 # holds # of NEO tracklets
object_type_tracklet_total['S1'] = 0 # holds # of MBA tracklets


# structure containers for identifying qualifying real objects
#  - This is creating "zeroed" dictionaries
for nx in range(multi_night_run.num_detections()):
    obj_name = multi_night_run.detection(nx).object_name()
    object_detection_counts[obj_name[:2]] += 1
    if obj_name[0] == 'S': # <<--Identify real objects (NEO & MBA)
        real_objects[obj_name] = np.zeros(multi_night_run.num_nights())
        night_object_detections[obj_name] = np.zeros(multi_night_run.num_nights())
        if obj_name[:2] == "S0": # <<--Identify NEOs
            neo_objects[obj_name] = np.zeros(multi_night_run.num_nights())
        else:                    # <<--Identify MBAs
            mba_objects[obj_name] = np.zeros(multi_night_run.num_nights())
            

# Increment object detection count per night    
for ny in range(multi_night_run.num_detections()):
    obj_name = multi_night_run.detection(ny).object_name()
    if obj_name[0] == 'S':
        night_object_detections[obj_name][multi_night_run.detection(ny).night] += 1
        if obj_name[1] == '0':
            neo_objects[obj_name][multi_night_run.detection(ny).night] += 1
        else:
            mba_objects[obj_name][multi_night_run.detection(ny).night] += 1


# Populate a dict keyed on the object names of all NEOs that should become tracklets, initial value = 0            
for key, value in neo_objects.items():
    if sum(value) >= 2:
        neo_objects_master[key] = 0
for key, value in mba_objects.items():
    if sum(value) >= 2:
        mba_objects_master[key] = 0
        
# num of neos that should become tracklets
print (len(neo_objects_master))
print (len(mba_objects_master))

# Evaluate object detection counts
# - want the total number of real objects with sufficient detections to be made into a tracklet
# - (i.e. at least 2 in a night)
# - At this stage we are just MAKING TRACKLETS, so we don't care (yet) about clustering. 
neo_object_count = 0
mba_object_count = 0
neo_object_detectable_count = 0
mba_object_detectable_count = 0
neo_object_detectable_detection_count = 0
mba_object_detectable_detection_count = 0


# how many objects of what kind are in the d.u.?
# how many should have tracklets? How many tracklets should they have?
for key, value in night_object_detections.items():
        
    night_count = 0
    should_have_tracklet = 0
    
    if key[:2] == "S0":
        neo_object_count += 1
    else :
        mba_object_count += 1
    
    for this_night in value:
        if this_night >= 2: # i.e. there are at least 2 detections of the object this night - should become tracklet
            
            should_have_tracklet = 1
            should_make_tracklets += 1
            object_type_tracklet_total[key[:2]] += 1 # separate NEO and MBA tracklet tally
            
            if key[:2] == "S0":
                
                neo_object_detectable_detection_count += this_night
            else:
                mba_object_detectable_detection_count += this_night
                
                        
    if should_have_tracklet == 1: # at least 1 tracklet should be made
            
            should_have_tracklets += 1
            if key[:2] == "S0":
                neo_object_detectable_count += 1
            else :
                mba_object_detectable_count += 1


t1=time.clock()  


### MAKE TRACKLETS ###

# Set tracklet-creation parameters ...
# ------------------------------------
params.DISTANCE_ASSUMED=1.6
params.CHI2_THRESH=3.0
params.RESIDUAL_LIMIT = 0.3
params.DMAG_LIMIT = 1
params.DMAG_LIMIT_SAME = 0.5
params.VELOCITY_FACTOR = 5.0
#params.VELOCITY_MAX = 5.

print(  params.DISTANCE_ASSUMED,
        params.CHI2_THRESH,
        params.RESIDUAL_LIMIT,
        params.DMAG_LIMIT,
        params.DMAG_LIMIT_SAME ,
        params.VELOCITY_FACTOR
)


# Make tracklets for each night
# ------------------------------------
for n in range(multi_night_run.num_nights()):
    print(n)
    multi_night_run.make_tracklets(n)
    
print ("Tracklets : " + str(multi_night_run.num_tracklets()))

# CLUSTERING
# statistics on # of objects in the data universe and # should become tracklets and clusters

objects = dict([]) # raw
night_objects = dict([]) # counts of detections per object per night
efficiency_test_base = dict() # used in plots to measure efficiency in finding real objects

all_nights = 0
two_nights = 0
one_night = 0
should_have_objects = 0
should_make_tracklets = 0


# structure containers for identifying qualifying real objects
for n in range(multi_night_run.num_detections()):
    if multi_night_run.detection(n).object_name()[0] == 'S':
        objects[multi_night_run.detection(n).object_name()] = np.zeros(multi_night_run.num_nights())
        night_objects[multi_night_run.detection(n).object_name()] = np.zeros(multi_night_run.num_nights())
    

# Increment object detection count per night    
for n in range(multi_night_run.num_detections()):
    if multi_night_run.detection(n).object_name()[0] == 'S':
        night_objects[multi_night_run.detection(n).object_name()][multi_night_run.detection(n).night] += 1

# Evaluate object detection counts
# We want the total number of real objects with sufficient detections on sufficient nights that they should ...
# have become tracklets (i.e. at least 2 in a night) and then clustered (at least three tracklets on at least ...
# three nights)
for key, value in night_objects.items():
    
    should_make_cluster = 0    
    
    for this_night in value:
        if this_night >= 2: # i.e. there are at least 2 detections of the object this night - should become tracklet
            should_make_tracklets += 1
            should_make_cluster += 1
            
    if should_make_cluster >= 3: # at least 3 tracklets should be made, so it should be clustered
            should_have_objects += 1
     
    
# of the tracklets that were actually made, how many objects appear in at least three separate nights?   
for x in multi_night_run.tracklets():
    if multi_night_run.tracklet(x).same_object():
        key = multi_night_run.tracklet(x).detection(0).object_name()
    
        valno = multi_night_run.tracklet(x).first_night()
    
        # at this point we only care whether the object has at least one tracklet in the night
        objects[key][valno] = 1
    
#********************
# efficiency_test_base USED IN LATER CELLS TO GAUGE EFFICIENCY
#**********************

efficiency_test_base = dict()

for key, value in objects.items():
    num_nights = np.sum(value)
    if num_nights >= 3:
        all_nights += 1
        efficiency_test_base[key] = 0
        
    if num_nights == 2:
        two_nights += 1
    if num_nights == 1:
        one_night += 1
        
# Denominator(s) for clustering completion/efficiency calculations
num_real_objects = len(efficiency_test_base) 
num_real_NEOs    = len([1 for k,v in efficiency_test_base.items() if k[:2]=='S0']) 
num_real_MBAs    = len([1 for k,v in efficiency_test_base.items() if k[:2]=='S1'])  

# could use 'should_have_objects' as denominator for efficiency of full process (i.e. tracklets + clustering)


print ("# real objects in data universe that appear in any form : ", len(objects))
print ("# real objects in data universe that 'should' become tracklet/arrows AND clustered : ", should_have_objects)
print ("# tracklets that 'should' be made from those real objects : ", should_make_tracklets)
print ("# real objects appearing in (3+ nights) of actual tracklets made that should be clustered : ", all_nights)
#print ("% real objects in data universe that should cluster that appear in in (3+ nights) of actual tracklets made : ", int( 100. * (all_nights / should_have_objects)),"%")
print ("# real objects appearing on two and one nights respectively in actual tracklets : ", two_nights, one_night)

print('num_real_objects', num_real_objects)
print('num_real_NEOs',num_real_NEOs)
print('num_real_MBAs',num_real_MBAs)

# Plot pure, pure+valid, and erroneous clusters and percent pure by cluster radius
# For one fixed gamma gamma_dot pair
# also plot efficiency unmarginalized by time transform and cluster radii.

# Do some timing analysis 
# print (datetime.datetime.now())
# sys.stdout.flush()
# start = time.time()


t2=time.clock()

### CLUSTERING ###

# Get useful params ...
params = multi_night_run.get_parameters()

# fixed gamma and gamma dot
# -----------------------------------------
# gamma = 1. / params.DISTANCE_ASSUMED
gamma = 0.4
gamma_dot = 0.
gamma_title = "gamma = " + str(gamma) + " : gamma_dot = "  + str(-gamma_dot)

params.GAMMA_GAMMA_DOT_ASSUMED = [[gamma, gamma_dot]]

# Values of the hyperparameters ...
# ... "time-transform" and "CLUSTERING_RADIUS" ...
# ... that we will iterate over 
# -----------------------------------------
timeTransformValues = [5] #range(1,11,1)
clusteringRadii = [0.004] #10.**np.arange(-3.9, -2.0, 0.1)

print("timeTransformValues", timeTransformValues)
print("clusteringRadii",clusteringRadii )

#SINGEL FIXED GAMMA GAMMADOT PAIR

# step through TIME_TRANSFORM values 
# -----------------------------------------
for t in timeTransformValues:

    params.CLUSTERING_TIME_TRANSFORM = t
    multi_night_run.set_parameters(params)
    
    # Make arrows from tracklets 
    multi_night_run.make_arrows()

    # Step through clustering radii
    # -------------------------------------
    for cR in clusteringRadii:
        print("t,cR = ", t,cR)
        
        tally = np.zeros(4)
        
        # Set params
        params.CLUSTERING_RADIUS = cR
        multi_night_run.set_parameters(params)
        
        # Do clustering
        multi_night_run.make_clusters()
        these_clusters = multi_night_run.clusters()

        t3=time.clock()
        
   ### ANALYSIS ###

        # initialize the object-finding efficiency check container
        this_eff_chk_pure = efficiency_test_base.copy()
        this_eff_chk_valid = efficiency_test_base.copy()

        # loop to check purity of each cluster
        # - i.e. this is using labelled/training data qualities
        for n in these_clusters:
            this_quality = multi_night_run.cluster(n).quality()
            tally[this_quality] +=1;
            #tally[3] += this_run.cluster(n).time_collision_product();
            
            if this_quality == 0: # pure
                this_name = multi_night_run.cluster(n).object_name()
                if this_name in this_eff_chk_pure:

                    # cluster contains a pure arrow for a real object
                    this_eff_chk_pure[this_name] = 1
                    this_eff_chk_valid[this_name] = 1
            
            if this_quality == 1: # valid
                
                for this_name in multi_night_run.cluster(n).object_names():
                    if this_name in this_eff_chk_valid:

                    # cluster contains a valid sub-cluster of arrows for a real object
                        this_eff_chk_valid[this_name] = 1

        # Do top-level calculations of cluster completeness, etc
        comp_pure = 100. * (sum(this_eff_chk_pure.values()) / num_real_objects)
        comp_pure_valid = 100. * (sum(this_eff_chk_valid.values()) / num_real_objects)

        # Completeness by NEO/MBA
        NEOcomp_p  = 100. * (sum( [v for k,v in this_eff_chk_pure.items()  if k[:2]=="S0"] ) / num_real_NEOs)
        NEOcomp_pv = 100. * (sum( [v for k,v in this_eff_chk_valid.items() if k[:2]=="S0"] ) / num_real_NEOs)
        MBAcomp_p  = 100. * (sum( [v for k,v in this_eff_chk_pure.items()  if k[:2]=="S1"] ) / num_real_MBAs)
        MBAcomp_pv = 100. * (sum( [v for k,v in this_eff_chk_valid.items() if k[:2]=="S1"] ) / num_real_MBAs)
        
        # Purity
        n_clusters=len(these_clusters)
        comp_purity = 100. * num_real_objects/n_clusters

t4=time.clock()
print('------------------- DONE ---------------------------------------')

# DATA UNIVERSE CENSUS                
# Pretty print ...
# ---------------------------------------------------------------
print("Data Universe:")
print("------------------------------------------")
print("HEALPix : ", universe['healpix_ids'])
print("start /end julian dates : ", universe['start_jd'], " - ", universe['end_jd'])
print("# exposures : ", str(multi_night_run.num_exposures()))
print("------------------------------------------")
print("total detections : ", str(multi_night_run.num_detections()))
print("NEO detections : ", str(object_detection_counts['S0']))
print("MBA detections : ", str(object_detection_counts['S1']))
print("FD detections : ", str(object_detection_counts['FD']))
print("NS detections : ", str(object_detection_counts['NS']))
print("------------------------------------------")
print("unique NEO objects : ", str(neo_object_count))
print("unique MBA objects : ", str(mba_object_count))
print("unique NEO+MBA objects : ", str(neo_object_count + mba_object_count))
print("------------------------------------------")
print("detectable unique NEO objects : ", str(neo_object_detectable_count))
print("detections in detectable unique NEO objects : ", str(neo_object_detectable_detection_count))
print("------------------------------------------")
print("detectable unique MBA objects : ", str(mba_object_detectable_count))
print("detections in detectable unique MBA objects : ", str(mba_object_detectable_detection_count))
print("------------------------------------------")
 

print(' COMPLETENESS ')
print(' NEO pure clusters: %f %%' % NEOcomp_p)
print(' NEO pure + valid clusters: %f %%' % NEOcomp_pv)
print(' MBA pure clusters: %f' % MBAcomp_p)
print(' MBA pure + valid clusters: %f %%' % MBAcomp_pv)

print(' PURITY ')
print(' number of real objects: %d ' % num_real_objects)
print(' number of clusters: %d ' % n_clusters)
print(' number of real objects in clusters : %d %%' % comp_purity)

print(' TIME REQUIRED ')
print(' gather data %f sec' % (t1-t0))
print(' make tracklets %f sec' % (t2-t1))
print(' clustering %f sec' % (t3-t2))
print(' total time %f sec' % (t4-t0))

#print results to output file
name_res=['cmpltns_NEO_p_%','cmpltns_NEO_pv_%','cmpltns_MBA_p_%','cmpltns_MBA_pv_%','num_real_obj','num_clusters','purity_%','t_init_s','t_trklt_s','t_clstr_s','t_tot_s']
res=[NEOcomp_p, NEOcomp_pv,MBAcomp_p,MBAcomp_pv,num_real_objects,n_clusters,comp_purity,(t1-t0),(t2-t1),(t3-t2),(t4-t0)]
resdf=pd.DataFrame(res).T
resdf.columns=name_res
resdf.to_csv(outpath,index=False)
