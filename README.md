# test_pytrax
Performance test of LSST Solar System Processing software (Pytrax)

## Dependencies: 
* pytrax 
* a database of simulated LSST observations (pytrax_LSST.db) currently at NCSA LFS under /project/eggl/pytrax
data/

# Sample Output 
## Parameters: 
clustering radius: 0.004
time transform: 5

## Data Universe:

HEALPix :  [753]

start /end julian dates :  2452424.0  -  2452455.0

exposures :  640

total detections :  2044519

NEO detections :  1318

MBA detections :  420222

FD detections :  1135452

NS detections :  487527

unique NEO objects :  292

unique MBA objects :  65784

unique NEO+MBA objects :  66076

detectable unique NEO objects :  228

detections in detectable unique NEO objects :  1171.0

detectable unique MBA objects :  57014

detections in detectable unique MBA objects :  386377.0

## COMPLETENESS
 NEO pure clusters: 50.000000 %
 
 NEO pure + valid clusters: 68.750000 %
 
 MBA pure clusters: 36.095178
 
 MBA pure + valid clusters: 97.835634 %
 
## PURITY
 number of real objects: 22348 
 
 number of clusters: 130933
 
 number of real objects in clusters : 17 %
 
## TIME REQUIRED (epyc.astro.washington.edu, 1 core)
 gather data 42.010000 sec
 
 make tracklets 228.910000 sec
 
 clustering 14.340000 sec
 
 total time 286.180000 sec
