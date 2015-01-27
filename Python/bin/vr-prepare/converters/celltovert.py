#
#   converted with COVISE (C) map_converter
#   from convert.net
#

import os
import time

# get file names
names = os.getenv("CONVERTFILES")
names = names + ' '
outfiles=['']
files = names.split('.covise ')[:-1]

# stop if not covise files
for file in files:
    if not file.find('.') == -1 :
        print('ERROR: Files should be covise files')
        sys.exit()

# assure that there are more than 2 files
if len(files)<2:
    print('USAGE: celltovert <gridfile.covise> <file1.covise> <file2.covise> ... <fileN.covise>')
    sys.exit()

# create global net
theNet = net()

# MODULE: CellToVert
CellToVert_1 = CellToVert()
theNet.add( CellToVert_1 )

# set parameter values 
CellToVert_1.set_algorithm( 1 )

# MODULE: RWCovise
RWCovise_1 = RWCovise()
theNet.add( RWCovise_1 )

# set parameter values 
RWCovise_1.set_grid_path( files[0] + '.covise' )
RWCovise_1.set_stepNo( 0 )
RWCovise_1.set_rotate_output( "FALSE" )
RWCovise_1.set_rotation_axis( 3 )
RWCovise_1.set_rot_speed( 2.000000 )

# del first filename to be able to loop through the data filenames
gridfile = files[0]
del files[0]

# MODULE: RWCovise
RWCovise_2 = RWCovise()
theNet.add( RWCovise_2 )

# set parameter values 
RWCovise_2.set_stepNo( 0 )
RWCovise_2.set_rotate_output( "FALSE" )
RWCovise_2.set_rotation_axis( 3 )
RWCovise_2.set_rot_speed( 2.000000 )

# MODULE: RWCovise
RWCovise_3 = RWCovise()
theNet.add( RWCovise_3 )

# set parameter values 
# output name is input name with N (for node)
RWCovise_3.set_stepNo( 0 )
RWCovise_3.set_rotate_output( "FALSE" )
RWCovise_3.set_rotation_axis( 3 )
RWCovise_3.set_rot_speed( 2.000000 )

# CONNECTIONS
theNet.connect( CellToVert_1, "data_out", RWCovise_3, "mesh_in" )
theNet.connect( RWCovise_1, "mesh", CellToVert_1, "grid_in" )
theNet.connect( RWCovise_2, "mesh", CellToVert_1, "data_in" )
    
# loop through the data files
for file in files:
    RWCovise_2.set_grid_path( file + '.covise' )
    RWCovise_3.set_grid_path( file + 'N.covise' )
    
    # execute
    runMap()
    
    # wait till map is executed
    theNet.finishedBarrier()

# making grid transient

# MODULE: RWCovise
RWCovise_4 = RWCovise()
theNet.add( RWCovise_4 )

# set parameter values 
RWCovise_4.set_grid_path( files[0] + '.covise' )
RWCovise_4.set_stepNo( 0 )
RWCovise_4.set_rotate_output( "FALSE" )
RWCovise_4.set_rotation_axis( 3 )
RWCovise_4.set_rot_speed( 2.000000 )

# execute
runMap()

# wait till map is executed
theNet.finishedBarrier()

#exit COVISE-Python interface
sys.exit()
