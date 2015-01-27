#
#   converted with COVISE (C) map_converter
#   from calc.net
#

import os

# get file names
arguments = os.getenv("CONVERTFILES")
arguments = arguments + ' '

#get scale factor
f = arguments.find(' ')
scale = arguments[:f]

files = arguments[f+1:].split('.covise ')[:-1]

# stop if not covise files
for file in files:
    if not file.find('.') == -1 :
        print('ERROR: Files should be covise files')
        sys.exit()

# assure that there are more than 2 files
if len(files)<1:
    print('USAGE: scaleVector <scalingfactor> <file1.covise> ... <fileN.covise>')
    sys.exit()

# create global net
theNet = net()
for file in files:
    
    # MODULE: RWCovise
    RWCovise_1 = RWCovise()
    theNet.add( RWCovise_1 )
    
    # set parameter values 
    RWCovise_1.set_grid_path( file + '.covise' )
    RWCovise_1.set_stepNo( 0 )
    RWCovise_1.set_rotate_output( "FALSE" )
    RWCovise_1.set_rotation_axis( 3 )
    RWCovise_1.set_rot_speed( 2.000000 )
    
    # MODULE: RWCovise
    RWCovise_2 = RWCovise()
    theNet.add( RWCovise_2 )
    
    # set parameter values 
    RWCovise_2.set_grid_path( file + scale +'.covise' )
    RWCovise_2.set_stepNo( 0 )
    RWCovise_2.set_rotate_output( "FALSE" )
    RWCovise_2.set_rotation_axis( 3 )
    RWCovise_2.set_rot_speed( 2.000000 )
    
    # MODULE: Calc
    Calc_1 = Calc()
    theNet.add( Calc_1 )
    
    # set parameter values 
    Calc_1.set_expression( 'v1*'+scale )

    # CONNECTIONS
    theNet.connect( RWCovise_1, "mesh", Calc_1, "v_indata1" )
    theNet.connect( Calc_1, "outdata2", RWCovise_2, "mesh_in" )
    
    # execute
    runMap()
    
    # wait till map is executed
    theNet.finishedBarrier()


sys.exit()
