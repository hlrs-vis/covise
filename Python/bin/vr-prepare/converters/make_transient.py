#
#   converted with COVISE (C) map_converter
#   from t.net
#
# get file names
import os

arguments = os.getenv("CONVERTFILES")
arguments = arguments + ' '

#get scale factor
f = arguments.find(' ')

files = arguments.split('.covise ')[:-1]

# stop if not covise files
for file in files:
    if not file.find('.') == -1 :
        print('ERROR: Files should be covise files')
        sys.exit()

# assure that there are more than 2 files
if len(files)!=2:
    print('USAGE: makeTransient <static.covise> <transient.covise>')
    sys.exit()

#
# create global net
#
theNet = net()
#
# MODULE: RWCovise
#
RWCovise_1 = RWCovise()
theNet.add( RWCovise_1 )
#
# set parameter values 
#
RWCovise_1.set_grid_path( files[0] + '.covise' )
RWCovise_1.set_stepNo( 0 )
RWCovise_1.set_rotate_output( "FALSE" )
RWCovise_1.set_rotation_axis( 3 )
RWCovise_1.set_rot_speed( 2.000000 )
#
# MODULE: RWCovise
#
RWCovise_2 = RWCovise()
theNet.add( RWCovise_2 )
#
# set parameter values 
#
RWCovise_2.set_grid_path( files[1]  + '.covise' )
RWCovise_2.set_stepNo( 0 )
RWCovise_2.set_rotate_output( "FALSE" )
RWCovise_2.set_rotation_axis( 3 )
RWCovise_2.set_rot_speed( 2.000000 )
#
# MODULE: MakeTransient
#
MakeTransient_1 = MakeTransient()
theNet.add( MakeTransient_1 )
#
# set parameter values 
#
MakeTransient_1.set_timesteps( 72 )
#
# MODULE: RWCovise
#
RWCovise_3 = RWCovise()
theNet.add( RWCovise_3 )
#
# set parameter values 
#
RWCovise_3.set_grid_path( files[0]+ 'T.covise'  )
RWCovise_3.set_stepNo( 0 )
RWCovise_3.set_rotate_output( "FALSE" )
RWCovise_3.set_rotation_axis( 3 )
RWCovise_3.set_rot_speed( 2.000000 )
#
# CONNECTIONS
#
theNet.connect( RWCovise_1, "mesh", MakeTransient_1, "inport" )
theNet.connect( RWCovise_2, "mesh", MakeTransient_1, "accordingTo" )
theNet.connect( MakeTransient_1, "outport", RWCovise_3, "mesh_in" )
#
# uncomment the following line if you want your script to be executed after loading
#
runMap()
# wait till map is executed
theNet.finishedBarrier()

#
# uncomment the following line if you want exit the COVISE-Python interface
#
sys.exit()
