#
#   converted with COVISE (C) map_converter
#   from transform.net
#

import os

# get file names
names = os.getenv("CONVERTFILES")

# get first file name
files = names.split('.covise ')

# there should be 3 files
if len(files) != 3:
    print('USAGE: transform <gridfile.covise> <datafile.covise> <transformfile.euc>')
    sys.exit()

#last one should be .euc file
f = files[2].find('.euc')
if f != len(files[2])-4 :
    print('USAGE: transform <gridfile.covise> <datafile.covise> <transformfile.euc>')
    sys.exit()
else :
    files[2] = files[2][:f]

# stop if not right files
for file in files:
    if not file.find('.') == -1 :
        print('USAGE: transform <gridfile.covise> <datafile.covise> <transformfile.euc>')
        sys.exit()

# create global net
theNet = net()

# MODULE: RWCovise
RWCovise_1 = RWCovise()
theNet.add( RWCovise_1 )

# set parameter values 
RWCovise_1.set_grid_path( files[0] + '.covise' )
RWCovise_1.set_stepNo( 0 )
RWCovise_1.set_rotate_output( "FALSE" )
RWCovise_1.set_rotation_axis( 3 )
RWCovise_1.set_rot_speed( 2.000000 )

# MODULE: RWCovise
RWCovise_2 = RWCovise()
theNet.add( RWCovise_2 )

# set parameter values 
RWCovise_2.set_grid_path( files[1] + '.covise' )
RWCovise_2.set_stepNo( 0 )
RWCovise_2.set_rotate_output( "FALSE" )
RWCovise_2.set_rotation_axis( 3 )
RWCovise_2.set_rot_speed( 2.000000 )

# MODULE: Transform
Transform_1 = Transform()
theNet.add( Transform_1 )

# set parameter values 
Transform_1.set_Transform( 8 )
Transform_1.set_normal_of_mirror_plane( 0, 0, 1. )
Transform_1.set_distance_to_origin( 0.000000 )
Transform_1.set_MirroredAndOriginal( "TRUE" )
Transform_1.set_vector_of_translation( 0, 0, 0. )
Transform_1.set_axis_of_rotation( 0, 0, 1. )
Transform_1.set_one_point_on_the_axis( 0, 0, 0. )
Transform_1.set_angle_of_rotation( 1.000000 )
Transform_1.set_scale_type( 1 )
Transform_1.set_scaling_factor( 1.000000 )
Transform_1.set_new_origin( 0, 0, 0. )
Transform_1.set_axis_of_multirotation( 0, 0, 1. )
Transform_1.set__one_point_on_the_axis( 0, 0, 0. )
Transform_1.set_angle_of_multirotation( 1.000000 )
Transform_1.set_number_of_rotations( 1 )
Transform_1.set_TilingPlane( 1 )
Transform_1.set_flipTile( "TRUE" )
Transform_1.set_TilingMin( 0, 0, 0.0 )
Transform_1.set_TilingMax( 3, 3, 0.0 )
Transform_1.set_EUC_file( files[2]  + '.euc'  )
Transform_1.set_InDataType_0( 1 )
Transform_1.set_InDataType_1( 1 )
Transform_1.set_InDataType_2( 1 )
Transform_1.set_InDataType_3( 1 )
Transform_1.set_createSet( "TRUE" )

# MODULE: RWCovise
RWCovise_3 = RWCovise()
theNet.add( RWCovise_3 )

# set parameter values 
RWCovise_3.set_grid_path( "." )
RWCovise_3.set_stepNo( 0 )
RWCovise_3.set_rotate_output( "FALSE" )
RWCovise_3.set_rotation_axis( 3 )
RWCovise_3.set_rot_speed( 2.000000 )

# MODULE: RWCovise
RWCovise_4 = RWCovise()
theNet.add( RWCovise_4 )

# set parameter values 
RWCovise_4.set_grid_path( "." )
RWCovise_4.set_stepNo( 0 )
RWCovise_4.set_rotate_output( "FALSE" )
RWCovise_4.set_rotation_axis( 3 )
RWCovise_4.set_rot_speed( 2.000000 )

# CONNECTIONS
theNet.connect( RWCovise_1, "mesh", Transform_1, "data_in0" )
theNet.connect( RWCovise_2, "mesh", Transform_1, "geo_in" )
theNet.connect( Transform_1, "geo_out", RWCovise_3, "mesh_in" )
theNet.connect( Transform_1, "data_out0", RWCovise_4, "mesh_in" )

# execute
runMap()

# wait till map is executed
theNet.finishedBarrier()

#exit COVISE-Python interface
sys.exit()
