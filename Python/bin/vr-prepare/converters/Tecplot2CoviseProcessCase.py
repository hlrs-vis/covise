from ChoiceGetterAction import ChoiceGetterAction
#import time

logFile.write("\nConverting Case:\n")
logFile.write("fullTecplotCaseName = %s\n"%(fullTecplotCaseName,))
logFile.flush()
print("")
print("Converting Case:")
print("fullTecplotCaseName ", fullTecplotCaseName)

gridXGetterAction = ChoiceGetterAction()
gridYGetterAction = ChoiceGetterAction()
gridZ0GetterAction = ChoiceGetterAction()
gridZ1GetterAction = ChoiceGetterAction()

vecXGetterAction = ChoiceGetterAction()
vecYGetterAction = ChoiceGetterAction()

scalar0GetterAction = ChoiceGetterAction()
scalar1GetterAction = ChoiceGetterAction()

#
# MODULE: ReadTecplot
#
ReadTecplot_1 = ReadTecplot()
theNet.add( ReadTecplot_1 )

#
# hang in variable-getters
#
ReadTecplot_1.addNotifier('grid_x', gridXGetterAction)
ReadTecplot_1.addNotifier('grid_y', gridYGetterAction)
ReadTecplot_1.addNotifier('grid_z0', gridZ0GetterAction)
ReadTecplot_1.addNotifier('grid_z1', gridZ1GetterAction)

ReadTecplot_1.addNotifier('vec_x', vecXGetterAction)
ReadTecplot_1.addNotifier('vec_y', vecYGetterAction)

ReadTecplot_1.addNotifier('scalar_0', scalar0GetterAction)
ReadTecplot_1.addNotifier('scalar_1', scalar1GetterAction)

#
# set format
#
ReadTecplot_1.set_format_of_file( int(format) )

#
# set filename
#
ReadTecplot_1.set_fullpath( fullTecplotCaseName )

#
# set scale
# 
ReadTecplot_1.set_scale_z( 1, 2*float(scaleZ), float(scaleZ) )

#
# wait for the choices parameters to be updated by the module
#
gridXGetterAction.waitForChoices()
gridYGetterAction.waitForChoices()
gridZ0GetterAction.waitForChoices()
gridZ1GetterAction.waitForChoices()

vecXGetterAction.waitForChoices()
vecYGetterAction.waitForChoices()

scalar0GetterAction.waitForChoices()
scalar1GetterAction.waitForChoices()


#
# read the variables
#
variables=gridXGetterAction.getChoices()

#
# print(the variables)
#
print("\nlist of all variables:")
for v in variables:
   print(v)

#
# identify variables
#
grid_x=0
grid_y=0
grid_z0=0
grid_z1=0
vel_x=0
vel_y=0

# omit "None"
choice=2
for v in variables:
   if "RECHTSWERT" in v or "X-COORDINATE" in v or "-X-" in v:
      grid_x=choice
      #print("grid_x=", grid_x)
   if "HOCHWERT" in v or "Y-COORDINATE" in v or "-Y-" in v:
      grid_y=choice
      #print("grid_y=", grid_y)
   if "TIME_DEPENDENT_BATHYMETRY" in v or "BOTTOM---" in v or "-Z-" in v:
      grid_z0=choice
      #print("grid_z0=", grid_z0      )
   if "FREE-SURFACE" in v or "-S-" in v: 
      grid_z1=choice
      #print("grid_z1=", grid_z1)
   if "CURRENT_VELOCITY_X" in v or "VELOCITY-U" in v or "-U-" in v:
      vel_x=choice
      #print("vel_x=", vel_x)
   if "CURRENT_VELOCITY_Y" in v or "VELOCITY-V" in v or "-V-" in v:
      vel_y=choice
      #print("vel_y=", vel_y)
   choice+=1

if grid_x == 0 or grid_y == 0 or grid_z0 == 0:
   print("\nERROR: Konnte Variablen fuer Sohle nicht identifizieren"       )
   sys.exit() 

if grid_z1 == 0:
   print("\nWARNING: Konnte Variable fuer Wasserspiegel nicht identifizieren")
  
# write variables in logfile
logFile.write("\nSelected variables for sohle:\n%s,\n%s,\n%s\n"%( variables[grid_x-2], variables[grid_y-2], variables[grid_z0-2]))
logFile.flush()
print("\nSelected variables for sohle:")
print(variables[grid_x-2])
print(variables[grid_y-2])
print(variables[grid_z0-2])

logFile.write("\nSelected variables for wasserspiegel:\n%s,\n%s,\n%s\n"%( variables[grid_x-2], variables[grid_y-2], variables[grid_z1-2]))
logFile.flush()
print("\nSelected variables for wasserspiegel:")
print(variables[grid_x-2])
print(variables[grid_y-2])
print(variables[grid_z1-2])

logFile.write("\nSelected variables for velocity:\n%s,\n%s\n"%( variables[vel_x-2], variables[vel_y-2]))
logFile.flush()
print("\nSelected variables for velocity:")
print(variables[vel_x-2])
print(variables[vel_y-2])

#
# set choices
#
ReadTecplot_1.set_grid_x( grid_x )
ReadTecplot_1.set_grid_y( grid_y )
ReadTecplot_1.set_grid_z0( grid_z0 )
ReadTecplot_1.set_grid_z1( grid_z1 )
ReadTecplot_1.set_vec_x( vel_x )
ReadTecplot_1.set_vec_y( vel_y )
ReadTecplot_1.set_vec_z( 1 )
ReadTecplot_1.set_scalar_0( grid_z0 )
ReadTecplot_1.set_scalar_1( grid_z1 )
ReadTecplot_1.set_scalar_2( 1 )

#
# enable translation in origin, this executes the module
#
ReadTecplot_1.set_auto_trans( "TRUE" )
theNet.finishedBarrier()


#
# Module Transform
#
if waterSurfaceOffset == 0:
    waterSurfaceOffset=None
if waterSurfaceOffset:
    Transform_1 = Transform()
    theNet.add( Transform_1 )
    Transform_1.set_Transform( 3 )
    Transform_1.set_vector_of_translation( 0,0, float(waterSurfaceOffset) )
    Transform_1.set_createSet( "FALSE" )

#
# MODULE: RWCovise
#
RWCovise_1 = RWCovise()
theNet.add( RWCovise_1 )
RWCovise_1.set_stepNo( 0 )
RWCovise_1.set_rotate_output( "FALSE" )
RWCovise_1.set_rotation_axis( 3 )
RWCovise_1.set_rot_speed( 2.000000 )
RWCovise_1.set_grid_path( "gitter_sohle.covise" )

RWCovise_2 = RWCovise()
theNet.add( RWCovise_2 )
RWCovise_2.set_stepNo( 0 )
RWCovise_2.set_rotate_output( "FALSE" )
RWCovise_2.set_rotation_axis( 3 )
RWCovise_2.set_rot_speed( 2.000000 )
RWCovise_2.set_grid_path( "gitter_wasserspiegel.covise" )

RWCovise_3 = RWCovise()
theNet.add( RWCovise_3 )
RWCovise_3.set_stepNo( 0 )
RWCovise_3.set_rotate_output( "FALSE" )
RWCovise_3.set_rotation_axis( 3 )
RWCovise_3.set_rot_speed( 2.000000 )
RWCovise_3.set_grid_path( "velocity.covise" )

RWCovise_4 = RWCovise()
theNet.add( RWCovise_4 )
RWCovise_4.set_stepNo( 0 )
RWCovise_4.set_rotate_output( "FALSE" )
RWCovise_4.set_rotation_axis( 3 )
RWCovise_4.set_rot_speed( 2.000000 )
RWCovise_4.set_grid_path( "sohle.covise" )

RWCovise_5 = RWCovise()
theNet.add( RWCovise_5 )
RWCovise_5.set_stepNo( 0 )
RWCovise_5.set_rotate_output( "FALSE" )
RWCovise_5.set_rotation_axis( 3 )
RWCovise_5.set_rot_speed( 2.000000 )
RWCovise_5.set_grid_path( "wasserspiegel.covise" )


# connect gitter sohle
theNet.connect( ReadTecplot_1, "grid", RWCovise_1, "mesh_in" )

# connect gitter wasserspiegel
if waterSurfaceOffset:
   theNet.connect( ReadTecplot_1, "grid2", Transform_1, "geo_in" )
   theNet.connect( Transform_1, "geo_out", RWCovise_2, "mesh_in" )
else:
   theNet.connect( ReadTecplot_1, "grid2", RWCovise_2, "mesh_in" )

# connect velocity
theNet.connect( ReadTecplot_1, "vector", RWCovise_3, "mesh_in" )

# connect sohle
theNet.connect( ReadTecplot_1, "dataout0", RWCovise_4, "mesh_in" )

# connect wasserspiegel
theNet.connect( ReadTecplot_1, "dataout1", RWCovise_5, "mesh_in" )


# convert
logFile.write("\nconverting ...\n")
logFile.flush()
print("\nconverting ....")

runMap()
theNet.finishedBarrier()

theNet.save( "test.net" )

# write logfile
logFile.write("... conversion successful!\n")
logFile.flush()
print("... conversion successful!")


#
# write cocase file
#
itemSurfaceBottom = CoviseCaseFileItem("bottomSurface", GEOMETRY_2D, "gitter_sohle.covise")
itemSurfaceBottom.addVariableAndFilename("bottom", "sohle.covise", SCALARVARIABLE)
itemSurfaceWater = CoviseCaseFileItem("waterSurface", GEOMETRY_2D, "gitter_wasserspiegel.covise")
itemSurfaceWater.addVariableAndFilename("water", "wasserspiegel.covise", SCALARVARIABLE)
itemSurfaceWater.addVariableAndFilename("velocity", "velocity.covise", VECTOR3DVARIABLE)

cocase.add(itemSurfaceBottom)
cocase.add(itemSurfaceWater)


#
# remove the modules
#
theNet.remove( ReadTecplot_1 )
theNet.remove( RWCovise_1 )
theNet.remove( RWCovise_2 )
theNet.remove( RWCovise_3 )
theNet.remove( RWCovise_4 )
theNet.remove( RWCovise_5 )

# write logfile
logFile.write("Conversion finished\n")
logFile.flush()
print("Conversion finished")

