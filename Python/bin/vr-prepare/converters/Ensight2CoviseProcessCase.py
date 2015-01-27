from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from paramAction import NotifyAction
from ChoiceGetterAction import ChoiceGetterAction

import os
import os.path

logFile.write("\nConverting Case:\n")
logFile.write("fullEnsightCaseName = %s\n"%(fullEnsightCaseName,))
logFile.flush()
print("")
print("Converting Case:")
print("fullEnsightCaseName ", fullEnsightCaseName)



scalarVariables3DGetterAction = ChoiceGetterAction()
vectorVariables3DGetterAction = ChoiceGetterAction()
scalarVariables2DGetterAction = ChoiceGetterAction()
vectorVariables2DGetterAction = ChoiceGetterAction()

aPartsCollectorAction = PartsCollectorAction()

#
# MODULE: ReadEnsightNT
#
ReadEnsight_1 = ReadEnsight()
theNet.add( ReadEnsight_1 )

aPartsCollectorAction = PartsCollectorAction()
CoviseMsgLoop().register(aPartsCollectorAction)

#
# hang in variable-getters
#
ReadEnsight_1.addNotifier('data_for_sdata1_3D', scalarVariables3DGetterAction)
ReadEnsight_1.addNotifier('data_for_vdata1_3D', vectorVariables3DGetterAction)
ReadEnsight_1.addNotifier('data_for_sdata1_2D', scalarVariables2DGetterAction)
ReadEnsight_1.addNotifier('data_for_vdata1_2D', vectorVariables2DGetterAction)

#
# set parameter values
#
ReadEnsight_1.set_data_byte_swap( byteswap )
ReadEnsight_1.set_case_file( fullEnsightCaseName )
ReadEnsight_1.set_include_polyhedra( "TRUE")

#ReadEnsight_1.set_data_for_sdata1_3D( 0 )
#ReadEnsight_1.set_data_for_vdata1_3D( 0 )
#ReadEnsight_1.set_data_for_sdata3_2D( 0 )
#ReadEnsight_1.set_data_for_vdata1_2D( 0 )
#ReadEnsight_1.set_data_for_vdata2_2D( 0 )
#ReadEnsight_1.set_choose_parts( "all" )
#ReadEnsight_1.set_repair_connectivity( "FALSE" )
#ReadEnsight_1.set_enable_autocoloring( "FALSE" )
#ReadEnsight_1.set_store_covgrp( "FALSE" )


#
# wait for choices to be updated
#
scalarVariables3DGetterAction.waitForChoices()
vectorVariables3DGetterAction.waitForChoices()
scalarVariables2DGetterAction.waitForChoices()
vectorVariables2DGetterAction.waitForChoices()

#
# wait for the part info message
#
aPartsCollectorAction.waitForPartsinfoFinished()


# get variables
scalarVariables3D=scalarVariables3DGetterAction.getChoices()
vectorVariables3D=vectorVariables3DGetterAction.getChoices()
scalarVariables2D=scalarVariables2DGetterAction.getChoices()
vectorVariables2D=vectorVariables2DGetterAction.getChoices()


#
# Module Transform
#
if scale == 1:
    scale=None
if scale:
    Transform_1 = Transform()
    theNet.add( Transform_1 )
    Transform_1.set_Transform( 5 )
    Transform_1.set_scaling_factor( scale )
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



#
# 3D parts
#
logFile.write("\nConverting 3D parts\n")
logFile.flush()
print("Converting 3D parts")

for partid in aPartsCollectorAction.getRefNameDict3dParts().keys():
    if int(partid) >= int(startId):
        # get partname
        partname = aPartsCollectorAction.getRefNameDict3dParts()[partid]
        # write logfile
        logFile.write("Part: id = %d name = %s\n"%(partid, partname))
        logFile.flush()
        print("Part id = %d name = %s"%(partid, partname))
        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.connect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "geoOut_3D", RWCovise_1, "mesh_in" )
        # select part
        ReadEnsight_1.set_choose_parts( str(partid) )
        # clean partname
        if "/" in partname:
            logFile.write("\t! Attention: Removing the / in partname = %s\n"%(partname,))
            logFile.flush()
            print("\t! Attention: Removing the / in partname = ", partname)
            partname=partname.replace("/","")
        # create RW Covise name
        covisename = partname + "-3D.covise"
        # check if file is already available
        counter=0
        while os.path.isfile(covisename):
            print("\t! Info: a file with this name is already available trying a new name")
            counter=counter+1
            covisename = partname + str(counter) + "-3D.covise"
            
        RWCovise_1.set_grid_path( covisename )
        # execute
        runMap()
        theNet.finishedBarrier()
        
        theNet.save( "grid.net" )


        # write logfile
        logFile.write("\tgrid: %s\n"%(covisename,))
        logFile.flush()
        print("\tgrid: ",covisename)
        # create cocase item
        item3D = CoviseCaseFileItem(partname, GEOMETRY_3D, covisename)
        # print(memory usage of module)
        #os.system('ps aux | grep ReadEnsightNT')
        # disconnect modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.disconnect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "geoOut_3D", RWCovise_1, "mesh_in" )

        #
        # scalar variables
        #

        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.connect( ReadEnsight_1, "sdata1_3D", Transform_1, "data_in0" )
            theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "sdata1_3D", RWCovise_1, "mesh_in" )
        choice=1
        for svar in scalarVariables3D:
            # select variable
            choice+=1
            ReadEnsight_1.set_data_for_sdata1_3D( choice )
            # clean variablename
            if "/" in svar:
                logFile.write("\t! Attention: Removing the / in svar = %s\n"%(svar,))
                logFile.flush()
                print("\t! Attention: Removing the / in svar = ", svar)
                svar=svar.replace("/","")
            # create RWCovise name
            covisename = partname + "-" + svar + "-3D.covise"
            # check if file is already available
            counter=0
            while os.path.isfile(covisename):
                print("\t! Info: a file with this name is already available trying a new name")
                counter=counter+1
                covisename = partname + str(counter) + "-" + svar + "-3D.covise"
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logfile
            logFile.write("\tscalar variable: %s\n"%(covisename,))
            logFile.flush()
            print("\tscalar variable: ",covisename)
            # add variable to cacase item
            item3D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
            # print(memory usage of module)
            #os.system('ps aux | grep ReadEnsight_1')
            theNet.save( "scalar.net" )

        # disconnect modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.disconnect( ReadEnsight_1, "sdata1_3D", Transform_1, "data_in0" )
            theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "sdata1_3D", RWCovise_1, "mesh_in" )

        #
        # vector variables
        #

        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.connect( ReadEnsight_1, "vdata1_3D", Transform_1, "data_in0" )
            theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "vdata1_3D", RWCovise_1, "mesh_in" )

        choice=1
        for vvar in vectorVariables3D:
            # select variable
            choice+=1
            ReadEnsight_1.set_data_for_vdata1_3D( choice )
            # clean variablename
            if "/" in vvar:
                logFile.write("\t! Attention: Removing the / in vvar = %s\n"%(vvar,))
                logFile.flush()
                print("\t! Attention: Removing the / in vvar = ", vvar)
                partname=partname.replace("/","")
            # create covisename
            covisename = partname + "-" + vvar + "-3D.covise"
            # check if file is already available
            counter=0
            while os.path.isfile(covisename):
                print("\t! Info: a file with this name is already available trying a new name")
                counter=counter+1
                covisename = partname + str(counter) + "-" + vvar + "-3D.covise"
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logfile
            logFile.write("\tvector variable: %s\n"%(covisename,))
            logFile.flush()
            print("\tvector variable:",covisename)
            # add variable to cocase item
            item3D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
            # print(memory usage of module)
            #os.system('ps aux | grep ReadEnsight_1')
 
        # disconnect modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_3D", Transform_1, "geo_in" )
            theNet.disconnect( ReadEnsight_1, "vdata1_3D", Transform_1, "data_in0" )
            theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "vdata1_3D", RWCovise_1, "mesh_in" )

        # add the cocase item to the case file
        cocase.add(item3D)

logFile.write("Conversion of 3D parts finished\n")
logFile.flush()
print("Conversion of 3D parts finished")

#
# l2D part
#
logFile.write("\nConverting 2D parts\n")
logFile.flush()
print("Converting 2D parts")

for partid in aPartsCollectorAction.getRefNameDict2dParts().keys():
    if int(partid) >= int(startId):
        partname = aPartsCollectorAction.getRefNameDict2dParts()[partid]
         # write logfile
        logFile.write("Part id = %d name = %s\n"%(partid, partname))
        logFile.flush()
        print("Part id = %d name = %s"%(partid, partname))
        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.connect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "geoOut_2D", RWCovise_1, "mesh_in" )
        # select part
        ReadEnsight_1.set_choose_parts( str(partid) )
        # clean partname
        if "/" in partname:
            logFile.write("\t! Attention: Removing the / in partname = %s\n"%(partname,))
            logFile.flush()
            print("\t! Attention: Removing the / in partname = ", partname)
            partname=partname.replace("/","")
        # create RWCovise name
        covisename = partname + "-2D.covise"
        RWCovise_1.set_grid_path( covisename )
        # execute
        runMap()
        theNet.finishedBarrier()
        # write logfile
        logFile.write("\tsurface: %s\n"%(covisename,))
        logFile.flush()
        print("\tsurface: ",covisename)
        # create cocase item
        item2D = CoviseCaseFileItem(partname, GEOMETRY_2D, covisename)
        # print(memory usage of module)
        #os.system('ps aux | grep ReadEnsight_1')

        # disconnect the modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.disconnect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "geoOut_2D", RWCovise_1, "mesh_in" )

        #
        # scalar variables
        #

        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.connect( ReadEnsight_1, "sdata1_2D", Transform_1, "data_in0" )
            theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "sdata1_2D", RWCovise_1, "mesh_in" )

        choice=1
        for svar in scalarVariables2D:
            # select variable
            choice+=1
            ReadEnsight_1.set_data_for_sdata1_2D( choice )
            # clean variablename
            if "/" in partname:
                logFile.write("\t! Attention: Removing the / in partname = %s\n"%(partname,))
                logFile.flush()
                print("\t! Attention: Removing the / in partname = ", partname)
                partname=partname.replace("/","")
            # create RWCovise name
            covisename = partname + "-" + svar + "-2D.covise"
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logfile
            logFile.write("\tscalar variable: %s\n"%(covisename,))
            logFile.flush()
            print("\tscalar variable: ",covisename)
            # add variable to cicase item
            item2D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
            # print(memory usage of module)
            #os.system('ps aux | grep ReadEnsight_1')

        # disconnect modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.disconnect( ReadEnsight_1, "sdata1_2D", Transform_1, "data_in0" )
            theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "sdata1_2D", RWCovise_1, "mesh_in" )

        #
        #  vector variables
        #


        # connect modules
        if scale:
            theNet.connect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.connect( ReadEnsight_1, "vdata1_2D", Transform_1, "data_in0" )
            theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.connect( ReadEnsight_1, "vdata1_2D", RWCovise_1, "mesh_in" )
        choice=1
        for vvar in vectorVariables2D:
            # select variable
            choice+=1
            ReadEnsight_1.set_data_for_vdata1_2D( choice )
            # clean partname
            if "/" in partname:
                logFile.write("\t! Attention: Removing the / in partname = %s\n"%(partname,))
                logFile.flush()
                print("\t! Attention: Removing the / in partname = ", partname)
                partname=partname.replace("/","")
            # create RWCovise name
            covisename = partname + "-" + vvar + "-2D.covise"
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logfile
            logFile.write("\tvector variable: %s\n"%(covisename,))
            logFile.flush()
            print("\tvector variable:",covisename)
            # add varibale to coscase item
            item2D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
            # print(memory usage of module)
            # os.system('ps aux | grep ReadEnsight_1')

        # disconnect modules
        if scale:
            theNet.disconnect( ReadEnsight_1, "geoOut_2D", Transform_1, "geo_in" )
            theNet.disconnect( ReadEnsight_1, "vdata1_2D", Transform_1, "data_in0" )
            theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadEnsight_1, "vdata1_2D", RWCovise_1, "mesh_in" )

        # add the cocase item to the case file
        cocase.add(item2D)

CoviseMsgLoop().unregister(aPartsCollectorAction)

theNet.remove( ReadEnsight_1 )
logFile.write("Conversion of 2D parts finished\n")
logFile.flush()
print("Conversion of 2D parts finished")
