from ChoiceGetterAction import ChoiceGetterAction
from IntGetterAction import IntGetterAction

#import time

logFile.write("\nConverting File:\n")
logFile.write("fullCfxCaseName = %s\n"%(fullCfxCaseName,))
logFile.flush()
print("")
print("Converting Case:")
print("fullCfxCaseName ", fullCfxCaseName)

domainsGetterAction = ChoiceGetterAction()
regionsGetterAction = ChoiceGetterAction()
boundariesGetterAction = ChoiceGetterAction()
scalarVariablesGetterAction = ChoiceGetterAction()
vectorVariablesGetterAction = ChoiceGetterAction()
timestepsGetterAction = IntGetterAction()
#
# MODULE: ReadCfx
#
ReadCFX_1 = ReadCFX()
theNet.add( ReadCFX_1 )

#
# hang in variable-getters
#
ReadCFX_1.addNotifier('domains', domainsGetterAction)
ReadCFX_1.addNotifier('regions', regionsGetterAction)
ReadCFX_1.addNotifier('boundaries', boundariesGetterAction)
ReadCFX_1.addNotifier('scalar_variables', scalarVariablesGetterAction)
ReadCFX_1.addNotifier('vector_variables', vectorVariablesGetterAction)
ReadCFX_1.addNotifier('timesteps', timestepsGetterAction)

#
# set parameter values
#
ReadCFX_1.set_result( fullCfxCaseName )
ReadCFX_1.set_read_grid("false")
if readTransient=="0":
    ReadCFX_1.set_force_transient_grid("False")
else:
    ReadCFX_1.set_force_transient_grid("True")

#
# wait for the choices parameters to be updated by the module
#
domainsGetterAction.waitForChoices()
regionsGetterAction.waitForChoices()
boundariesGetterAction.waitForChoices()
scalarVariablesGetterAction.waitForChoices()
vectorVariablesGetterAction.waitForChoices()


#
# read the domain values
#
domains=domainsGetterAction.getChoices()


ReadCFX_1.set_regions( 1 )

# force timestep 1
# ReadCFX_1.set_timesteps( 1 )
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



def calculatePDYN(caseFileItem):
    if not calculatePDYN:
        return
    ptot_filename = ""
    pres_filename = ""
    for (varName, varFile, varDimension) in caseFileItem.variables_:
        if (varName == "PTOT") and (varDimension == SCALARVARIABLE):
            ptot_filename = varFile
        if (varName == "PRES") and (varDimension == SCALARVARIABLE):
            pres_filename = varFile
    if (ptot_filename != "") and (pres_filename != ""):
        pdyn_filename = ptot_filename.replace("-PTOT-", "-PDYN-")
        # calculate
        os.spawnlp(os.P_WAIT, "calcCovise", "calcCovise", pdyn_filename, "s1-s2", ptot_filename, pres_filename)
        # add
        caseFileItem.addVariableAndFilename("PDYN", pdyn_filename, SCALARVARIABLE)



def convert(domainname, domainchoice, processGrid, processBoundaries):

    # write logfile
    logFile.write("\nprocessing %s\n"%(domainname))
    logFile.flush()
    print("\nprocessing %s"%(domainname))
    # reset wait variable    
    regionsGetterAction.resetWait()
    boundariesGetterAction.resetWait()
    scalarVariablesGetterAction.resetWait()
    vectorVariablesGetterAction.resetWait()
    # select the domain
    ReadCFX_1.set_domains( domainchoice )
    # wait for choices to be updated by the module
    regionsGetterAction.waitForChoices()
    boundariesGetterAction.waitForChoices()
    scalarVariablesGetterAction.waitForChoices()
    vectorVariablesGetterAction.waitForChoices()
    # read chocies
    regions=regionsGetterAction.getChoices()
    boundaries=boundariesGetterAction.getChoices()
    scalarVariables=scalarVariablesGetterAction.getChoices()
    vectorVariables=vectorVariablesGetterAction.getChoices()
    # reset the region to none
    ReadCFX_1.set_regions( 1 )
    # set the boundaries to all
    # choice param is "None", "All", "Affenfelsen",....
    if processBoundaries:
        ReadCFX_1.set_boundaries( 2 )  
    else:
        ReadCFX_1.set_boundaries( 1 )
    # print(vraiables and boundaries of this domain)
    print("\n\tscalar variables:")
    for s in scalarVariables:
        print("\t",s)
    print("\n\tvector variables:")
    for v in vectorVariables:
        print("\t",v)
    print("\n\tboundaries:")
    for b in boundaries:
        print("\t",b)

    timesteps = timestepsGetterAction.getInt()
    if not timesteps==None:
        print("\n\ttimesteps: ", timesteps        )
        
    # connect grid
    if scale:
        theNet.connect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
        theNet.connect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
    else:
        theNet.connect( ReadCFX_1, "mesh", RWCovise_1, "mesh_in" )

    # set variables to None
    ReadCFX_1.set_scalar_variables( 1 )
    ReadCFX_1.set_vector_variables( 1 )    

    #
    # CONVERT GRID
    #
    covisename = domainname + "-3D.covise"         
    # create the cocase file item
    item3D = None
    if processGrid:
        ReadCFX_1.set_read_grid('TRUE')
        item3D = CoviseCaseFileItem(domainname, GEOMETRY_3D, covisename)  
        # write logfile
        logFile.write("\n\tconverting grid %s ...\n"%(domainname))
        logFile.flush()
        print("\n\tconverting grid", domainname , "...")
        # clean the domainname
        if "/" in domainname:
            logFile.write("\t! Attention: Replacing the / in domainname = %s\n"%(domainname,))
            logFile.flush()
            print("\t! Attention: Replacing the / in domainname = ", domainname)
            domainname=domainname.replace("/","per")
        if "\x7f" in domainname:
            logFile.write("\t! Attention: Replacing a special character in domainname = %s\n"%(domainname,))
            logFile.flush()
            print("\t! Attention: Removing a special character in domainname = ", domainname)
            domainname=domainname.replace("\x7f","_")
        # create the RWCovise name
        RWCovise_1.set_grid_path( covisename )
        # execute
        runMap()     
        theNet.finishedBarrier()
        # write logfile
        logFile.write("\t... conversion successful! File: %s\n"%(covisename,))
        logFile.flush()
        print("\t... conversion successful! File: ",covisename)
        
        # disconnect grid
        if scale:
            theNet.disconnect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
            theNet.disconnect( Transform_1, "geo_out", RWCovise_1, "mesh_in" )
        else:
            theNet.disconnect( ReadCFX_1, "mesh", RWCovise_1, "mesh_in" )
            
        if not scale:    
            ReadCFX_1.set_read_grid("false")
    else:
        item3D = CoviseCaseFileItem(domainname, GEOMETRY_3D, covisename)  
        for item in cocase.items_:
            print("name "+item.name_)
            print("covisename "+domainname)
            if item.name_ == domainname:
                item3D = item
                break             
        runMap()

    #
    # CONVERT BOUNDARIES
    #

    # woraround for bug in readCFX, some parts are not convertes correctly,
    # but "all" parts are allways correct, therefore we use all parts
    # and extract the parts with GetSubset
    item2D={}
    if processBoundaries:
        # 
        # RWCovise
        #
        RWCovise_2 = RWCovise()
        theNet.add( RWCovise_2 )
        RWCovise_2.set_stepNo( 0 )
        RWCovise_2.set_rotate_output( "FALSE" )
        RWCovise_2.set_rotation_axis( 3 )
        RWCovise_2.set_rot_speed( 2.000000 )
        #
        # Module GetSubset
        #
        GetSubset_1 = GetSubset()
        theNet.add( GetSubset_1 )      
    
        #
        # Module FixUsg
        #                 
        FixUsg_1 = FixUsg()
        theNet.add( FixUsg_1 )      

        # transient case
        if not timesteps==None:
            GetSetelem_1 = GetSetelem()
            theNet.add( GetSetelem_1 )
            PipelineCollect_1 = PipelineCollect()
            theNet.add( PipelineCollect_1 )
            
        # connect boundaries
        if timesteps==None:
            theNet.connect( ReadCFX_1, "boundary_mesh", GetSubset_1, "input_0" )      
            theNet.connect( GetSubset_1, "output_0", FixUsg_1, "usg_in" )      
            if scale:
                theNet.connect( FixUsg_1, "usg_out", Transform_1, "geo_in" )
                theNet.connect( Transform_1, "geo_out", RWCovise_2, "mesh_in" )
            else:
                theNet.connect( FixUsg_1, "usg_out", RWCovise_2, "mesh_in" )
        else :
            # scale not supported yet
            theNet.connect( ReadCFX_1, "boundary_mesh", GetSetelem_1, "input_0" )
            theNet.connect( GetSetelem_1, "output_0", GetSubset_1, "input_0" )
            theNet.connect( GetSetelem_1, "output_0", PipelineCollect_1, "inport_0" )
            theNet.connect( GetSubset_1, "output_0",  PipelineCollect_1, "inport_1" )
            theNet.connect( PipelineCollect_1, "outport_1", RWCovise_2, "mesh_in" )
        

        subset=0
        boundchoice=1
        for boundname in boundaries:
            # ommit names "None" "all"
            boundchoice+=1            
            if int(boundchoice) > 2 :
                # write logfile
                logFile.write("\n\tconverting surface %s ...\n"%(boundname,))
                logFile.flush()
                print("\n\tconverting surface %s ..."%(boundname,))
                bname=boundname
                # clean boundname
                if "/" in boundname:
                    logFile.write("\t! Attention: Replacing the / in boundname = %s\n"%(boundname,))
                    logFile.flush()
                    print("\t! Attention: Replacing the / in boundname = ", boundname)
                    bname=boundname.replace("/","per")
                if "\x7f" in boundname:
                    logFile.write("\t! Attention: Replacing a special character in boundname = %s\n"%(boundname,))
                    logFile.flush()
                    print("\t! Attention: Replacing a special character in boundname = ", boundname)
                    bname=boundname.replace("\x7f","_")
                # set the subset
                GetSubset_1.set_selection( str(subset) )
                # create RWCovise name
                covisename = domainname + "-boundary-" + bname + "-2D.covise"
                RWCovise_2.set_grid_path( covisename )
                # execute
                if timesteps==None:
                    GetSubset_1.execute()
                else :
                    GetSetelem_1.set_stepNo(1)
                    GetSetelem_1.execute()    
                theNet.finishedBarrier()
                # write logfile
                logFile.write("\t... conversion successful! Filen: %s\n"%(covisename,))
                logFile.flush()
                print("\t... conversion successful! File: ",covisename)
                # create cocase item
                item2D[boundname] = CoviseCaseFileItem(domainname + "-" + bname, GEOMETRY_2D, covisename)
                subset+=1
            
        # disconnect boundaries
        if timesteps==None:
            theNet.disconnect( ReadCFX_1, "boundary_mesh", GetSubset_1, "input_0" )      
            theNet.disconnect( GetSubset_1, "output_0", FixUsg_1, "usg_in" )      
            if scale:
                theNet.disconnect( FixUsg_1, "usg_out", Transform_1, "geo_in" )
                theNet.disconnect( Transform_1, "geo_out", RWCovise_2, "mesh_in" )
            else:
                theNet.disconnect( FixUsg_1, "usg_out", RWCovise_2, "mesh_in" )
        else :
            theNet.disconnect( ReadCFX_1, "boundary_mesh", GetSetelem_1, "input_0" )
            theNet.disconnect( GetSetelem_1, "output_0", GetSubset_1, "input_0" )
            theNet.disconnect( GetSetelem_1, "output_0", PipelineCollect_1, "inport_0" )
            theNet.disconnect( GetSubset_1, "output_0",  PipelineCollect_1, "inport_1" )
            theNet.disconnect( PipelineCollect_1, "outport_1", RWCovise_2, "mesh_in" )        
        
    
        theNet.remove( RWCovise_2 )
        theNet.remove( GetSubset_1 )
        theNet.remove( FixUsg_1 )
    
        if not timesteps==None:
            theNet.remove( GetSetelem_1 )
            theNet.remove( PipelineCollect_1 )
            

    #
    # CONVERT SCALAR VARIABLES
    #

    # connect the 3D scalar data
    if scale:
        theNet.connect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
        theNet.connect( ReadCFX_1, "scalar_data", Transform_1, "data_in0" )
        theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
    else:
        theNet.connect( ReadCFX_1, "scalar_data", RWCovise_1, "mesh_in" )


    # loop through the scalar variables
    # select the variable, ommit variable "none"
    varchoice=1
    for svar in scalarVariables:
        varchoice+=1
        # convert only the first numVariables variables
        if ( fixresult=="None" and int(varchoice) < (int(numVariables)+2) ) or \
                svar==fixresult :
        
            #
            # CONVERT SCALAR VARIABLES OF GRID
            #

            logFile.write("\n\tconverting scalar variable %s on grid %s...\n"%(svar,domainname))
            logFile.flush()
            print("\n\tconverting scalar variable", svar , "on grid" , domainname , "...")
            # clean variablename
            if "/" in svar:
                logFile.write("\t! Attention: Replacing the / in svar = %s\n"%(svar,))
                logFile.flush()
                print("\t! Attention: Replacing the / in svar = ", svar)
                svar=svar.replace("/","per")
            if "\x7f" in svar:
                logFile.write("\t! Attention: Replacing a special character in svar = %s\n"%(svar,))
                logFile.flush()
                print("\t! Attention: Replacing a special character in svar = ", svar)
                svar=svar.replace("\x7f","_")
            # select variable
            ReadCFX_1.set_scalar_variables( varchoice )
            if not scale: 
                ReadCFX_1.set_read_grid("false")
            covisename = domainname + "-" + svar + "-3D.covise"
            #create RWCovise name
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logile
            logFile.write("\t... conversion successful! Filename: %s\n"%(covisename,))
            logFile.flush()
            print("\t... conversion successful! Filename: ",covisename)
            # add variable to cocase item
            item3D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
            

            #
            # CONVERT SCALAR VARIABLES OF BOUNDARIES
            #

            if processBoundaries:
                # 
                # RWCovise
                #
                RWCovise_2 = RWCovise()
                theNet.add( RWCovise_2 )
                RWCovise_2.set_stepNo( 0 )
                RWCovise_2.set_rotate_output( "FALSE" )
                RWCovise_2.set_rotation_axis( 3 )
                RWCovise_2.set_rot_speed( 2.000000 )
                #
                # Module GetSubset
                #                 
                GetSubset_1 = GetSubset()
                theNet.add( GetSubset_1 )      

                #
                # Module FixUsg
                #                 
                FixUsg_1 = FixUsg()
                theNet.add( FixUsg_1 )      

                # transient case
                if not timesteps==None:
                    GetSetelem_1 = GetSetelem()
                    theNet.add( GetSetelem_1 )
                    PipelineCollect_1 = PipelineCollect()
                    theNet.add( PipelineCollect_1 )
                
                # connect the boundary scalar data
                if timesteps==None:
                    theNet.connect( ReadCFX_1, "boundary_mesh", GetSubset_1, "input_0" )      
                    theNet.connect( GetSubset_1, "output_0", FixUsg_1, "usg_in" )      
                    theNet.connect( ReadCFX_1, "boundary_scalar_data", GetSubset_1, "input_1" )
                    theNet.connect( GetSubset_1, "output_1", FixUsg_1, "data_00_in" )      

                    if scale:
                        theNet.connect( FixUsg_1, "data_00_out", Transform_1, "data_in0" )
                        theNet.connect( Transform_1, "data_out0", RWCovise_2, "mesh_in" )
                    else:
                        theNet.connect( FixUsg_1, "data_00_out", RWCovise_2, "mesh_in" )
                else :
                    # scale not supported yet
                    theNet.connect( ReadCFX_1, "boundary_scalar_data", GetSetelem_1, "input_0" )
                    theNet.connect( GetSetelem_1, "output_0", GetSubset_1, "input_0" )
                    theNet.connect( GetSetelem_1, "output_0", PipelineCollect_1, "inport_0" )
                    theNet.connect( GetSubset_1, "output_0",  PipelineCollect_1, "inport_1" )
                    theNet.connect( PipelineCollect_1, "outport_1", RWCovise_2, "mesh_in" )    

                subset=0
                boundchoice=1
                for boundname in boundaries:
                    # ommit names "None" "all"
                    boundchoice+=1            
                    if int(boundchoice) > 2 :
                        logFile.write("\n\tconverting scalar variable %s on surface %s...\n"%(svar,boundname))
                        logFile.flush()
                        print("\n\tconverting scalar variable", svar , "on surface" , boundname , "...")
                        bname=boundname
                        # clean boundname
                        if "/" in boundname:
                            logFile.write("\t! Attention: Replacing the / in boundname = %s\n"%(boundname,))
                            logFile.flush()
                            print("\t! Attention: Replacing the / in boundname = ", boundname)
                            bname=boundname.replace("/","per")
                        if "\x7f" in boundname:
                            logFile.write("\t! Attention: Replacing a special character in boundname = %s\n"%(boundname,))
                            logFile.flush()
                            print("\t! Attention: Replacing a special character in boundname = ", boundname)
                            bname=boundname.replace("\x7f","_")
                        # clean variablename
                        if "/" in svar:
                            logFile.write("\t! Attention: Replacing the / in svar = %s\n"%(svar,))
                            logFile.flush()
                            print("\t! Attention: Replacing the / in svar = ", svar)
                            svar=svar.replace("/","per")
                        if "\x7f" in svar:
                            logFile.write("\t! Attention: Replacing a special character in svar = %s\n"%(svar,))
                            logFile.flush()
                            print("\t! Attention: Replacing a special character in svar = ", svar)
                            svar=svar.replace("\x7f","_")
                            print("\t! new svar = ", svar)
                        # set the subset
                        GetSubset_1.set_selection( str(subset) )
                        # create RWCovise name
                        covisename = domainname + "-boundary-" + bname + "-" + svar + "-2D.covise"
                        RWCovise_2.set_grid_path( covisename )
                        # execute                    
                        if timesteps==None:
                            GetSubset_1.execute()
                        else :
                            GetSetelem_1.set_stepNo(1)
                            GetSetelem_1.execute()                    
                        theNet.finishedBarrier()
                        # write logfile
                        logFile.write("\t... conversion successful: Filename %s\n"%(covisename,))
                        logFile.flush()
                        print("\t... conversion successful: ",covisename)
                        # append variable to cocase item
                        if boundname in item2D:
                                item2D[boundname].addVariableAndFilename(svar, covisename, SCALARVARIABLE)
                        subset+=1
                                            
                # disconnect the boundaries
                if timesteps==None:
                    theNet.disconnect( ReadCFX_1, "boundary_mesh", GetSubset_1, "input_0" )      
                    theNet.disconnect( GetSubset_1, "output_0", FixUsg_1, "usg_in" )      
                    theNet.disconnect( ReadCFX_1, "boundary_scalar_data", GetSubset_1, "input_1" )
                    theNet.disconnect( GetSubset_1, "output_1", FixUsg_1, "data_00_in" )      

                    if scale:
                        theNet.disconnect( FixUsg_1, "data_00_out", Transform_1, "data_in0" )
                        theNet.disconnect( Transform_1, "data_out0", RWCovise_2, "mesh_in" )
                    else:
                        theNet.disconnect( FixUsg_1, "data_00_out", RWCovise_2, "mesh_in" )
                else :
                    # scale not supported yet
                    theNet.disconnect( ReadCFX_1, "boundary_scalar_data", GetSetelem_1, "input_0" )
                    theNet.disconnect( GetSetelem_1, "output_0", GetSubset_1, "input_0" )
                    theNet.disconnect( GetSetelem_1, "output_0", PipelineCollect_1, "inport_0" )
                    theNet.disconnect( GetSubset_1, "output_0",  PipelineCollect_1, "inport_1" )
                    theNet.disconnect( PipelineCollect_1, "outport_1", RWCovise_2, "mesh_in" )    
                theNet.remove( RWCovise_2 )
                theNet.remove( GetSubset_1 )            
                theNet.remove( FixUsg_1 )            
                if not timesteps==None:
                    theNet.remove( GetSetelem_1 )
                    theNet.remove( PipelineCollect_1 ) 

            
    # disconnect the grid
    if scale:
        theNet.disconnect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
        theNet.disconnect( ReadCFX_1, "scalar_data", Transform_1, "data_in0" )
        theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
    else:
        theNet.disconnect( ReadCFX_1, "scalar_data", RWCovise_1, "mesh_in" )



    #
    # convert the vector variables
    #

    # read no boundaries
    ReadCFX_1.set_boundaries( 1 )  

    # connect the modules
    if scale:
        theNet.connect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
        theNet.connect( ReadCFX_1, "vector_data", Transform_1, "data_in0" )
        theNet.connect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
    else:
        theNet.connect( ReadCFX_1, "vector_data", RWCovise_1, "mesh_in" )

    # loop though the variables
    # ommit variable "none"
    varchoice=1
    for vvar in vectorVariables:
        varchoice+=1
        # select only the first numVariables variables
        if ( fixresult=="None" and int(varchoice) < (int(numVariables)+2) ) or \
                vvar==fixresult :
            logFile.write("\n\tconverting vector variable %s ...\n"%(vvar,))
            logFile.flush()
            print("\n\tconverting vector variable", vvar , "...")
            # clean variablename
            if "/" in vvar:
                logFile.write("\t! Attention: Replacing the / in vvar = %s\n"%(vvar,))
                logFile.flush()
                print("\t! Attention: Replacing the / in vvar = ", vvar)
                vvar=vvar.replace("/","per")
            if "\x7f" in vvar:
                logFile.write("\t! Attention: Replacing a special character in vvar = %s\n"%(vvar,))
                logFile.flush()
                print("\t! Attention: Replacing a special character in svar = ", vvar)
                vvar=vvar.replace("\x7f","_")
                print("\t! new vvar = ", vvar)
            # select variable
            ReadCFX_1.set_vector_variables( varchoice )
            if not scale:
                ReadCFX_1.set_read_grid("false")
            # create RWCovise name
            covisename = domainname + "-" + vvar + "-3D.covise"
            RWCovise_1.set_grid_path( covisename )
            # execute
            runMap()
            theNet.finishedBarrier()
            # write logfile
            logFile.write("\t... conversion successful! Filename: %s\n"%(covisename,))
            logFile.flush()
            print("\t... conversion successful! Filename: ",covisename)
            # append variable to cocase item
            item3D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
    # disconnect the modules
    if scale:
        theNet.disconnect( ReadCFX_1, "mesh", Transform_1, "geo_in" )
        theNet.disconnect( ReadCFX_1, "vector_data", Transform_1, "data_in0" )
        theNet.disconnect( Transform_1, "data_out0", RWCovise_1, "mesh_in" )
    else:
        theNet.disconnect( ReadCFX_1, "vector_data", RWCovise_1, "mesh_in" )




    # add the domain item to the case file
    if processGrid:
        calculatePDYN(item3D)
        cocase.add(item3D)

    # add the boundary item do the case file
    if processBoundaries:
        boundchoice=1
        for boundname in boundaries:
            # ommit names "None" "all"
            boundchoice+=1            
            if int(boundchoice) > 2 :
                calculatePDYN(item2D[boundname])
                cocase.add(item2D[boundname])









############################ START ########################

print("list of all domains=", domains)

print("Fixdomain", fixdomain)

#
# loop through the domains parts
# ('all', 'Domain\x7f1', '')
#
domainchoice=1
for domainname in domains:
    domainchoice+=1
    if fixdomain=="None" or fixdomain==domainname:
        convert(domainname, domainchoice, (noGrid=="0"), (readBoundaries=="1"))

#
# create composed grid
#
if (noGrid=="0") and (composedGrid=="1"):
    convert("all", 1, True, False)



#
# remove the CFX module
#
theNet.remove( ReadCFX_1 )

# write logfile
logFile.write("Conversion finished\n")
logFile.flush()
print("Conversion finished")
