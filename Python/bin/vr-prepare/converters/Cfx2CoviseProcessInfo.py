from ChoiceGetterAction import ChoiceGetterAction
from IntGetterAction import IntGetterAction
print("Covise Case Name = ", cocasename)
print(" ")
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

#
# loop through the domains parts
# ('all', 'Domain\x7f1', '')
#
print("list of all domains=", domains)
print("Fixdomain", fixdomain)
# choice param is "All", "domain_1",....
# ommit domain "all", because we don't have it any more in the partlist
domainchoice=1
for domainname in domains:
    domainchoice+=1
    if fixdomain=="None" or fixdomain==domainname:
        # write logfile
        print("\ninfo %s"%(domainname))
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
        ReadCFX_1.set_boundaries( 2 )
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

#
# remove the CFX module
#
theNet.remove( ReadCFX_1 )
