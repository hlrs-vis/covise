####################################################################################
#  
# Script: scriptInterface.py
#
# Description: This is the UIF runned by the covise controller. In essence it does
#              two things:
#
#              1. Initializes the connection to covise.
#              2. Runs the python startup for covise.
#
#
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# (C) 2003 by VirCinity IT Consulting GmbH                                          
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# initial version 07.03.2003 (RM)
#
####################################################################################
import sys
import covise

#
# extract args
#
argv = sys.argv
argc = len(sys.argv)
#print("COVISE PYTHON INTERFACE1")
#print(argc)
#print(type(argv[0]))
#print(argv[1])
#print(argv[2])
#print(argv[3])
#print(argv[4])
#print(argv[5])
#print(argv[6])
#print(argv[7])
#print(argv[8])
#print(argv[9])
#print(argv[10])
#print(argv[11])
#
# remove first arg of the userinterface script
# it is the filename to be executed (if 8 args are given)
#
global execFileFlag
execFileFlag=-1
global execFile
execFile=''

if (argc == 13):
    argc = argc-1
    execFile =argv[1]
    argv.remove(execFile) 
    if ( len(execFile)>0 ): execFileFlag=1

#    argc = argc+1

#
# run the init function
#
print("*  ******* COVISE PYTHON INTERFACE ********                   *")
covise.run_xuif(argc, argv)


#
# do the internal startup
# this function is essential for a proper working scripting interface
# !!! you better read the comments inside coviseStartup.py
#
from coviseStartup import *

globalHostInfo.setName( argv[5] )


coviseStartupFunc()

print(covise.FileToLoad)
if(covise.FileToLoad != ''):
    with open(covise.FileToLoad) as f:
        code = compile(f.read(), covise.FileToLoad, 'exec')
        exec(code)

#
# we want to have a system specific prompt
#
sys.ps1="covise> "

#
# last but not least run a script if given on the covise
# command-line i.e. covise --script do_funny_things.py
#

if ( execFileFlag > 0 ):
    #execfile( execFile )
        with open(execFile) as f:
                code = compile(f.read(), execFile, 'exec')
                exec(code)
