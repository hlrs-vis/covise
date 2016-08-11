#!/usr/bin/python
# -*- coding: cp1252 -*-
import sys
import subprocess
#import string
from string import *


class makePyMod :
    
    def __init__( self, covModOutput ):
        self.fields_ = covModOutput.split("\n")
##        self.name_   = self.fields_[0].split()[1]
##        ln    = len(self.name_)-1
##        self.name_   = self.name_.strip()[1:ln]

        self.outPortPos_ = 0
        self.outPortNum_ = 0
        self.inPortPos_  = 0
        self.inPortNum_  = 0
        self.paramPos_   = 0
        self.paramNum_   = 0
        self.name_       = None
        
        #
        # extract port indices
        #        
        cnt = 0
        errorCnt = 0
        for i in self.fields_:
            #print(i)
            bknowntag = False
            x = i.split()
            try:
                if ( x[0] == "Module:"):
                    self.name_  = str(x[1]).strip('"')
                    bknowntag = True
                if ((self.name_ != None) and (self.name_.endswith(".exe"))):
                    self.name_ = self.name_[0:len(self.name_)-4]
                    bknowntag = True
                if (self.name_ != None):
                   self.name_ = self.name_.replace('-','_')
                   self.name_ = self.name_.replace('.','_')
                if ( x[0] == "InPorts:"):
                    self.inPortPos_ = cnt + 1
                    self.inPortNum_ = int(x[1])
                    bknowntag = True
                if ( x[0] == "OutPorts:"):
                    self.outPortPos_ = cnt + 1
                    self.outPortNum_ = int(x[1])
                    bknowntag = True
                if ( x[0] == "Parameters:"):
                    self.paramPos_ = cnt + 1
                    self.paramNum_ = int(x[1])
                    bknowntag = True
                # for debugging:
                #if (not bknowntag):
                #    print("# ", str(x[0])
            except IndexError:
                    errorCnt = errorCnt + 1
                    print("# IndexError for", self.name_)
            except ValueError:
                    errorCnt = errorCnt + 1
                    print("# ValueError for", self.name_)
            cnt = cnt + 1


    def parseSimple( self, inp, fillchar='_' ):         
        pIt = inp.split()
        pList=[]
        entry = ""
        bopen=0
        for ii in pIt:
            ll = len(ii)-1
            if (ii[0]=='\"') and (ii[ll]=='\"'):
                if ( ll > 0):
                    entry = ii[1:ll]
                    pList.append( entry )
                    entry=""
                    bopen=0
                else:
                    if (bopen==1):
                        pList.append( entry )
                        entry=""
                        bopen=0
                    else:
                        bopen = 1                    
            elif (ii[0]=='\"') and (ii[ll]!='\"'):
                entry = ii[1:]
                bopen=1
            elif (ii[0]!='\"') and (ii[ll]!='\"'):
                entry = entry + fillchar + ii
            elif (ii[0]!='\"') and (ii[ll]=='\"'):
                entry = entry + fillchar + ii[:ll]
                pList.append( entry )
                entry=""
                bopen=0
##            elif (ii  =='\"'):                
##                if (bopen==1):
##                    pList.append( entry )
##                    entry=""
##                    bopen=0
##                else:
##                    bopen = 1

        #print("PLIST ",pList)
        return pList

        

    def header( self ):   
        print("class %s(CoviseModule):"%(self.name_))
        print("#==================================")
        print("    name_ = \"%s\""%(self.name_))
        print()
        print("    def __init__(self):")
        print("        self.ports = []")
        print("        self.params_ = []")
        print("        self.host_=globalHostInfo.getName()")
        print("        self.choiceDict_ =", self.choiceDict)
        print()

    def addAllInPorts( self ):
        #
        # extract inPorts
        #
        print("        # IN ports")
        for i in self.fields_[self.inPortPos_:self.inPortPos_ + self.inPortNum_]:
            pList = self.parseSimple(i,'')
            print("        self.addPort( \""+pList[0]+"\" ,\""+pList[1]+"\", \"IN\")")


    def addAllOutPorts( self ):
        #
        # extract outPorts
        #
        print()
        print("        # OUT ports")
        
        for i in self.fields_[self.outPortPos_:self.outPortPos_+self.outPortNum_]:
            pList = self.parseSimple(i,'')
            print("        self.addPort( \""+pList[0]+"\" ,\""+pList[1]+"\", \"OUT\")")


    def addAllParameters( self ):
        #
        # extract parameters
        #
        argListDict = { 
                        "IntScalar"   : "( self, x )",
                        "FloatScalar" : "( self, x )",
                        "IntVector"   : "( self, x, y, z )",
                        "FloatVector" : "( self, x, y, z )",
                        "Color"       : "( self, r, g, b, a )",
                        "Choice"      : "( self, x )",
                        "Choice"      : "( self, x )",
                        "choice"      : "( self, x )",
                        "Browser"     : "( self, x )",                        
                        "BrowserFilter"  : "( self, x )",                        
                        "String"      : "( self, x )",
                        "Boolean"     : "( self, x )",
                        "Colormap"    : "( self, x )",
                        "ColormapChoice"    : "( self, x )",
                        "Material"    : "( self, x )",
                        "Timer"       : "( self, x )",                        
                        "IntSlider"   : "( self, x, y, z )",
                        "FloatSlider" : "( self, x, y, z )" }
           
           
        valstrDict = {
                       "IntScalar"   : " \"%d\" % x ",
                       "FloatScalar" : " \"%f\" % x ",
                       "IntVector"   : " \"%d\" % x + \" %d\" % y + \" %d\" % z " ,
                       "FloatVector" : " \"%f\" % x + \" %f\" % y + \" %f\" % z " ,
                       "Color"       : " \"%f\" % r + \" %f\" % g + \" %f\" % b + \" %f\" % a ",
                       "Choice"      : "x ",
                       "Browser"     : "x ",
                       "BrowserFilter"  : " x ",
                       "String"      : "x ",
                       "Boolean"     : "x ",
                       "Colormap"    : "x ",
                       "ColormapChoice"    : "x ",
                       "Material"    : "x ",
                       "Timer"       : "x ",
                       "IntSlider"   : " \"%d\" % x + \" %d\" % y + \" %d\" % z " ,
                       "FloatSlider" : " \"%f\" % x + \" %f\" % y + \" %f\" % z " }


        allParamItems = []
        allRawParamNames = []
        
        # extract parameter name (if it contains " " these will be replaced by "_"
        print()
        print("        # parameter handling")
        print("        #     1. register paramters to covise module stub")

        for i in self.fields_[self.paramPos_:self.paramPos_+self.paramNum_]:

            pIt = i.split()

            pList = self.parseSimple(i)
            rawL  = self.parseSimple(i,' ')
            
            parameterName = pList[0]

            parameterType = pList[1]
            parameterName = parameterName.replace(':','')
            parameterName = parameterName.replace('/','')

            oPname = rawL[0]

            try:
                parameterMode = pList[4]
            except IndexError:
                sys.stderr.write("index out of range for module " + self.name_ + ", parameter name " + parameterName + " -- exiting\n")
                sys.exit(1)
            print("        self.addParam( \"" + oPname + "\", \"" +parameterType+"\", \"" +parameterMode+"\")")
            allParamItems.append( pList )
            allRawParamNames.append( oPname )
            
        print() 
        print("    #     2. create parameter set members")

        for ii in allParamItems:            
            ky = ii[1]

            if ( ky == "FloatVector"):
                vals = ii[2].split('_')
                if (len(vals) == 3):
                    argList = "( self, x, y, z )"
                else:
                    argList = "( self, x, y )"
            else:
                try:
                    argList = argListDict[ ky ]
                except KeyError:
                    sys.stderr.write(" key '" + ky + "' not handled in module " + self.name_ + "-- exiting\n")
                    sys.exit(1)
            idx = allParamItems.index( ii )
            oPname = allRawParamNames[idx] 
            #function names must not contain the following characters +-()/:*.
            pname = ii[0]
            pname = pname.replace(':','')
            pname = pname.replace('/','')
            pname = pname.replace('(','')
            pname = pname.replace(')','')
            pname = pname.replace('-','')
            pname = pname.replace('*','')
            pname = pname.replace('.','')
            
            paramMemberName= "set_" + pname + argList
            print("    def", paramMemberName,":")
            
            if (ky == "FloatVector"):
                vals = ii[2].split('_')
                if (len(vals) == 3):
                    print("        valstr = "+valstrDict[ky])
                else:
                    print("        valstr = \"%f\" % x + \" %f\" % y")
            else:
                print("        valstr = "+valstrDict[ky])
            print("        self.setParamValue( \"" + oPname + "\", valstr )")
            paramMemberName= "get_" + pname + "( self ) :"
            print("    def", paramMemberName)
            print("        return self.getParamValue( \"" + oPname + "\" )")
            print()
            
            
    #
    # creates choice dictionary which is used to create a
    # convenience function to set choice values
    #
    def makeChoiceDict( self ):
    #==========================    
        self.choiceDict = {}
        try:
           for i in self.fields_[self.paramPos_:self.paramPos_+self.paramNum_]:
              pl = self.parseSimple( i, ' ' )
              if ( pl[1] == "Choice" ) or ( pl[1] == "Choice" ):
                 valList = pl[2].split()
                 subDict = {}
                 val = 1
                 for x in  valList[1:]:
                    ky = x.replace('\x7f',' ')
                    subDict[ ky ] = val
                    val=val+1
                 self.choiceDict[ pl[0] ] = subDict
        except IndexError:
           print("#IndexError while creating choiceDict for", self.paramNum_)


                
    def create( self ):
        if self.name_:
            self.makeChoiceDict()
            self.header()
            self.addAllInPorts()
            self.addAllOutPorts()
            self.addAllParameters()
            if ( self.name_ == "OpenCOVER" ):
                self.name_ = "VRRenderer"
                self.header()
                self.addAllInPorts()
                self.addAllOutPorts()
                self.addAllParameters()
            if ( self.name_ == "QtRenderer" ):
                self.name_ = "Renderer"
                self.header()
                self.addAllInPorts()
                self.addAllOutPorts()
                self.addAllParameters()
            return 1
        else:
            return -1




import os, sys
from stat import *

class Conversion:
    fcnt_ = 0
    scnt_ = 0
    
    def convertFromFile(self, file, repFile):
        print('#')
        print('# PYTHON module stub made from: ')
        print('#             ', file)
        print('#')
    
        cmd = file +" -d"

        #r, w, e = popen2.popen3(cmd)
        #got = r.read()
        #errout = e.read()
        p=subprocess.Popen(
        cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE 
        )
                #, close_fds=True
        (w, r, e) = (p.stdin, p.stdout, p.stderr)
        got=r.read()
        errout=e.read()
        repFile.write( errout)
        
        #try:
        pyMod = makePyMod( got.decode('UTF-8') )
        #except e:
        #    self.fcnt_ = self.fcnt_ + 1
        #    print("# ERROR for ", file)
        #    #print("# ", e.message
        #    return

        stat = pyMod.create()

        if (stat==-1):
            oStr="--> could not generate Python module stub for " + file
            repFile.write( bytes('\n'+oStr+'\n\n','UTF-8'))
            self.fcnt_ = self.fcnt_ + 1
        else:
            self.scnt_ = self.scnt_ + 1
            

    def convertGroups( self, dir, repFile, ignorefilelist):
        for f in os.listdir(dir):
            pathname = '%s/%s' % (dir, f)
            mode = os.lstat(pathname)[ST_MODE]
            if S_ISDIR(mode):
                # It's a directory therefore it must be a COVISE group
                for ff in os.listdir(pathname):
                    execName = '%s/%s' % (pathname, ff)
                    if not execName.endswith(".pdb") and not execName.endswith(".ilk") and not execName.endswith(".exp") and not execName.endswith(".lib") and not execName.endswith(".suo"):
                        skipfile = False
                        # check to see, if filename is to be ignored
                        if (ignorefilelist != None):
                            for fignore in ignorefilelist:
                                actualfile = '%s/%s' % (f, ff)
                                #repFile.write("convertGroups - comparing ignorefile=%s w/ actualfile=%s\n" % (fignore, actualfile))
                                if (fignore == actualfile):
                                    #repFile.write("convertGroups - will ignore file=%s\n" % fignore)
                                    skipfile = True
                        if not skipfile:
                            repFile.write(bytes("convertGroups - processing file=",'UTF-8'))
                            repFile.write(bytes(execName,'UTF-8'))
                            repFile.write(bytes("\n",'UTF-8'))
                            mmode = os.lstat(execName)[ST_MODE]
                            if S_ISREG(mmode) and (ff[0] != "."):
                                # It's a file, call the callback function
                                self.convertFromFile( execName, repFile)
                        else:
                            repFile.write("convertGroups - ignoring file=%s\n" % execName)


    def getScnt(self):
        return self.scnt_


    def getFcnt(self):
        return self.fcnt_


    def writeFileHeader(self, sname):
        from time import gmtime, strftime
        tstr = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        print("##########################################################################################")
        print("#")
        print("# Python-Module: coviseModules.py")
        print("#")
        print("# Description: python module wrapping all COVISE modules to python (meta) modules")
        print("#")
        print("#              THIS SCRIPT WAS GENERATED BY", sname)
        print("#              DATE of GENERATION", tstr)
        print("#")
        print("# Usage: python coviseModules.py ")
        print("#           [<name of single file to be processed> | ")
        print("#            {-i<name of single file not to be processed>}*]")
        print("#        If no arguments are given, all available COVISE modules get processed.")
        print("#")
        print("# (C) Copyright 2005-2009 Visual Engineering Solutions GmbH, Stuttgart   info@visenso.de")
        print("#")
        print("##########################################################################################")
        print("from coviseModuleBase import *")



                
#
# main
#
if __name__ == '__main__':

    # create report file
    from time import gmtime, strftime
    reportname = strftime("covise_module_stub_gen_report-%d%b%Y-%H%M%S", gmtime())
    reportfile = open(reportname,'wb+')

    
    #
    # get $COVISEDIR
    #
    covisedir = os.environ["COVISEDIR"]
    archsuffix = os.environ["ARCHSUFFIX"]
    covisedir = covisedir+'/'+archsuffix+"/bin"

    argv = sys.argv
    argc = len(sys.argv)

    c=Conversion()

    if ( argc >= 1):
        c.writeFileHeader(argv[0])
        
    if ( argc == 1):
        # no arguments; the first argument is the script's path + name
        c.convertGroups(covisedir, reportfile, None)
    elif( argc >= 2):
        if ( ( len(argv[1]) > 2 ) and ( argv[1][0] == "-" ) and ( argv[1][1] == "i" ) ):
            # expect list of filenames to ignore, when searching COVISE modules
            ignorefilelist = []
            for _run1 in range (0, argc-1):
                ignorefilelist.append(argv[_run1 + 1].lstrip('-i'))
                reportfile.write("main - adding to ignorelist file=%s\n" % argv[_run1 + 1].lstrip('-i'))
            c.convertGroups(covisedir, reportfile, ignorefilelist)
        else:
            # consider argument a filename to be processed (possible other args get ignored)
            c.convertFromFile(argv[1], reportfile)

    reportfile.write(bytes("REPORT:\n",'UTF-8'))
    reportfile.write(bytes(" >>>  %d  python module representations SUCESSFULLY created\n" % c.getScnt(),'UTF-8'))
    reportfile.write(bytes(" >>>  %d  modules did not respond as expected - these are not available in the covise scripting-interface\n" % c.getFcnt() ,'UTF-8'))

    reportfile.close()
