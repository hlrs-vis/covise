@ECHO OFF

REM ###############################################################
REM
REM purpose: classes that cannot yet be converted automatically get
REM    "patched" into the python source 'coPyModules.py'
REM 
REM call:
REM    patch_coPyModules.bat >> coPyModules.py
REM
REM (C) Copyright 2009 Visual Engineerign Solutions GmbH
REM 
REM ###############################################################

SETLOCAL

REM mask the percentage symbol
REM note: apparently the demasking character ^ does not work on the percentage symbol
SET /P _PRCNT_=<%~dp0\patch_coPyModules.txt

ECHO # PYTHON module stub made by
ECHO # "/work/sk_te/Covise6.0/covise/gcc3/bin/Tools/IsoCutter -d".
ECHO class IsoCutter(CoviseModule):
ECHO     """Wrapper for COVISE-module "IsoCutter"."""

ECHO     name_ = "IsoCutter"

ECHO     def __init__(self):
ECHO         CoviseModule.__init__(self)
ECHO         self.choiceDict_ = {}

ECHO         # IN ports
ECHO         self.addPort("inPolygons" ,"Polygons", "IN")
ECHO         self.addPort("inData" ,"Unstructured_S3D_Data", "IN")

ECHO         # OUT ports
ECHO         self.addPort("outPolygons" ,"Polygons", "OUT")
ECHO         self.addPort("outData" ,"Unstructured_S3D_Data", "OUT")

ECHO         # Parameters
ECHO         self.addParam("iso_value", "Slider", "IMM")
ECHO         self.addParam("auto_minmax", "Boolean", "IMM")
ECHO         self.addParam("cutoff_side", "Boolean", "IMM")

ECHO     # Parameter set member functions

ECHO     def set_iso_value(self, x, y, z):
ECHO         valstr = "3\n" + "%_PRCNT_%.10f\n" %_PRCNT_% x + "%_PRCNT_%.10f\n" %_PRCNT_% y + "%_PRCNT_%.10f\n" %_PRCNT_% z + "\n"
ECHO         self.setParamValue('iso_value', valstr)

ECHO     def set_auto_minmax(self, x):
ECHO         valstr = "1\n" + x + "\n"
ECHO         self.setParamValue('auto_minmax', valstr)

ECHO     def set_cutoff_side(self, x):
ECHO         valstr = "1\n" + x + "\n"
ECHO         self.setParamValue('cutoff_side', valstr)
        
ECHO # PYTHON module stub made by
ECHO # "/work/sk_te/Covise6.0/covise/gcc3/bin/Unsupported/MagmaTrace -d".
ECHO class MagmaTrace(CoviseModule):
ECHO     """Wrapper for COVISE-module "MagmaTrace"."""

ECHO     name_ = "MagmaTrace"

ECHO     def __init__(self):
ECHO         CoviseModule.__init__(self)
ECHO         self.choiceDict_ = {}

ECHO         # IN ports
ECHO         self.addPort("geo_in" ,"Points", "IN")
ECHO         self.addPort("data_in" ,"Unstructured_S3D_Data", "IN")

ECHO         # OUT ports
ECHO         self.addPort("geo_out" ,"Lines", "OUT")
ECHO         self.addPort("data_out" ,"Unstructured_S3D_Data", "OUT")

ECHO         # Parameters
ECHO         self.addParam("len", "Scalar", "START")
ECHO         self.addParam("skip", "Scalar", "START")

ECHO     # Parameter set member functions

ECHO     def set_len(self, x):
ECHO         valstr = "1\n" + "%_PRCNT_%.10f\n" %_PRCNT_% x + "\n"
ECHO         self.setParamValue('len', valstr)

ECHO     def set_skip(self, x):
ECHO         valstr = "1\n" + "%_PRCNT_%.10f\n" %_PRCNT_% x + "\n"
ECHO         self.setParamValue('skip', valstr)

ENDLOCAL
EXIT /B 0