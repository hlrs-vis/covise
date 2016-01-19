# @file
# 
# Simply descend to subdirectories
#
# @author Blasius Czink
IF(NOT WIN32)
ADD_SUBDIRECTORY(rTCPTunnel)
ENDIF()
ADD_SUBDIRECTORY(vrcw)
ADD_SUBDIRECTORY(comote)
ADD_SUBDIRECTORY(CoolEmAllCasePreparation)
ADD_SUBDIRECTORY(CoolEmAllWriteToDatabase)
ADD_SUBDIRECTORY(samsungctrl)
ADD_SUBDIRECTORY(map_converter)
ADD_SUBDIRECTORY(giswalk)
#ADD_SUBDIRECTORY(zSpaceTest)

IF(NOT UNIX)
  RETURN()
ENDIF()

ADD_SUBDIRECTORY(clean)
ADD_SUBDIRECTORY(bison++)
#ADD_SUBDIRECTORY(catcov)
#ADD_SUBDIRECTORY(erg2cov)
