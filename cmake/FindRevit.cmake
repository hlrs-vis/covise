# - Try to find revit-library
# Once done this will define
#
#  REVIT_DIRECTORY    - where to find Revit
#  REVIT_LIBRARIES      - list of libraries when using revit
#  REVIT_FOUND          - True if revit was found.



FIND_PATH(REVIT_DIRECTORY NAMES AdWindows.dll RevitAPI.dll
  PATHS
  $ENV{ProgramFiles}/Autodesk/"Revit 2024"
  $ENV{ProgramFiles}/Autodesk/"Revit 2023"
  $ENV{ProgramFiles}/Autodesk/"Revit 2022"
  $ENV{ProgramFiles}/Autodesk/"Revit 2021"
  $ENV{ProgramFiles}/Autodesk/"Revit 2020"
  $ENV{ProgramFiles}/Autodesk/"Revit 2019"
  "c:/Program Files/Autodesk/Revit 2020"
  NO_DEFAULT_PATH
)
IF (REVIT_DIRECTORY)
  SET(REVIT_LIBRARIES ${REVIT_DIRECTORY}/AdWindows.dll ${REVIT_DIRECTORY}/RevitAPI.dll ${REVIT_DIRECTORY}/RevitAPIUI.dll)
ELSE (REVIT_DIRECTORY)
  SET(REVIT_LIBRARIES NOTFOUND)
ENDIF (REVIT_DIRECTORY)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Revit DEFAULT_MSG REVIT_DIRECTORY)
MARK_AS_ADVANCED(REVIT_DIRECTORY)
