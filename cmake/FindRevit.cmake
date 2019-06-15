# - Try to find revit-library
# Once done this will define
#
#  REVIT_DIRECTORY    - where to find Revit
#  REVIT_LIBRARIES      - list of libraries when using revit
#  REVIT_FOUND          - True if revit was found.



FIND_LIBRARY(REVIT_DIRECTORY NAMES "Revit 2024" "Revit 2023" "Revit 2022" "Revit 2021" "Revit 2020" "Revit 2019" "Revit 2018"
  PATHS
  $ENV{ProgramFiles}/Autodesk
  NO_DEFAULT_PATH
)
IF (REVIT_DIRECTORY)
  SET(REVIT_LIBRARIES ${REVIT_DIRECTORY}/AdWindows.dll ${REVIT_DIRECTORY}/RevitAPI.dll${REVIT_DIRECTORY}/RevitAPIUI.dll)
ELSE (REVIT_DIRECTORY)
  SET(REVIT_LIBRARIES NOTFOUND)
ENDIF (REVIT_DIRECTORY)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Revit DEFAULT_MSG REVIT_DIRECTORY)
MARK_AS_ADVANCED(REVIT_DIRECTORY)
