INCLUDE_DIRECTORIES(
../lib
)
ADD_DEFINITIONS("-DAXIAL_RUNNER -DDEBUG -DDEBUG_ALL -DCOVISE_MODULE -DGRID4COV -DRROUTLET_NOGAP -DPARA_IN -DGRID_IN_DEFAULT -DPARA_OUT -DRR_IONODES -DBLNODES_OUT -DFENFLOSS_OUT -DPATRAN_SES_OUT -DBL_REF -DVAR_VECTOR -DMODIFY_LINE14 -DGRID4COV -DREADJUST_PERIODIC")

SET(HEADERS
)
SET(SOURCES
  AxialRunner.cpp
  CheckUserInput.cpp
  CreateGeo.cpp
  CreateGrid.cpp
  CtrlPanel.cpp
  Menues.cpp
  ModLib.cpp
  Plot.cpp
  vatew.cpp
  ../lib/RunnerGridGen/4all/rr_mesh.cpp
  ../lib/RunnerGridGen/4all/rr_meshinit.cpp
  ../lib/RunnerGridGen/4all/rr_meshwrite.cpp
  ../lib/General/4all/cg4c.cpp 
  ../lib/General/4all/cg.cpp
  ../lib/General/4all/wg.cpp
  ../lib/RunnerGridGen/rr_meshnod.cpp
  ../lib/RunnerGridGen/new_meshnod.cpp
)
covise_add_module(VISiT AxialRunner_dev ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(AxialRunner_dev coApi coAppl coCore visit_axialrunner visit_bspline visit_runnergridgen visit_general)
