INCLUDE_DIRECTORIES(
../lib
)
ADD_DEFINITIONS("-DRADIAL_RUNNER -DGRID4COV -DLIN_WRAPOPT -DCAMB_POSITION -DCOVISE_MODULE -DRROUTLET_NOGAP -DPARA_IN -DGRID_IN_DEFAULT -DPARA_OUT -DRR_IONODES -DBLNODES_OUT -DFENFLOSS_OUT -DPATRAN_SES_OUT -DBL_REF -DVAR_VECTOR -DMODIFY_LINE14 -DCONTOUR_WIREFRAME -DPROGGEO -DREADJUST_PERIODIC")

SET(HEADERS
  RadialRunner_Gap.h
  RadialRunner.h
)
SET(SOURCES
  CheckUserInput.cpp
  CreateGrid.cpp
  CtrlPanel.cpp
  Menues.cpp
  ModLib.cpp
  Plot.cpp
  RadialRunner.cpp
  ../lib/RadialRunner/rr_comp.cpp
  ../lib/RunnerGridGen/4all/rr_mesh.cpp
  ../lib/RunnerGridGen/4all/rr_meshinit.cpp
  ../lib/RunnerGridGen/4all/rr_meshwrite.cpp 
  ../lib/RunnerGridGen/rr_meshnod.cpp 
  ../lib/RunnerGridGen/new_meshnod.cpp 
  ../lib/General/4all/cg4c.cpp 
  ../lib/General/4all/cg.cpp
  ../lib/General/4all/wg.cpp
)

covise_add_module(VISiT RadialRunner_dev ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(RadialRunner_dev coApi coAppl coCore visit_radialrunner visit_runnergridgen visit_general visit_bspline)
