# PROJECT(bison++)

ADD_DEFINITIONS(
  -DPACKAGE_NAME=\"FULL-PACKAGE-NAME\"
  -DPACKAGE_TARNAME=\"full-package-name\"
  -DPACKAGE_VERSION=\"VERSION\"
  -DPACKAGE_STRING=\"FULL-PACKAGE-NAME_VERSION\"
  -DDEFPACKAGE_BUGREPORT=\"BUG-REPORT-ADDRESS\"
  -DPACKAGE=\"bison++\"
  -DVERSION=\"2.21.5\"
  -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1
  -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1
  -DHAVE_ALLOCA_H=1 -DHAVE_MALLOC_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STDDEF_H=1 -DHAVE_STDLIB_H=1
  -DHAVE_STRING_H=1 -DHAVE_STRINGS_H=1 -DHAVE_ALLOCA_H=1 -DHAVE_ALLOCA=1 -DHAVE_STDLIB_H=1
  -DHAVE_MALLOC=1
  -DXPFILE="${COVISEDIR}/src/tools/bison++/bison.cc"
  -DXHFILE="${COVISEDIR}/src/tools/bison++/bison.h"
  -DXPFILE1="${COVISEDIR}/src/tools/bison++/bison.hairy"
)

INCLUDE_DIRECTORIES(
)

SET(HEADERS
)

SET(SOURCES
  alloca.c
  conflict.cc
  getopt1.cc
  lex.cc
  old.c
  reduce.cc
  warshall.cc
  allocate.cc
  derives.cc
  getopt.cc
  lr0.cc
  output.cc
  symtab.cc
  files.cc
  gram.cc
  main.cc
  print.cc
  version.cc
  closure.cc
  getargs.cc
  lalr.cc
  nullable.cc
  reader.cc
)

SET(EXTRASOURCES
)

ADD_COVISE_EXECUTABLE(bison++ )
COVISE_WNOERROR(bison++)
COVISE_NOWARN(bison++)
TARGET_LINK_LIBRARIES(bison++  )

COVISE_INSTALL_TARGET(bison++)
