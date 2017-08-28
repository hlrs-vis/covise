include_directories(..)

SET(HEADERS
ChargedObject.h
ChargedObjectHandler.h
ChargedPlate.h
ChargedPoint.h
ElectricFieldPlugin.h
Probe.h
Tracer.h
)

SET(SOURCES
ChargedObject.cpp
ChargedObjectHandler.cpp
ChargedPlate.cpp
ChargedPoint.cpp
ElectricFieldPlugin.cpp
Probe.cpp
Tracer.cpp
)
cover_add_plugin(ElectricField CovisePluginUtil ${COVISE_SHM_LIBRARY} ${COVISE_ALG_LIBRARY} ${COVISE_CORE_LIBRARY} ${COVISE_APPL_LIBRARY} ${COVISE_GRMSG_LIBRARY})
