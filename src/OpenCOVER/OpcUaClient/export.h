#ifndef COVER_OPCUA_EXPORT_H
#define COVER_OPCUA_EXPORT_H

#include <util/coExport.h>

#if defined(coOpenOpcUaClient_EXPORTS)
#define OPCUACLIENTEXPORT COEXPORT
#else
#define OPCUACLIENTEXPORT COIMPORT
#endif

#endif // COVER_OPCUA_EXPORT_H