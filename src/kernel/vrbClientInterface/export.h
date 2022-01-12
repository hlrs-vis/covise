#ifndef VRB_CLIENT_INTERFACE_EXPORT_H
#define VRB_CLIENT_INTERFACE_EXPORT_H

#if defined(_WIN32) && !defined(NODLL)
#define IMPORT __declspec(dllimport)
#define EXPORT __declspec(dllexport)

#elif defined(__GNUC__) && __GNUC__ >= 4
#define EXPORT __attribute__((visibility("default")))
#define IMPORT V_EXPORT
#else
#define IMPORT
#define EXPORT
#endif


#if defined(coVRBClientInterface_EXPORTS)
#define VRBClientInterfaceEXPORT EXPORT
#else
#define VRBClientInterfaceEXPORT IMPORT
#endif

#endif //VRB_CLIENT_INTERFACE_EXPORT_H