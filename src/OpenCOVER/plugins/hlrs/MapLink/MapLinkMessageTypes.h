#ifndef _MAPLINK_MESSAGE_TYPES_H
#define _MAPLINK_ MESSAGE_TYPES_H
enum MessageTypes
    {
        MSG_GetHeight = 500, // this must be the first Message
        MSG_GetMap = 501,
        MSG_SetModules = 502,
        MSG_DeleteModules = 503,
        MSG_ClearAllModules = 504
    };
#endif
