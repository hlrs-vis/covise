/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MESSAGE_TYPES_H
#define MESSAGE_TYPES_H

#include <util/coExport.h>

/*
 $Log:  $
 * Revision 1.4  1994/03/23  18:07:03  zrf30125
 * Modifications for multiple Shared Memory segments have been finished
 * (not yet for Cray)
 *
 * Revision 1.3  93/10/11  09:22:19  zrhk0125
 * new types DM_CONTACT_DM and APP_CONTACT_DM included
 *
 * Revision 1.2  93/10/08  19:18:06  zrhk0125
 * data type sizes introduced
* some fixed type sizes with sizeof calls replaced
 *
 * Revision 1.1  93/09/25  20:47:03  zrhk0125
 * Initial revision
 *
 */

/***********************************************************************\
 **                                                                     **
 **   Message classes                              Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Message, ShmMessage                                **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                        HOSTID                                             **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  15.04.93  Ver 1.1 new Messages and type added      **
 **                                    sender and send_type added       **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

// IDs for all messages that go between processes are fixed here

enum covise_msg_type
{
    COVISE_MESSAGE_EMPTY = -1, // -1
    COVISE_MESSAGE_MSG_FAILED, //  0
    COVISE_MESSAGE_MSG_OK, //  1
    COVISE_MESSAGE_INIT, //  2
    COVISE_MESSAGE_FINISHED, //  3
    COVISE_MESSAGE_SEND, //  4
    COVISE_MESSAGE_ALLOC, //  5
    COVISE_MESSAGE_UI, //  6
    COVISE_MESSAGE_APP_CONTACT_DM, //  7
    COVISE_MESSAGE_DM_CONTACT_DM, //  8
    COVISE_MESSAGE_SHM_MALLOC, //  9
    COVISE_MESSAGE_SHM_MALLOC_LIST, // 10
    COVISE_MESSAGE_MALLOC_OK, // 11
    COVISE_MESSAGE_MALLOC_LIST_OK, // 12
    COVISE_MESSAGE_MALLOC_FAILED, // 13
    COVISE_MESSAGE_PREPARE_CONTACT, // 14
    COVISE_MESSAGE_PREPARE_CONTACT_DM, // 15
    COVISE_MESSAGE_PORT, // 16
    COVISE_MESSAGE_GET_SHM_KEY, // 17
    COVISE_MESSAGE_NEW_OBJECT, // 18
    COVISE_MESSAGE_GET_OBJECT, // 19
    COVISE_MESSAGE_REGISTER_TYPE, // 20
    COVISE_MESSAGE_NEW_SDS, // 21
    COVISE_MESSAGE_SEND_ID, // 22
    COVISE_MESSAGE_ASK_FOR_OBJECT, // 23
    COVISE_MESSAGE_OBJECT_FOUND, // 24
    COVISE_MESSAGE_OBJECT_NOT_FOUND, // 25
    COVISE_MESSAGE_HAS_OBJECT_CHANGED, // 26
    COVISE_MESSAGE_OBJECT_UPDATE, // 27
    COVISE_MESSAGE_OBJECT_TRANSFER, // 28
    COVISE_MESSAGE_OBJECT_FOLLOWS, // 29
    COVISE_MESSAGE_OBJECT_OK, // 30
    COVISE_MESSAGE_CLOSE_SOCKET, // 31
    COVISE_MESSAGE_DESTROY_OBJECT, // 32
    COVISE_MESSAGE_CTRL_DESTROY_OBJECT, // 33
    COVISE_MESSAGE_QUIT, // 34
    COVISE_MESSAGE_START, // 35
    COVISE_MESSAGE_COVISE_ERROR, // 36
    COVISE_MESSAGE_INOBJ, // 37
    COVISE_MESSAGE_OUTOBJ, // 38
    COVISE_MESSAGE_OBJECT_NO_LONGER_USED, // 39
    COVISE_MESSAGE_SET_ACCESS, // 40
    COVISE_MESSAGE_FINALL, // 41
    COVISE_MESSAGE_ADD_OBJECT, // 42
    COVISE_MESSAGE_DELETE_OBJECT, // 43
    COVISE_MESSAGE_NEW_OBJECT_VERSION, // 44
    COVISE_MESSAGE_RENDER, // 45
    COVISE_MESSAGE_WAIT_CONTACT, // 46
    COVISE_MESSAGE_PARINFO, // 47
    COVISE_MESSAGE_MAKE_DATA_CONNECTION, // 48
    COVISE_MESSAGE_COMPLETE_DATA_CONNECTION, // 49
    COVISE_MESSAGE_SHM_FREE, // 50
    COVISE_MESSAGE_GET_TRANSFER_PORT, // 51
    COVISE_MESSAGE_TRANSFER_PORT, // 52
    COVISE_MESSAGE_CONNECT_TRANSFERMANAGER, // 53
    COVISE_MESSAGE_STDINOUT_EMPTY, // 54
    COVISE_MESSAGE_WARNING, // 55
    COVISE_MESSAGE_INFO, // 56
    COVISE_MESSAGE_REPLACE_OBJECT, // 57
    COVISE_MESSAGE_PLOT, // 58
    COVISE_MESSAGE_GET_LIST_OF_INTERFACES, // 59
    COVISE_MESSAGE_USR1, // 60
    COVISE_MESSAGE_USR2, // 61
    COVISE_MESSAGE_USR3, // 62
    COVISE_MESSAGE_USR4, // 63
    COVISE_MESSAGE_NEW_OBJECT_OK, // 64
    COVISE_MESSAGE_NEW_OBJECT_FAILED, // 65
    COVISE_MESSAGE_NEW_OBJECT_SHM_MALLOC_LIST, // 66
    COVISE_MESSAGE_REQ_UI, // 67
    COVISE_MESSAGE_NEW_PART_ADDED, // 68
    COVISE_MESSAGE_SENDING_NEW_PART, // 69
    COVISE_MESSAGE_FINPART, // 70
    COVISE_MESSAGE_NEW_PART_AVAILABLE, // 71
    COVISE_MESSAGE_OBJECT_ON_HOSTS, // 72
    COVISE_MESSAGE_OBJECT_FOLLOWS_CONT, // 73
    COVISE_MESSAGE_CRB_EXEC, // 74
    COVISE_MESSAGE_COVISE_STOP_PIPELINE, // 75
    COVISE_MESSAGE_PREPARE_CONTACT_MODULE, // 76
    COVISE_MESSAGE_MODULE_CONTACT_MODULE, // 77
    COVISE_MESSAGE_SEND_APPL_PROCID, // 78
    COVISE_MESSAGE_INTERFACE_LIST, // 79
    COVISE_MESSAGE_MODULE_LIST, // 80
    COVISE_MESSAGE_HOSTID, // 81
    COVISE_MESSAGE_MODULE_STARTED, // 82
    COVISE_MESSAGE_GET_USER, // 83
    COVISE_MESSAGE_SOCKET_CLOSED, // 84
    COVISE_MESSAGE_NEW_COVISED, // 85
    COVISE_MESSAGE_USER_LIST, // 86
    COVISE_MESSAGE_STARTUP_INFO, // 87
    COVISE_MESSAGE_CO_MODULE, // 88
    COVISE_MESSAGE_WRITE_SCRIPT, // 89
    COVISE_MESSAGE_CRB, // 90
    COVISE_MESSAGE_GENERIC, // 91
    COVISE_MESSAGE_RENDER_MODULE, // 92
    COVISE_MESSAGE_FEEDBACK, // 93
    COVISE_MESSAGE_VRB_CONTACT, // 94
    COVISE_MESSAGE_VRB_CONNECT_TO_COVISE, // 95
    COVISE_MESSAGE_VRB_CHECK_COVER, // 96
    COVISE_MESSAGE_END_IMM_CB, // 97
    COVISE_MESSAGE_NEW_DESK, // 98
    COVISE_MESSAGE_VRB_SET_USERINFO, // 99
    COVISE_MESSAGE_VRB_GET_ID, // 100
    COVISE_MESSAGE_VRB_SET_GROUP, // 101
    COVISE_MESSAGE_VRB_QUIT, // 102
    COVISE_MESSAGE_VRB_SET_MASTER, // 103
    COVISE_MESSAGE_VRB_GUI, // 104
    COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION, // 105
    COVISE_MESSAGE_VRB_REQUEST_FILE, // 106
    COVISE_MESSAGE_VRB_SEND_FILE, // 107
    COVISE_MESSAGE_VRB_CURRENT_FILE, // 108
    COVISE_MESSAGE_CRB_QUIT, // 109
    COVISE_MESSAGE_REMOVED_HOST, // 110
    COVISE_MESSAGE_START_COVER_SLAVE, // 111
    COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED, // 112
    COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED, // 113
    COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS, // 114
    COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE, // 115
    COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY, // 116
    COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE, // 117
    COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY, // 118
    COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS, // 119
    COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE, // 120
    COVISE_MESSAGE_SYNCHRONIZED_ACTION, // 121
    COVISE_MESSAGE_ACCESSGRID_DAEMON, // 122
    COVISE_MESSAGE_TABLET_UI, // 123
    COVISE_MESSAGE_QUERY_DATA_PATH, // 124
    COVISE_MESSAGE_SEND_DATA_PATH, // 125
    COVISE_MESSAGE_VRB_FB_RQ, // 126
    COVISE_MESSAGE_VRB_FB_SET, // 127
    COVISE_MESSAGE_VRB_FB_REMREQ, // 128
    COVISE_MESSAGE_UPDATE_LOADED_MAPNAME, // 129
    COVISE_MESSAGE_CRB_EXEC_DEBUG, // 130
    COVISE_MESSAGE_CRB_EXEC_MEMCHECK, // 131
    COVISE_MESSAGE_SSLDAEMON, // 132
    COVISE_MESSAGE_VISENSO_UI, // 133
    COVISE_MESSAGE_PARAMDESC, // 134
    COVISE_MESSAGE_LAST_DUMMY_MESSAGE, // 135
    COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION, //136
    COVISE_MESSAGE_VRBC_SET_SESSION, //137
    COVISE_MESSAGE_VRBC_SEND_SESSIONS, //138
    COVISE_MESSAGE_VRBC_CHANGE_SESSION, //139
    COVISE_MESSAGE_VRBC_UNOBSERVE_SESSION //140
};

#ifdef DEFINE_MSG_TYPES
NETEXPORT const char *covise_msg_types_array[COVISE_MESSAGE_LAST_DUMMY_MESSAGE+1] = {
    "FAILED", //  0
    "OK", //  1
    "INIT", //  2
    "FINISHED", //  3
    "SEND", //  4
    "ALLOC", //  5
    "UI", //  6
    "APP_CONTACT_DM", //  7
    "DM_CONTACT_DM", //  8
    "SHM_MALLOC", //  9
    "SHM_MALLOC_LIST", // 10
    "MALLOC_OK", // 11
    "MALLOC_LIST_OK", // 12
    "MALLOC_FAILED", // 13
    "PREPARE_CONTACT", // 14
    "PREPARE_CONTACT_DM", // 15
    "PORT", // 16
    "GET_SHM_KEY", // 17
    "NEW_OBJECT", // 18
    "GET_OBJECT", // 19
    "REGISTER_TYPE", // 20
    "NEW_SDS", // 21
    "SEND_ID", // 22
    "ASK_FOR_OBJECT", // 23
    "OBJECT_FOUND", // 24
    "OBJECT_NOT_FOUND", // 25
    "HAS_OBJECT_CHANGED", // 26
    "OBJECT_UPDATE", // 27
    "OBJECT_TRANSFER", // 28
    "OBJECT_FOLLOWS", // 29
    "OBJECT_OK", // 30
    "CLOSE_SOCKET", // 31
    "DESTROY_OBJECT", // 32
    "CTRL_DESTROY_OBJECT", // 33
    "QUIT", // 34
    "START", // 35
    "COVISE_ERROR", // 36
    "INOBJ", // 37
    "OUTOBJ", // 38
    "OBJECT_NO_LONGER_USED", // 39
    "SET_ACCESS", // 40
    "FINALL", // 41
    "ADD_OBJECT", // 42
    "DELETE_OBJECT", // 43
    "NEW_OBJECT_VERSION", // 44
    "RENDER", // 45
    "WAIT_CONTACT", // 46
    "PARINFO", // 47
    "MAKE_DATA_CONNECTION", // 48
    "COMPLETE_DATA_CONNECTION", // 49
    "SHM_FREE", // 50
    "GET_TRANSFER_PORT", // 51
    "TRANSFER_PORT", // 52
    "CONNECT_TRANSFERMANAGER", // 53
    "STDINOUT_EMPTY", // 54
    "WARNING", // 55
    "INFO", // 56
    "REPLACE_OBJECT", // 57
    "PLOT", // 58
    "GET_LIST_OF_INTERFACES", // 59
    "USR1", // 60
    "USR2", // 61
    "USR3", // 62
    "USR4", // 63
    "NEW_OBJECT_OK", // 64
    "NEW_OBJECT_FAILED", // 65
    "NEW_OBJECT_SHM_MALLOC_LIST", // 66
    "REQ_UI", // 67
    "NEW_PART_ADDED", // 68
    "SENDING_NEW_PART", // 69
    "FINPART", // 70
    "NEW_PART_AVAILABLE", // 71
    "OBJECT_ON_HOSTS", // 72
    "OBJECT_FOLLOWS_CONT", // 73
    "CRB_EXEC", // 74
    "COVISE_STOP_PIPELINE", // 75
    "PREPARE_CONTACT_MODULE", // 76
    "MODULE_CONTACT_MODULE", // 77
    "SEND_APPL_PROCID", // 78
    "INTERFACE_LIST", // 79
    "MODULE_LIST", // 80
    "HOSTID", // 81
    "MODULE_STARTED", // 82
    "GET_USER", // 83
    "SOCKET_CLOSED", // 84
    "NEW_COVISED", // 85
    "USER_LIST", // 86
    "STARTUP_INFO", // 87
    "CO_MODULE", // 88
    "WRITE_SCRIPT", // 89
    "CRB", // 90
    "GENERIC", // 91
    "RENDER_MODULE", // 92
    "FEEDBACK", // 93
    "VRB_CONTACT", // 94
    "VRB_CONNECT_TO_COVISE", // 95
    "VRB_CHECK_COVER", // 96
    "END_IMM_CB", // 97
    "NEW_DESK", // 98
    "VRB_SET_USERINFO", // 99
    "VRB_GET_ID", // 100
    "VRB_SET_GROUP", // 101
    "VRB_QUIT", // 102
    "VRB_SET_MASTER", // 103
    "VRB_GUI", // 104
    "VRB_CLOSE_VRB_CONNECTION", // 105
    "VRB_REQUEST_FILE", // 106
    "VRB_SEND_FILE", // 107
    "VRB_CURRENT_FILE", // 108
    "CRB_QUIT", // 109
    "REMOVED_HOST", // 110
    "START_COVER_SLAVE", // 111
    "VRB_REGISTRY_ENTRY_CHANGED", // 112
    "VRB_REGISTRY_ENTRY_DELETED", // 113
    "VRB_REGISTRY_SUBSCRIBE_CLASS", // 114
    "VRB_REGISTRY_SUBSCRIBE_VARIABLE", // 115
    "VRB_REGISTRY_CREATE_ENTRY", // 116
    "VRB_REGISTRY_SET_VALUE", // 117
    "VRB_REGISTRY_DELETE_ENTRY", // 118
    "VRB_REGISTRY_UNSUBSCRIBE_CLASS", // 119
    "VRB_REGISTRY_UNSUBSCRIBE_VARIABLE", // 120
    "SYNCHRONIZED_ACTION", // 121
    "ACCESSGRID_DAEMON", // 122
    "TABLET_UI", // 123
    "QUERY_DATA_PATH", // 124
    "SEND_DATA_PATH", // 125
    "VRB_FB_RQ", // 126
    "VRB_FB_SET", // 127
    "VRB_FB_REMREQ", // 128
    "UPDATE_LOADED_MAPNAME", // 129
    "CRB_EXEC_DEBUG", // 130
    "CRB_EXEC_MEMCHECK", // 131
    "SSLDAEMON", // 132
    "VISENSO_UI", // 133
    "COVISE_MESSAGE_PARAMDESC", //134
    "LAST_DUMMY_MESSAGE",
};
#else
NETEXPORT extern const char *covise_msg_types_array[COVISE_MESSAGE_LAST_DUMMY_MESSAGE+1];
#endif

enum sender_type
{
    UNDEFINED = 0,
    CONTROLLER = 1,
    DATAMANAGER = 2,
    USERINTERFACE = 3,
    RENDERER = 4,
    APPLICATIONMODULE = 5,
    TRANSFERMANAGER = 6,
    SIMPLEPROCESS = 7,
    SIMPLECONTROLLER = 8,
    STDINOUT = 9,
    COVISED = 10
};

NETEXPORT bool isVrbMessageType(int type);

}
#endif
