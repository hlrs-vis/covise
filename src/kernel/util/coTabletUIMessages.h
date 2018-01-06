/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TABLET_UI_MESSAGES_H
#define CO_TABLET_UI_MESSAGES_H

#define TABLET_CREATE 1
#define TABLET_REMOVE 2
#define TABLET_SET_VALUE 3
#define TABLET_QUIT 4

////////////////////////////////////////////////////////////
// EVENTS
#define TABLET_PRESSED 1
#define TABLET_RELEASED 2
#define TABLET_ACTIVATED 3
#define TABLET_DISACTIVATED 4

////////////////////////////////////////////////////////////
// VALUES
#define TABLET_BOOL 1
#define TABLET_INT 2
#define TABLET_FLOAT 3
#define TABLET_STRING 4
#define TABLET_MIN 5
#define TABLET_MAX 6
#define TABLET_STEP 7
#define TABLET_NUM_TICKS 8
#define TABLET_POS 9
#define TABLET_ADD_ENTRY 10
#define TABLET_REMOVE_ENTRY 11
#define TABLET_SELECT_ENTRY 12
#define TABLET_LABEL 13
#define TABLET_SIZE 14
#define TABLET_STYLE 15
#define TABLET_SHAPE 16
#define TABLET_ADD_MAP 17
#define TABLET_RGBA 18
#define TABLET_RED 19
#define TABLET_GREEN 20
#define TABLET_BLUE 21
#define TABLET_TEX 22
#define TABLET_CLICK 23
#define TABLET_ALL_TEXTURES 24
#define TABLET_TEX_MODE 25
#define TABLET_TEX_CHANGE 26
#define TABLET_TEX_UPDATE 27
#define TABLET_TRAVERSED_TEXTURES 28
#define TABLET_ECHOMODE 29
#define TABLET_IPADDRESS 30
#define TABLET_NODE_TEXTURES 31
#define TABLET_NO_TEXTURES 32
#define TABLET_TF_WIDGET_LIST 33
#define TABLET_TF_HISTOGRAM 34
#define TABLET_TEX_PORT 35
#define TABLET_LOADFILE_SATE 36
#define TABLET_SIM_SHOW_HIDE 37
#define TABLET_SIM_SETSIMPAIR 38
#define TABLET_INIT_SECOND_CONNECTION 39
#define TABLET_TYPE 40
#define TABLET_ORIENTATION 41
#define TABLET_COLOR 42
#define TABLET_SET_HIDDEN 43
#define TABLET_REMOVE_ALL 44
#define TABLET_SET_ENABLED 45

////////////////////////////////////////////////////////////
//ObjectTypes
#define TABLET_BUTTON 1
#define TABLET_BITMAP_BUTTON 2
#define TABLET_TEXT_FIELD 3
#define TABLET_TAB_FOLDER 4
#define TABLET_TOGGLE_BUTTON 5
#define TABLET_BITMAP_TOGGLE_BUTTON 6
#define TABLET_MESSAGE_BOX 7
#define TABLET_EDIT_FIELD 8
#define TABLET_INT_EDIT_FIELD 9
#define TABLET_FLOAT_EDIT_FIELD 10
#define TABLET_SPIN_EDIT_FIELD 11
#define TABLET_SLIDER 12
#define TABLET_COMBOBOX 13
#define TABLET_LISTBOX 14
#define TABLET_TAB 15
#define TABLET_SPLITTER 16
#define TABLET_FRAME 17
#define TABLET_FLOAT_SLIDER 18
#define TABLET_MAP 19
#define TABLET_NAV_ELEMENT 20
#define TABLET_TEXT_SPIN_EDIT_FIELD 21
#define TABLET_PROGRESS_BAR 22
#define TABLET_COLOR_TRIANGLE 23
#define TABLET_TEXTURE_TAB 24
#define TABLET_BROWSER_TAB 25
#define TABLET_COLOR_TAB 26
#define TABLET_TEXT_EDIT_FIELD 27
#define TABLET_ANNOTATION_TAB 28
#define TABLET_FILEBROWSER_BUTTON 29
#define TABLET_FUNCEDIT_TAB 30
#define TABLET_COLOR_BUTTON 31
#define TABLET_SCROLLAREA 32
#define TABLET_POPUP 33
#define TABLET_UI_TAB 34
#define TABLET_GROUPBOX 35

////////////////////////////////////////////////////////////
// MESSAGES

////////////////////////////////////////////////////////////
// SGBrowser (100)
#define TABLET_BROWSER_UPDATE 100
#define TABLET_BROWSER_NODE 101
#define TABLET_BROWSER_END 102
#define TABLET_BROWSER_CURRENT_NODE 103
#define TABLET_BROWSER_SELECTED_NODE 104
#define TABLET_BROWSER_CLEAR_SELECTION 105
#define TABLET_BROWSER_SHOW_NODE 106
#define TABLET_BROWSER_HIDE_NODE 107
#define TABLET_BROWSER_EXPAND_UPDATE 108
#define TABLET_BROWSER_FIND 109
#define TABLET_BROWSER_COLOR 110
#define TABLET_BROWSER_WIRE 111
#define TABLET_BROWSER_SEL_ONOFF 112
#define TABLET_BROWSER_REMOVE_NODE 113
#define TABLET_BROWSER_PROPERTIES 114
#define TABLET_BROWSER_LOAD_FILES 115

// Visitor Modes (SGBrowser)
#define UPDATE_NODES 0
#define CLEAR_SELECTION 1
#define SET_SELECTION 2
#define SHOW_NODE 3
#define HIDE_NODE 4
#define UPDATE_EXPAND 5
#define FIND_NODE 6
#define CURRENT_NODE 7
#define UPDATE_COLOR 8
#define UPDATE_WIRE 9
#define UPDATE_SEL 10
#define LOAD_FILES 11

// Property Modes (SGBrowser)
#define SET_PROPERTIES 0
#define GET_PROPERTIES 1
#define REMOVE_TEXTURE 2
#define GET_SHADER 3
#define GET_UNIFORMS 4
#define GET_SOURCE 5
#define SET_SHADER 6
#define SET_UNIFORM 7
#define SET_VERTEX 8
#define SET_FRAGMENT 9
#define REMOVE_SHADER 10
#define STORE_SHADER 11
#define UPDATE_UNIFORM 12
#define UPDATE_VERTEX 13
#define UPDATE_FRAGMENT 14
#define CENTER_OBJECT 15
#define SET_GEOMETRY 16
#define UPDATE_GEOMETRY 17
#define SET_NUM_VERT 18
#define SET_INPUT_TYPE 19
#define SET_OUTPUT_TYPE 20
#define SET_TESSCONTROL 21
#define SET_TESSEVAL 22
#define UPDATE_TESSCONTROL 23
#define UPDATE_TESSEVAL 24

#define SEND_IMAGES 0
#define SEND_LIST 1

#define SHADER_UNIFORM 0
#define SHADER_FRAGMENT 1
#define SHADER_VERTEX 2
#define SHADER_GEOMETRY 3
#define SHADER_TESSCONTROL 4
#define SHADER_TESSEVAL 5

////////////////////////////////////////////////////////////
// ANNOTATION (200)
#define TABLET_ANNOTATION_NEW 200
#define TABLET_ANNOTATION_DELETE 201
#define TABLET_ANNOTATION_DELETE_ALL 202
#define TABLET_ANNOTATION_SCALE 203
#define TABLET_ANNOTATION_SCALE_ALL 204
#define TABLET_ANNOTATION_SET_COLOR 205
#define TABLET_ANNOTATION_SET_ALL_COLORS 206
#define TABLET_ANNOTATION_SHOW_OR_HIDE 207
#define TABLET_ANNOTATION_SHOW_OR_HIDE_ALL 208
#define TABLET_ANNOTATION_SEND_TEXT 209
#define TABLET_ANNOTATION_CHANGE_NEW_BUTTON_STATE 210
#define TABLET_ANNOTATION_SET_SELECTION 211

////////////////////////////////////////////////////////////
//TabletUI - File Browser Messages (300)
/**
 * Requests a file from a remote point which is either
 * the VRB or another OpenCOVER client
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - IP  IP-address of the remote point
 *    4 - Path path and filename of the file requested
 */
#define TABLET_FB_FILE_SEL 300

/**
 * Sends the requested file to the requesting client
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Size of binary buffer
 *    3..n - Binary buffer
 */
#define TABLET_SET_FILE 301
/**
 * Sets the currently selected directory
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - string containing the directory
 */
#define TABLET_SET_CURDIR 302
/**
 * Sets the list of directories for the Filebrowser's directory-list.
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the directory list
 *    3..n - Entries Directorylist entries 
 */
#define TABLET_SET_DIRLIST 303
/**
 * Sets the list of files for the Filebrowser's file-list.
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the file list
 *    3..n - Entries Filelist entries 
 */
#define TABLET_SET_FILELIST 304
/**
 * Sets the used filter in the Filebrowser's filter combobox.
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Entry The filter to be set
 */
#define TABLET_SET_FILTER 305
/**
 * Requests a directory list based on the given filter and filesystem location
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Filter Filter given as c-string in format "*.<extension>"
 *    4 - Location at file system where to search given as absolute path
 *    5 - IP-Adress identifying the machine where to search
 */
#define TABLET_REQ_DIRLIST 306
/**
 * Requests a file list based on the given filter and filesystem location
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Filter Filter given as c-string in format "*.<extension>"
 *    4 - Location at file system where to search given as absolute path
 *    5 - IP-Adress identifying the machine where to search
 */
#define TABLET_REQ_FILELIST 307
/**
 * Requests a file list based on the given filter and filesystem location
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Filter Filter given as c-string in format "*.<extension>"
 *    4 - Location at file system where to search given as absolute path
 *    5 - IP-Adress identifying the machine where to search
 */
#define TABLET_REQ_CURDIR 308
/**
 * Requests a file list based on the given filter and filesystem location
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Filter to be set to be the new filter
 */
#define TABLET_REQ_FILTERCHANGE 309
/**
 * Requests the current directory location to be set to the given one
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Directory to be accepted as the new current directory
 */
#define TABLET_REQ_DIRCHANGE 310
/**
 * Requests a list of connected remote clients to vrb/controler
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 */
#define TABLET_REQ_CLIENTS 311
/**
 * Sets a list of connected remote clients in the TUI FileBrowser dialog
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 */
#define TABLET_SET_CLIENTS 312
/**
 * Sets a location for file info retrieval
 * Message format: 
 *    0 - ID  Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Location to use for file info retrieval
 */
#define TABLET_SET_LOCATION 313

/**
 * Request the content of the local home directory
 * Only on local machine
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 */
#define TABLET_REQ_HOME 314

/**
 * Request a list of root drives
 * Only on local machine
 * Message format: 
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    1 - Type Covise Message Type (this define)
 *    2 - Location/Request target client (IP)
 */
#define TABLET_REQ_DRIVES 315

/**
 * Message for setting the requested drives on FileBrowser-side
 * Message format: 
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the file list
 *    3..n - Entries DriveList entries 
 */
#define TABLET_SET_DRIVES 316
/**
 * Message for sending directory lists from a remote client to the vrb
 * for forwarding to requesting client
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - Size of directory list
 *    4..n - directory list entries 
 */
#define TABLET_REMSET_DIRLIST 317
/**
 * Message for sending file lists from a remote client to the vrb
 * for forwarding to requesting client
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - Size of file list
 *    4..n - file list entries 
 */
#define TABLET_REMSET_FILELIST 318
/**
 * Message for sending file lists from a remote client to the vrb
 * for forwarding to requesting client
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - Size of binary buffer
 *    4 - binary file buffer
 */
#define TABLET_REMSET_FILE 319
/**
 * Message for sending a directory from a remote client to the vrb
 * for forwarding/updating to requesting client
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - string containing the absolute path
 */
#define TABLET_REMSET_DIRCHANGE 320
/**
 * Message for sending file lists from a remote client to the vrb
 * for forwarding to requesting client
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - Size of drive list
 *    4..n - drive list entries
 */
#define TABLET_REMSET_DRIVES 321
/**
 * Message for sending the mode of the dialog to
 * be used
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 *    2 - Mode (1 = Open, 2 = Save)
 */
#define TABLET_SET_MODE 322
/**
 * Message for sending the filterlist settings 
 * to the TUIFileBrowserDialog
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 *    2 - List of file-extension filters in the format:
 *        "*.<ext>;*.<ext>"
 */
#define TABLET_SET_FILTERLIST 323

/**
 * Message for request ing the current location the FileBrowser
 * is pointing to.
 * Message format: 
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 */
#define TABLET_REQ_LOCATION 324

/**
 * Message to check for active connection to VRB Server 
 * 0 - SenderID
 * 1 - This message id
 */
#define TABLET_REQ_VRBSTAT 325

/**
 * Message to set the state of the OpenCOVER VRB connection
 * 0 - Message Type
 * 1 - Sender ID
 * 2 - VRB Connection state
 */
#define TABLET_SET_VRBSTAT 326

/**
 * Message to request the directories from remote HOME
 * 0 - Message Type
 * 1 - sending TUI object ID
 * 2 - Sender ID (VRB)
 * 3 - Location (RemoteHost)
 */
#define TABLET_REQ_HOMEDIR 327

/**
 * Message to request the files from remote HOME
 * 0 - Message Type
 * 1 - sending TUI object ID
 * 2 - Sender ID (VRB)
 * 3 - File extension filter
 * 4 - Location (RemoteHost)
 */
#define TABLET_REQ_HOMEFILES 328

/**
 * Message to indicate that a requested VRB-Server file was not retrievable
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - opt. Code
 */
#define TABLET_SET_FILE_NOSUCCESS 329

/**
 * Message to indicate that a requested Remote-OpenCOVER file was not retrievable
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - opt. Code
 */
#define TABLET_REMSET_FILE_NOSUCCESS 330

/**
 * Message to collaborative request status of OpenCOVER client
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 */
#define TABLET_REQ_MASTER 331

/**
 * Message to indicate whether OpenCOVER client is collaborative master
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - bool value indicating whether OpenCOVER client is collaborative master
 */
#define TABLET_SET_MASTER 332

/**
 * Message to signal to VRB server that a file should be 
 * loaded for all partners
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - char* url of the file
 */
#define TABLET_REQ_GLOBALLOAD 333

/**
 * Message to signal to VRB server that a file should be 
 * loaded for all partners
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - char* url of the file
 */
#define TABLET_SET_GLOBALLOAD 334

/**
 * Reports a path being selected in the FileBrowser
 * for saving
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - char* selected path
 */
#define TABLET_FB_PATH_SELECT 335

////////////////////////////////////////////////////////////
// UI (400)

/**
 * Sends the UI description to the tablet ui
 * 0 - Message type
 * 1 - Sender ID
 * 2 - UTF16 encoded XML designer file content
 */
#define TABLET_UI_USE_DESCRIPTION 400

/**
 * Sends a command from the tablet ui to covise
 * 0 - Message type
 * 1 - Sender ID
 * 2 - char* target ("de.hlrs.covise"/"de.hlrs.cover"/...)
 * 3 - UTF16 encoded command
 */
#define TABLET_UI_COMMAND 401

////////////////////////////////////////////////////////////
//Texture Environment Mode
#define TEX_ENV_DECAL 0
#define TEX_ENV_MODULATE 1
#define TEX_ENV_BLEND 2
#define TEX_ENV_REPLACE 3
#define TEX_ENV_ADD 4

////////////////////////////////////////////////////////////
//TexGen Mode
#define TEX_GEN_NONE 0
#define TEX_GEN_OBJECT_LINEAR 1
#define TEX_GEN_EYE_LINEAR 2
#define TEX_GEN_SPHERE_MAP 3
#define TEX_GEN_NORMAL_MAP 4
#define TEX_GEN_REFLECTION_MAP 5

////////////////////////////////////////////////////////////
//Node Types
#define SG_NODE 0
#define SG_GEODE 1
#define SG_BILLBOARD 2
#define SG_GROUP 3
#define SG_CLEAR_NODE 4
#define SG_COORDINATE_SYSTEM_NODE 5
#define SG_LIGHT_SOURCE 6
#define SG_LOD 7
#define SG_PAGED_LOD 8
#define SG_OCCLUDER_NODE 9
#define SG_PROJECTION 10
#define SG_PROXY_NODE 11
#define SG_SEQUENCE 12
#define SG_SWITCH 13
#define SG_TEX_GEN_NODE 14
#define SG_TRANSFORM 15
#define SG_AUTO_TRANSFORM 16
#define SG_MATRIX_TRANSFORM 17
#define SG_POSITION_ADDITUDE_TRANSFORM 18
#define SG_CLIP_NODE 19
#define SG_SIM_NODE 20

#endif
