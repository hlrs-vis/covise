/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TABLET_UI_MESSAGES_H
#define CO_TABLET_UI_MESSAGES_H

enum TabletAction {
    TABLET_CREATE = 1,
    TABLET_REMOVE,
    TABLET_SET_VALUE,
    TABLET_QUIT
};

////////////////////////////////////////////////////////////
//ObjectTypes
enum TabletObjectType
{
    TABLET_BUTTON = 1,
    TABLET_BITMAP_BUTTON,
    TABLET_TEXT_FIELD,
    TABLET_TAB_FOLDER,
    TABLET_TOGGLE_BUTTON,
    TABLET_BITMAP_TOGGLE_BUTTON,
    TABLET_MESSAGE_BOX,
    TABLET_EDIT_FIELD,
    TABLET_INT_EDIT_FIELD,
    TABLET_FLOAT_EDIT_FIELD,
    TABLET_SPIN_EDIT_FIELD,
    TABLET_SLIDER,
    TABLET_COMBOBOX,
    TABLET_LISTBOX,
    TABLET_TAB,
    TABLET_SPLITTER,
    TABLET_FRAME,
    TABLET_FLOAT_SLIDER,
    TABLET_MAP,
    TABLET_NAV_ELEMENT,
    TABLET_TEXT_SPIN_EDIT_FIELD,
    TABLET_PROGRESS_BAR,
    TABLET_COLOR_TRIANGLE,
    TABLET_BROWSER_TAB = TABLET_COLOR_TRIANGLE+2,
    TABLET_COLOR_TAB,
    TABLET_TEXT_EDIT_FIELD,
    TABLET_ANNOTATION_TAB,
    TABLET_FILEBROWSER_BUTTON,
    TABLET_FUNCEDIT_TAB,
    TABLET_COLOR_BUTTON,
    TABLET_SCROLLAREA,
    TABLET_POPUP,
    TABLET_UI_TAB,
    TABLET_GROUPBOX,
};

////////////////////////////////////////////////////////////
// EVENTS
enum TabletEvent  {
    TABLET_PRESSED = 1,
    TABLET_RELEASED,
    TABLET_ACTIVATED,
    TABLET_DISACTIVATED,
    TABLET_MOVED,
};

////////////////////////////////////////////////////////////
// VALUES
enum TabletValue {
    TABLET_BOOL = 1,
    TABLET_INT,
    TABLET_FLOAT,
    TABLET_STRING,
    TABLET_MIN,
    TABLET_MAX,
    TABLET_STEP,
    TABLET_NUM_TICKS,
    TABLET_POS,
    TABLET_ADD_ENTRY,
    TABLET_REMOVE_ENTRY,
    TABLET_SELECT_ENTRY,
    TABLET_LABEL,
    TABLET_SIZE,
    TABLET_STYLE,
    TABLET_SHAPE,
    TABLET_ADD_MAP,
    TABLET_RGBA,
    TABLET_RED,
    TABLET_GREEN,
    TABLET_BLUE,
    TABLET_TEX,
    TABLET_CLICK,
    TABLET_ALL_TEXTURES,
    TABLET_TEX_MODE,
    TABLET_TEX_CHANGE,
    TABLET_TEX_UPDATE,
    TABLET_TRAVERSED_TEXTURES,
    TABLET_ECHOMODE,
    TABLET_IPADDRESS,
    TABLET_NODE_TEXTURES,
    TABLET_NO_TEXTURES,
    TABLET_TF_WIDGET_LIST,
    TABLET_TF_HISTOGRAM,
    TABLET_TEX_PORT,
    TABLET_LOADFILE_SATE,
    TABLET_SIM_SHOW_HIDE,
    TABLET_SIM_SETSIMPAIR,
    TABLET_INIT_SECOND_CONNECTION,
    TABLET_TYPE,
    TABLET_ORIENTATION,
    TABLET_COLOR,
    TABLET_SET_HIDDEN,
    TABLET_REMOVE_ALL,
    TABLET_SET_ENABLED,
    TABLET_SLIDER_SCALE,

    ////////////////////////////////////////////////////////////
    // VALUES from SGBrowser (100)
    TABLET_BROWSER_UPDATE = 100,
    TABLET_BROWSER_NODE,
    TABLET_BROWSER_END,
    TABLET_BROWSER_CURRENT_NODE,
    TABLET_BROWSER_SELECTED_NODE,
    TABLET_BROWSER_CLEAR_SELECTION,
    TABLET_BROWSER_SHOW_NODE,
    TABLET_BROWSER_HIDE_NODE,
    TABLET_BROWSER_EXPAND_UPDATE,
    TABLET_BROWSER_FIND,
    TABLET_BROWSER_COLOR,
    TABLET_BROWSER_WIRE,
    TABLET_BROWSER_SEL_ONOFF,
    TABLET_BROWSER_REMOVE_NODE,
    TABLET_BROWSER_PROPERTIES,
    TABLET_BROWSER_LOAD_FILES,
    GET_SHADER,
    GET_UNIFORMS,
    GET_SOURCE,
    UPDATE_UNIFORM,
    UPDATE_VERTEX,
    UPDATE_TESSCONTROL,
    UPDATE_TESSEVAL,
    UPDATE_FRAGMENT,
    UPDATE_GEOMETRY,
    SET_NUM_VERT,
    SET_INPUT_TYPE,
    SET_OUTPUT_TYPE,

////////////////////////////////////////////////////////////
// ANNOTATION (200)
    TABLET_ANNOTATION_NEW = 200,
    TABLET_ANNOTATION_DELETE,
    TABLET_ANNOTATION_DELETE_ALL,
    TABLET_ANNOTATION_SCALE,
    TABLET_ANNOTATION_SCALE_ALL,
    TABLET_ANNOTATION_SET_COLOR,
    TABLET_ANNOTATION_SET_ALL_COLORS,
    TABLET_ANNOTATION_SHOW_OR_HIDE,
    TABLET_ANNOTATION_SHOW_OR_HIDE_ALL,
    TABLET_ANNOTATION_SEND_TEXT,
    TABLET_ANNOTATION_CHANGE_NEW_BUTTON_STATE,
    TABLET_ANNOTATION_SET_SELECTION,

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
    TABLET_FB_FILE_SEL = 300,

/**
 * Sends the requested file to the requesting client
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Size of binary buffer
 *    3..n - Binary buffer
 */
    TABLET_SET_FILE,
/**
 * Sets the currently selected directory
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - string containing the directory
 */
    TABLET_SET_CURDIR,
/**
 * Sets the list of directories for the Filebrowser's directory-list.
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the directory list
 *    3..n - Entries Directorylist entries
 */
    TABLET_SET_DIRLIST,
/**
 * Sets the list of files for the Filebrowser's file-list.
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the file list
 *    3..n - Entries Filelist entries
 */
    TABLET_SET_FILELIST,
/**
 * Sets the used filter in the Filebrowser's filter combobox.
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Entry The filter to be set
 */
    TABLET_SET_FILTER,
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
    TABLET_REQ_DIRLIST,
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
    TABLET_REQ_FILELIST,
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
    TABLET_REQ_CURDIR,
/**
 * Requests a file list based on the given filter and filesystem location
 * Message format:
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Filter to be set to be the new filter
 */
    TABLET_REQ_FILTERCHANGE,
/**
 * Requests the current directory location to be set to the given one
 * Message format:
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 *    3 - Directory to be accepted as the new current directory
 */
    TABLET_REQ_DIRCHANGE,
/**
 * Requests a list of connected remote clients to vrb/controler
 * Message format:
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 */
    TABLET_REQ_CLIENTS,
/**
 * Sets a list of connected remote clients in the TUI FileBrowser dialog
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 */
    TABLET_SET_CLIENTS,
/**
 * Sets a location for file info retrieval
 * Message format:
 *    0 - ID  Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Location to use for file info retrieval
 */
    TABLET_SET_LOCATION,

/**
 * Request the content of the local home directory
 * Only on local machine
 * Message format:
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    2 - Type Covise Message Type (this define)
 */
    TABLET_REQ_HOME,

/**
 * Request a list of root drives
 * Only on local machine
 * Message format:
 *    0 - ID   Tablet-Sender ID
 *    1 - ID of VRB client
 *    1 - Type Covise Message Type (this define)
 *    2 - Location/Request target client (IP)
 */
    TABLET_REQ_DRIVES,

/**
 * Message for setting the requested drives on FileBrowser-side
 * Message format:
 *    0 - ID   Tablet-Receiver ID
 *    1 - Type Covise Message Type (this define)
 *    2 - Size Number of list entries of the file list
 *    3..n - Entries DriveList entries
 */
    TABLET_SET_DRIVES,
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
    TABLET_REMSET_DIRLIST,
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
    TABLET_REMSET_FILELIST,
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
    TABLET_REMSET_FILE,
/**
 * Message for sending a directory from a remote client to the vrb
 * for forwarding/updating to requesting client
 * Message format:
 *    0 - Type Covise Message Type (this define)
 *    1 - Remote Client Sender Id
 *    2 - Tablet-Target receiver Id
 *    3 - string containing the absolute path
 */
    TABLET_REMSET_DIRCHANGE,
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
    TABLET_REMSET_DRIVES,
/**
 * Message for sending the mode of the dialog to
 * be used
 * Message format:
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 *    2 - Mode (1 = Open, 2 = Save)
 */
    TABLET_SET_MODE,
/**
 * Message for sending the filterlist settings
 * to the TUIFileBrowserDialog
 * Message format:
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 *    2 - List of file-extension filters in the format:
 *        "*.<ext>;*.<ext>"
 */
    TABLET_SET_FILTERLIST,

/**
 * Message for request ing the current location the FileBrowser
 * is pointing to.
 * Message format:
 *    0 - Type Covise Message Type (this define)
 *    1 - Tablet-Target receiver Id
 */
    TABLET_REQ_LOCATION,

/**
 * Message to check for active connection to VRB Server
 * 0 - SenderID
 * 1 - This message id
 */
    TABLET_REQ_VRBSTAT,

/**
 * Message to set the state of the OpenCOVER VRB connection
 * 0 - Message Type
 * 1 - Sender ID
 * 2 - VRB Connection state
 */
    TABLET_SET_VRBSTAT,

/**
 * Message to request the directories from remote HOME
 * 0 - Message Type
 * 1 - sending TUI object ID
 * 2 - Sender ID (VRB)
 * 3 - Location (RemoteHost)
 */
    TABLET_REQ_HOMEDIR,

/**
 * Message to request the files from remote HOME
 * 0 - Message Type
 * 1 - sending TUI object ID
 * 2 - Sender ID (VRB)
 * 3 - File extension filter
 * 4 - Location (RemoteHost)
 */
    TABLET_REQ_HOMEFILES,

/**
 * Message to indicate that a requested VRB-Server file was not retrievable
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - opt. Code
 */
    TABLET_SET_FILE_NOSUCCESS,

/**
 * Message to indicate that a requested Remote-OpenCOVER file was not retrievable
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - opt. Code
 */
    TABLET_REMSET_FILE_NOSUCCESS,

/**
 * Message to collaborative request status of OpenCOVER client
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 */
    TABLET_REQ_MASTER,

/**
 * Message to indicate whether OpenCOVER client is collaborative master
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - bool value indicating whether OpenCOVER client is collaborative master
 */
    TABLET_SET_MASTER,

/**
 * Message to signal to VRB server that a file should be
 * loaded for all partners
 * 0 - Message Type
 * 1 - Sender ID (VRB
 * 2 - char* url of the file
 */
    TABLET_REQ_GLOBALLOAD,

/**
 * Message to signal to VRB server that a file should be
 * loaded for all partners
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - char* url of the file
 */
    TABLET_SET_GLOBALLOAD,

/**
 * Reports a path being selected in the FileBrowser
 * for saving
 * 0 - Message Type
 * 1 - Sender ID (VRB)
 * 2 - char* selected path
 */
    TABLET_FB_PATH_SELECT,

////////////////////////////////////////////////////////////
// UI (400)
/**
 * Sends the UI description to the tablet ui
 * 0 - Message type
 * 1 - Sender ID
 * 2 - UTF16 encoded XML designer file content
 */
    TABLET_UI_USE_DESCRIPTION = 400,

/**
 * Sends a command from the tablet ui to covise
 * 0 - Message type
 * 1 - Sender ID
 * 2 - char* target ("de.hlrs.covise"/"de.hlrs.cover"/...)
 * 3 - UTF16 encoded command
 */
    TABLET_UI_COMMAND,
};


////////////////////////////////////////////////////////////
// MESSAGES

// Visitor Modes (SGBrowser)
enum TabletSgVisitor {
    UPDATE_NODES = 0,
    CLEAR_SELECTION,
    SET_SELECTION,
    SHOW_NODE,
    HIDE_NODE,
    UPDATE_EXPAND,
    FIND_NODE,
    CURRENT_NODE,
    UPDATE_COLOR,
    UPDATE_WIRE,
    UPDATE_SEL,
    LOAD_FILES,
};

// Property Modes (SGBrowser)
enum TabletSgProperty {
SET_PROPERTIES = 0,
GET_PROPERTIES,
REMOVE_TEXTURE,
//#define GET_SHADER,
//#define GET_UNIFORMS,
//#define GET_SOURCE,
SET_SHADER,
SET_UNIFORM,
SET_VERTEX,
SET_FRAGMENT,
REMOVE_SHADER,
STORE_SHADER,
//UPDATE_UNIFORM,
//UPDATE_VERTEX,
//UPDATE_FRAGMENT,
CENTER_OBJECT,
SET_GEOMETRY,
//UPDATE_GEOMETRY,
//SET_NUM_VERT,
//SET_INPUT_TYPE,
//SET_OUTPUT_TYPE,
SET_TESSCONTROL,
SET_TESSEVAL,
//UPDATE_TESSCONTROL,
//UPDATE_TESSEVAL,
};

enum TabletSgSendMode {
    SEND_IMAGES = 0,
    SEND_LIST,
};

enum TabletSgShader {
    SHADER_UNIFORM = 0,
    SHADER_FRAGMENT,
    SHADER_VERTEX,
    SHADER_GEOMETRY,
    SHADER_TESSCONTROL,
    SHADER_TESSEVAL,
};

////////////////////////////////////////////////////////////
//Texture Environment Mode
enum TabletTexEnv {
    TEX_ENV_DECAL = 0,
    TEX_ENV_MODULATE,
    TEX_ENV_BLEND,
    TEX_ENV_REPLACE,
    TEX_ENV_ADD,
};

////////////////////////////////////////////////////////////
//TexGen Mode
enum TabletTexGen {
    TEX_GEN_NONE = 0,
    TEX_GEN_OBJECT_LINEAR,
    TEX_GEN_EYE_LINEAR,
    TEX_GEN_SPHERE_MAP,
    TEX_GEN_NORMAL_MAP,
    TEX_GEN_REFLECTION_MAP,
};

////////////////////////////////////////////////////////////
//Node Types
enum TabletSgNode {
    SG_NODE = 0,
    SG_GEODE,
    SG_BILLBOARD,
    SG_GROUP,
    SG_CLEAR_NODE,
    SG_COORDINATE_SYSTEM_NODE,
    SG_LIGHT_SOURCE,
    SG_LOD,
    SG_PAGED_LOD,
    SG_OCCLUDER_NODE,
    SG_PROJECTION,
    SG_PROXY_NODE,
    SG_SEQUENCE,
    SG_SWITCH,
    SG_TEX_GEN_NODE,
    SG_TRANSFORM,
    SG_AUTO_TRANSFORM,
    SG_MATRIX_TRANSFORM,
    SG_POSITION_ADDITUDE_TRANSFORM,
    SG_CLIP_NODE,
    SG_SIM_NODE,
};

enum TabletSliderScale {
    TABLET_SLIDER_LINEAR = 1,
    TABLET_SLIDER_LOGARITHMIC,
};

#endif
