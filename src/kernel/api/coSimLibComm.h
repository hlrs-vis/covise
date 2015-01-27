/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COSIMLIBCOMM_H
#define COSIMLIBCOMM_H

enum CommandType
{
    COMM_ERROR = -1,
    COMM_NONE = 0,
    COMM_QUIT, /* 01 */
    COMM_TEST, /* 02 */
    EXEC_COVISE, /* 03 */
    GET_SLI_PARA, /* 04 */
    GET_SC_PARA_FLO, /* 05 */
    GET_SC_PARA_INT, /* 06 */
    GET_CHOICE_PARA, /* 07 */
    GET_BOOL_PARA, /* 08 */
    GET_TEXT_PARA, /* 09 */
    GET_FILE_PARA, /* 10 */
    SEND_USG, /* 11 */
    SEND_1DATA, /* 12 */
    SEND_3DATA, /* 13 */
    EXEC_MOD, /* 14 */
    PARA_INIT, /* 15 */
    PARA_PORT, /* 16 */
    PARA_CELL_MAP, /* 17 */
    PARA_VERTEX_MAP, /* 18 */
    PARA_NODE, /* 19 */
    ATTRIBUTE, /* 20 */
    GEO_DIM, /* 21 */
    BOCO_DIM, /* 22 */
    SEND_GEO, /* 23 */
    SEND_BOCO, /* 24 */
    NEWBC, /* 25 */
    USE_BOCO2, /* 26 */
    COMM_EXIT, /* 27 */
    COMM_DETACH, /* 28 */
    GET_INITIAL_PARA_DONE, /* 29 */
    GET_V3_PARA_FLO /* 30 */
};

#endif
