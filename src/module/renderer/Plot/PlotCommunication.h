/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLOT_COMMUNICATION_H
#define _PLOT_COMMUNICATION_H

/* $Id: PlotCommunication.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: PlotCommunication.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  the communication message handler
//
//
// * Class(es)      : PlotCommunication
//
//
// * inherited from : none
//
//
// * Author  : Uwe Woessner
//
//
// * History : 11.11.94 V 1.0
//
//
//
//**************************************************************************
//
//
//

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>
//#include <X11/Xm/Xm.h>
#include <Xm/Xm.h>

//
// ec stuff
//
#include <covise/covise_process.h>

//
// CLASSES
//
class PlotCommunication;

//
// other classes
//
#include "PlotError.h"

//
// defines
//
#define MAXDATALEN 255
#define MAXTOKENS 25
#define MAXHOSTLEN 20
#define MAXSETS 200
enum plot_command_type
{
    EMPTY_COMMAND = -1,
    AUTOSCALE_PROC,
    SET_PAGE,
    AUTOTICKS_PROC,
    GWINDLEFT_PROC,
    GWINDRIGHT_PROC,
    GWINDDOWN_PROC,
    GWINDUP_PROC,
    GWINDSHRINK_PROC,
    GWINDEXPAND_PROC,
    SCROLL_PROC,
    SCROLLINOUT_PROC,
    PUSH_AND_ZOOM,
    CYCLE_WORLD_STACK,
    PUSH_WORLD,
    POP_WORLD,
    BOXES_DEF_PROC,
    LINES_DEF_PROC,
    DEFINE_STRING_DEFAULTS,
    SET_ACTION,
    MY_PROC,
    SETALL_COLORS_PROC,
    SETALL_SYM_PROC,
    SETALL_LINEW_PROC,
    SET_CSET_PROC,
    LEGEND_LOAD_PROC,
    ACCEPT_LEDIT_PROC,
    ACCEPT_SYMMISC,
    DEFINE_BOXPLOT_PROC,
    DEFINE_LEGENDS_PROC,
    DEFINE_LEGENDS_PROC2,
    DEFINE_ERRBAR_PROC,
    DEFINE_SYMBOLS,
    DEFINE_SYMBOLS2,
    DEFINE_SYMBOLS3,
    DEFINE_AUTOS,
    DEFINE_ARRANGE,
    SET_AXIS_PROC,
    TICKS_DEFINE_NOTIFY_PROC,
    ACCEPT_AXIS_PROC,
    ACCEPT_AXISLABEL_PROC,
    ACCEPT_TICKLABEL_PROC,
    ACCEPT_TICKLABEL_PROC2,
    ACCEPT_TICKMARK_PROC,
    ACCEPT_TICKMARK_PROC2,
    ACCEPT_AXISBAR_PROC,
    ACCEPT_SPECIAL_PROC,
    PAGE_SPECIAL_NOTIFY_PROC,
    DRAWGRAPH,
    DO_COMPUTE_PROC,
    GETSELECTEDSETS0,
    GETSELECTEDSETS1,
    GETSELECTEDSETS2,
    GETSELECTEDSETS3,
    DO_HISTO_PROC,
    DO_FOURIER_PROC,
    DO_FFT_PROC,
    DO_WINDOW_PROC,
    DO_RUNAVG_PROC,
    DO_REGRESS_PROC,
    DO_DIFFER_PROC,
    DO_INT_PROC,
    DO_SEASONAL_PROC,
    DO_INTERP_PROC,
    DO_XCOR_PROC,
    DO_SPLINE_PROC,
    DO_SAMPLE_PROC,
    DO_DIGFILTER_PROC,
    DO_LINEARC_PROC,
    DO_COMPUTE_PROC2_1,
    DO_COMPUTE_PROC2_2,
    DO_COMPUTE_PROC2_3,
    DO_COMPUTE_PROC2_4,
    DO_COMPUTE_PROC2,
    DO_NTILES_PROC,
    DO_GEOM_PROC,
    GETSELECTEDSET,
    DO_ACTIVATE,
    DO_ACTIVATE2,
    DO_DEACTIVATE,
    DO_REACTIVATE,
    DO_CHANGETYPE,
    DO_CHANGETYPE2,
    DO_SETLENGTH,
    DO_COPY,
    DO_MOVE,
    DO_SWAP,
    DO_DROP_POINTS,
    DO_JOIN_SETS,
    DO_REVERSE_SETS,
    DO_COALESCE_SETS,
    DO_KILL,
    DO_SORT,
    DO_WRITESETS,
    DO_WRITESETS2,
    DO_WRITESETS_BINARY,
    DO_SPLITSETS,
    MOVEGRAPH
};

//================================================================
// PlotCommunication
//================================================================

class PlotCommunication
{
private:
public:
    PlotCommunication();
    int parseMessage(char *line, char *token[], int tmax, char *sep);
    void sendCommandMessage(plot_command_type command, int data1, int data2);
    void sendCommand_StringMessage(plot_command_type command, char *string);
    void sendCommand_FloatMessage(plot_command_type command, double data1, double data2, double data3, double data4, double data5, double data6, double data7, double data8, double data9, double data10);
    void sendCommand_ValuesMessage(plot_command_type command, int data1, int data2, int data3, int data4, int data5, int data6, int data7, int data8, int data9, int data10);
    void sendQuitMessage();
    void sendFinishMessage();
    void receiveAddObjectMessage(char *message, int doreplace);
    void receiveDeleteObjectMessage(char *message);
    void receiveCommandMessage(char *message);
    void receiveCommand_FloatMessage(char *message);
    void receiveCommand_ValuesMessage(char *message);
    void receiveCommand_StringMessage(char *message);
    void receiveMasterMessage();
    void receiveSlaveMessage();
    void receiveMasterSlaveMessage();
    void addset(char *object, double *x_d, double *y_d, int no_elems, int doreplace);
    void deleteObject(char *object);
    void parseCommands(const char *commands);
    ~PlotCommunication(){};
};
#endif
