/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: PlotCommunication.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//static char rcsid[] = "$Id: PlotCommunication.C,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $";

//**************************************************************************
//
// * Description    : This is the communication message handler for the renderer
//                    all messages going through here
//                    comin' from ports or the renderer
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
// * History : 29.07.93 V 1.0
//
//
//
//**************************************************************************
//
// debug stuff (local use)
//
#ifdef DEBUG
#define DBG
#endif

//
// ec stuff
//
#include <covise/covise.h>
#include <covise/covise_process.h>
#include <covise/covise_appproc.h>
#include <net/covise_connect.h>
#include <net/message.h>
#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>

//
//  class definition
//
#include "PlotCommunication.h"
#include "PlotPort.h"

#include <unistd.h>
#include "globals.h"
#include "extern.h"
void set_plotstr_string(plotstr *pstr, char *buf);
#include <Xm/Xm.h>
#include "xprotos.h"
#include "motifinc.h"
#undef RENDER

//
// prototypes
//

//
// coDoSet handling
//
using namespace covise;

static int anzset = 0;
static char *setnames[MAXSETS];
static int elemanz[MAXSETS];
static char **elemnames[MAXSETS];
static int gno = 0;
static char *define_symbol_string = NULL;
static char *labelstring = NULL, *string_s, *strings1, *strings2, *strings3, *strings4;
static int sdata1, sdata2, sdata3;
static int *selecteds, selectedcnt;
static tickmarks t;
static const char *commands;
int savedretval; // globale Variable fuer GetSelectedSet

//
// external info
//
extern class ApplicationProcess *appmod;
extern int port;
extern char *host;
extern int proc_id;
extern int socket_id;
extern char *instance;

//
// need for the ec virtual constructors
//
extern coDistributedObject *coDoFloat_vc(coShmArray *arr);
extern coDistributedObject *coDoVec3_vc(coShmArray *arr);

typedef struct _Int_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget sum_item;
    Widget *region_item;
    Widget rinvert_item;
} Int_ui;

extern Int_ui iui;
extern void do_seasonal_diff(int setno, int period);
extern void do_ntiles(int gno, int setno, int nt);

//#########################################################################
// PlotCommunication
//#########################################################################

//=========================================================================
// constructor
//=========================================================================
PlotCommunication::PlotCommunication()
{
}

//=========================================================================
// parse the message string
//=========================================================================
int PlotCommunication::parseMessage(char *line, char *token[], int tmax, char *sep)
{
    char *tp;
    int count;

    count = 0;
    tp = strtok(line, sep);
    for (count = 0; count < tmax && tp != NULL;)
    {
        token[count] = tp;
        tp = strtok(NULL, sep);
        count++;
    }
    token[count] = NULL;
    return count;
}

//==========================================================================
// send a command message
//==========================================================================
void PlotCommunication::sendCommandMessage(plot_command_type command, int data1, int data2)
{
    char DataBuffer[MAXDATALEN];
    sprintf(DataBuffer, "COMMAND\n%d %d %d\n", (int)command, data1, data2);
    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);

}

//==========================================================================
// send a command message
//==========================================================================
void PlotCommunication::sendCommand_FloatMessage(plot_command_type command, double data1, double data2, double data3, double data4, double data5, double data6, double data7, double data8, double data9, double data10)
{
    char DataBuffer[MAXDATALEN];
    sprintf(DataBuffer, "COMMAND_F\n%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", (int)command, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10);
    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);
}

//==========================================================================
// send a command message
//==========================================================================
void PlotCommunication::sendCommand_StringMessage(plot_command_type command, char *string)
{
    char DataBuffer[MAXDATALEN];
    sprintf(DataBuffer, "COMMAND_S\n%d %d %s\n", (int)command, (int)strlen(string), string);
    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);
}

//==========================================================================
// send a command message
//==========================================================================
void PlotCommunication::sendCommand_ValuesMessage(plot_command_type command, int data1, int data2, int data3, int data4, int data5, int data6, int data7, int data8, int data9, int data10)
{
    char DataBuffer[MAXDATALEN];
    sprintf(DataBuffer, "COMMAND_V\n%d %d %d %d %d %d %d %d %d %d %d\n", (int)command, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10);
    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);
}

//==========================================================================
// send a quit message
//==========================================================================
void PlotCommunication::sendQuitMessage()
{
    char DataBuffer[MAXDATALEN];
    char *key = (char *)"";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    Message msg{ COVISE_MESSAGE_QUIT , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);

    print_comment(__LINE__, __FILE__, "sended quit message");

}

//==========================================================================
// send a finish message
//==========================================================================
void PlotCommunication::sendFinishMessage()
{
    char DataBuffer[MAXDATALEN];
    char *key = (char *)"";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    Message msg{ COVISE_MESSAGE_FINISHED , DataHandle{DataBuffer, strlen(DataBuffer) + 1, false} };
    appmod->send_ctl_msg(&msg);

    print_comment(__LINE__, __FILE__, "sended finished message");

}

//==========================================================================
// receive a add object  message
//==========================================================================
void PlotCommunication::receiveAddObjectMessage(char *object, int doreplace)
{

    int no_elems = 0, no_sets = 0, x_n = 0, y_n, z_n, j, i, n, k;
    float *x_p, *y_p, *z_p;
    double *x_d = NULL, *y_d = NULL;
    const char *dataType, *commandss;

    const coDoVec2 *data;
    const coDoSet *set;
    const coDoRectilinearGrid *data2;
    const coDistributedObject *data_obj;
    const coDistributedObject *const *dobjs = 0L;

    if (doreplace)
    {
        receiveDeleteObjectMessage(object);
    }
    set_wait_cursor();
    data_obj = coDistributedObject::createFromShm(object);
    if (data_obj != 0L)
    {
        dataType = data_obj->getType();
        if (strcmp(dataType, "USTSTD") == 0)
        {

            data = (const coDoVec2 *)data_obj;
            no_elems = data->getNumPoints();
            data->getAddresses(&x_p, &y_p);
            x_d = (double *)calloc(no_elems, sizeof(double));
            y_d = (double *)calloc(no_elems, sizeof(double));
            for (i = 0; i < no_elems; i++)
            {
                x_d[i] = (double)x_p[i];
                y_d[i] = (double)y_p[i];
            }
            commands = NULL;
            commands = data->getAttribute("COMMANDS");
        }
        else if (strcmp(dataType, "SETELE") == 0)
        {
            set = (const coDoSet *)data_obj;
            if (set != NULL)
            {
                // Get Set
                setnames[anzset] = new char[strlen(object) + 1];
                strcpy(setnames[anzset], object);
                dobjs = set->getAllElements(&no_sets);
                elemanz[anzset] = no_sets;
                elemnames[anzset] = new char *[no_sets];
                commandss = NULL;
                commandss = set->getAttribute("COMMANDS");

                for (i = 0; i < no_sets; i++)
                {
                    no_elems = 0;
                    elemnames[anzset][i] = dobjs[i]->getName();
                    if (strcmp(dobjs[i]->getType(), "USTSTD") == 0)
                    {
                        data = (const coDoVec2 *)dobjs[i];
                        no_elems = data->getNumPoints();
                        data->getAddresses(&x_p, &y_p);
                        x_d = (double *)calloc(no_elems, sizeof(double));
                        y_d = (double *)calloc(no_elems, sizeof(double));
                        for (n = 0; n < no_elems; n++)
                        {
                            x_d[n] = (double)x_p[n];
                            y_d[n] = (double)y_p[n];
                        }
                        commands = NULL;
                        commands = data->getAttribute("COMMANDS");
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: Data object of set has wrong data type");
                        unset_wait_cursor();
                        return;
                    }
                    if (no_elems > 0)
                    {
                        for (k = 0; k < gno; k++)
                        {
                            if (strncmp(object, getcomment(k, 0), strlen(object)) == 0)
                                break;
                        }
                        if (k == gno)
                            for (k = 0; k < gno; k++)
                            {
                                if (strlen(getcomment(k, 0)) == 0)
                                    break;
                            }
                        if ((j = nextset(k)) == -1)
                        {
                            unset_wait_cursor();
                            return;
                        }
                        activateset(k, j);
                        settype(k, j, XY);
                        setcol(k, x_d, j, no_elems, 0);
                        setcol(k, y_d, j, no_elems, 1);
                        setcomment(k, j, elemnames[anzset][i]);
                        log_results(elemnames[anzset][i]);
                        updatesetminmax(k, j);
                        parseCommands(commands);
                        if (k == gno)
                            gno++;
                    }
                }
                no_elems = 0;
                anzset++;
                parseCommands(commandss); // command of 'top level' set
                commandss = NULL;
                commands = NULL;
                drawgraph();
            }
        }
        else if (strcmp(dataType, "RCTGRD") == 0)
        {
            data2 = (const coDoRectilinearGrid *)data_obj;
            data2->getGridSize(&x_n, &y_n, &z_n);
            data2->getAddresses(&x_p, &y_p, &z_p);
            x_d = (double *)calloc(x_n, sizeof(double));
            y_d = (double *)calloc(x_n, sizeof(double));
            for (i = 0; i < x_n; i++)
            {
                x_d[i] = (double)x_p[i];
                y_d[i] = (double)y_p[i];
            }
        }
        else
        {

            print_comment(__LINE__, __FILE__, "ERROR: Data object has wrong data type");
            unset_wait_cursor();
            return;
        }
        if (no_elems > 0)
        {
            //addset(object,x_d,y_d,no_elems,doreplace);
            addset(object, x_d, y_d, no_elems, 0);
        }
        else if (x_n > 0)
        {
            //addset(object,x_d,y_d,x_n,doreplace);
            addset(object, x_d, y_d, x_n, 0);
        }
    }
    else
    {
#ifndef TOLERANT
        print_comment(__LINE__, __FILE__, "ERROR: Data object can't be accessed in shared memory");
#endif
        unset_wait_cursor();
        return;
    }
    if ((no_elems == 0) && (no_sets == 0))
    {
        print_comment(__LINE__, __FILE__, "WARNING: Data object is empty");
    }

    unset_wait_cursor();
}

//==========================================================================
// receive a object delete message
//==========================================================================
void PlotCommunication::receiveDeleteObjectMessage(char *object)
{
    int i, n;
    for (i = 0; i < anzset; i++)
    {
        if (strcmp(setnames[i], object) == 0)
        {
            for (n = 0; n < elemanz[i]; n++)
                deleteObject(elemnames[i][n]);
            delete[] elemnames[i];
            n = i;
            anzset--;
            while (n < (anzset))
            {
                elemanz[n] = elemanz[n + 1];
                elemnames[n] = elemnames[n + 1];
                setnames[n] = setnames[n + 1];
                n++;
            }
            return;
        }
    }
    deleteObject(object);
}

//==========================================================================
// receive a command message
//==========================================================================
void PlotCommunication::receiveCommandMessage(char *message)
{
    // char DataBuffer[MAXDATALEN];
    long data1, data2, i;
    // double sum=0.0;
    plot_command_type command;
    sscanf(message, "%d %ld %ld", (int *)&command, &data1, &data2);
    switch (command)
    {
    case AUTOSCALE_PROC:
        autoscale_proc((Widget)NULL, (XtPointer)data1, (XtPointer)data2);
        drawgraph();
        break;
    case AUTOTICKS_PROC:
        autoticks_proc((Widget)NULL, (XtPointer)data1, (XtPointer)data2);
        break;
    case SET_PAGE:
        set_page((Widget)NULL, (XtPointer)data1, (XtPointer)data2);
        break;
    case GWINDLEFT_PROC:
        gwindleft_proc();
        break;
    case GWINDRIGHT_PROC:
        gwindright_proc();
        break;
    case GWINDUP_PROC:
        gwindup_proc();
        break;
    case GWINDDOWN_PROC:
        gwinddown_proc();
        break;
    case GWINDSHRINK_PROC:
        gwindshrink_proc();
        break;
    case GWINDEXPAND_PROC:
        gwindexpand_proc();
        break;
    case SCROLL_PROC:
        scroll_proc(data1);
        break;
    case SCROLLINOUT_PROC:
        scrollinout_proc(data1);
        break;
    case PUSH_AND_ZOOM:
        push_and_zoom();
        break;
    case CYCLE_WORLD_STACK:
        cycle_world_stack();
        break;
    case PUSH_WORLD:
        push_world();
        break;
    case POP_WORLD:
        pop_world();
        break;
    case SET_ACTION:
        set_action(data1);
        break;
    case SETALL_COLORS_PROC:
        setall_colors_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case SETALL_SYM_PROC:
        setall_sym_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case SETALL_LINEW_PROC:
        setall_linew_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case SET_CSET_PROC:
        set_cset_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case LEGEND_LOAD_PROC:
        legend_load_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case ACCEPT_LEDIT_PROC:
        set_plotstr_string(&g[data1].l.str[data2], labelstring);
        break;
    case ACCEPT_SYMMISC:
        accept_symmisc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case DEFINE_BOXPLOT_PROC:
        define_boxplot_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case SET_AXIS_PROC:
        set_axis_proc((Widget)NULL, (XtPointer)data1, (XtPointer)NULL);
        break;
    case DRAWGRAPH:
        drawgraph();
        break;
    case PAGE_SPECIAL_NOTIFY_PROC:
        page_special_notify_proc((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        break;
    case DO_COMPUTE_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_compute(selecteds[i], data1, data2, string_s);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_WINDOW_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_window(selecteds[i], data1, data2);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_DIFFER_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_differ(selecteds[i], data1);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_INT_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_int(selecteds[i], data1);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_SEASONAL_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_seasonal_diff(selecteds[i], data1);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_INTERP_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            /* do_interp(selecteds[i], data1); */
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_DIGFILTER_PROC:
        set_wait_cursor();
        do_digfilter(data1, data2);
        unset_wait_cursor();
        break;
    case DO_LINEARC_PROC:
        set_wait_cursor();
        do_linearc(data1, data2);
        unset_wait_cursor();
        break;
    case DO_COMPUTE_PROC2:
        set_wait_cursor();
        do_compute2(strings1, strings2, strings3, strings4, data1, data2);
        unset_wait_cursor();
        break;
    case DO_NTILES_PROC:
        set_wait_cursor();
        do_ntiles(cg, data1, data2);
        unset_wait_cursor();
        break;
    case DO_DEACTIVATE:
        set_wait_cursor();
        do_deactivate(cg, data1);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_REACTIVATE:
        set_wait_cursor();
        do_reactivate(cg, data1);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_CHANGETYPE:
        set_wait_cursor();
        setcomment(cg, data1, strings1);
        setname(cg, data1, strings2);
        do_changetype(data1, data2);
        unset_wait_cursor();
        break;
    case DO_SETLENGTH:
        set_wait_cursor();
        do_setlength(data1, data2);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_REVERSE_SETS:
        set_wait_cursor();
        do_reverse_sets(data1);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_COALESCE_SETS:
        set_wait_cursor();
        do_coalesce_sets(data1);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_KILL:
        set_wait_cursor();
        do_kill(cg, data1, data2);
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_WRITESETS_BINARY:
        set_wait_cursor();
        do_writesets_binary(cg, data1, strings1);
        unset_wait_cursor();
        break;
    case DO_SPLITSETS:
        set_wait_cursor();
        do_splitsets(cg, data1, data2);
        unset_wait_cursor();
        drawgraph();
        break;
    default:
        print_comment(__LINE__, __FILE__, "WARNING: command not implemented");
        break;
    }
}

//==========================================================================
// receive a string message
//==========================================================================
void PlotCommunication::receiveCommand_StringMessage(char *message)
{
    plot_command_type command;
    char *string, *buf;
    int slen = 0, i;
    sscanf(message, "%d %d ", (int *)&command, &slen);
    string = new char[slen + 1];
    sscanf(message, "%d %d %s", (int *)&command, &slen, string);
    switch (command)
    {
    case DEFINE_SYMBOLS3:
        //  if(define_symbol_string != NULL)
        //	delete(define_symbol_string);
        define_symbol_string = string;
        break;
    case ACCEPT_LEDIT_PROC:
        labelstring = string;
        break;
    case TICKS_DEFINE_NOTIFY_PROC:
        get_graph_tickmarks(cg, &t, curaxis);
        set_plotstr_string(&t.label, string);
        delete string;
        break;
    case ACCEPT_TICKLABEL_PROC:
        strcpy(t.tl_appstr, string);
        delete string;
        break;
    case ACCEPT_TICKLABEL_PROC2:
        strcpy(t.tl_prestr, string);
        delete string;
        break;
    case ACCEPT_SPECIAL_PROC:
        set_plotstr_string(&t.t_speclab[sdata1], string);
        break;
    case DO_COMPUTE_PROC:
        if (string_s != NULL)
            delete (string_s);
        string_s = string;
        break;
    case DO_COMPUTE_PROC2_1:
        if (strings1 != NULL)
            delete (strings1);
        strings1 = string;
        break;
    case DO_COMPUTE_PROC2_2:
        if (strings2 != NULL)
            delete (strings2);
        strings2 = string;
        break;
    case DO_COMPUTE_PROC2_3:
        if (strings3 != NULL)
            delete (strings3);
        strings3 = string;
        break;
    case DO_COMPUTE_PROC2_4:
        if (strings4 != NULL)
            delete (strings4);
        strings4 = string;
        break;
    case DO_SAMPLE_PROC:
        set_wait_cursor();
        buf = new char[slen + 1];
        for (i = 0; i < selectedcnt; i++)
        {
            strcpy(buf, string);
            do_sample(selecteds[i], sdata1, buf, sdata2, sdata3);
        }
        delete (selecteds);
        selecteds = NULL;
        delete (string);
        delete (buf);
        unset_wait_cursor();
        drawgraph();
        break;
    default:
        print_comment(__LINE__, __FILE__, "WARNING: String command not implemented");
        break;
    }
}

//==========================================================================
// receive a float message
//==========================================================================
void PlotCommunication::receiveCommand_FloatMessage(char *message)
{
    plot_command_type command;
    int i, j;
    int minset, maxset;
    int k, order[3];
    double degrees, sx, sy, rotx, roty, tx, ty, xtmp, ytmp, *x, *y;
    double data1, data2, data3, data4, data5, data6, data7, data8, data9, data10;
    sscanf(message, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", (int *)&command, &data1, &data2, &data3, &data4, &data5, &data6, &data7, &data8, &data9, &data10);
    switch (command)
    {
    case DEFINE_LEGENDS_PROC:
        g[cg].l.legx = data1;
        g[cg].l.legy = data2;
        g[cg].l.charsize = data3;
        break;
    case DEFINE_ERRBAR_PROC:
        g[cg].p[(int)data2].errbarper = data1;
        break;
    case DEFINE_ARRANGE:
        define_arrange(sdata1, sdata2, sdata3, data1, data2, data3, data4, data5, data6);
        break;
    case ACCEPT_AXIS_PROC:
        get_graph_tickmarks(cg, &t, curaxis);
        t.alt = OFF;
        t.tmin = 0.0;
        t.tmax = 1.0;
        t.offsx = data1;
        t.offsy = data2;
        set_graph_tickmarks(cg, &t, curaxis);
        drawgraph();
        break;
    case ACCEPT_SPECIAL_PROC:
        get_graph_tickmarks(cg, &t, curaxis);
        sdata1 = (int)data1;
        t.t_specloc[sdata1] = data2;
        break;
    case ACCEPT_TICKLABEL_PROC2:
        t.tl_start = data1;
        t.tl_stop = data2;
        break;
    case TICKS_DEFINE_NOTIFY_PROC:
        t.tmajor = data1;
        t.tminor = data2;
        break;
    case DO_HISTO_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_histo(selecteds[i], (int)data1, (int)data2, data3, data4, data5, (int)data6);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_SPLINE_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_spline(selecteds[i], data1, data2, (int)data3);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_GEOM_PROC:
        switch ((int)data3)
        {
        case 0:
            order[0] = 0; /* rotate */
            order[1] = 1; /* translate */
            order[2] = 2; /* scale */
            break;
        case 1:
            order[0] = 0;
            order[1] = 2;
            order[2] = 1;
        case 2:
            order[0] = 1;
            order[1] = 2;
            order[2] = 0;
            break;
        case 3:
            order[0] = 1;
            order[1] = 0;
            order[2] = 2;
            break;
        case 4:
            order[0] = 2;
            order[1] = 1;
            order[2] = 0;
            break;
        case 5:
            order[0] = 2;
            order[1] = 0;
            order[2] = 1;
            break;
        }
        set_wait_cursor();
        degrees = data4;
        tx = data5;
        ty = data6;
        sx = data7;
        sy = data8;
        rotx = data9;
        roty = data10;
        minset = (int)data1;
        maxset = (int)data2;
        for (k = minset; k <= maxset; k++)
        {
            if (isactive(cg, k))
            {
                x = getx(cg, k);
                y = gety(cg, k);
                for (j = 0; j < 3; j++)
                {
                    switch (order[j])
                    {
                    case 0:
                        if (degrees == 0.0)
                        {
                            break;
                        }
                        for (i = 0; i < getsetlength(cg, k); i++)
                        {
                            xtmp = x[i] - rotx;
                            ytmp = y[i] - roty;
                            x[i] = rotx + cos(degrees) * xtmp - sin(degrees) * ytmp;
                            y[i] = roty + sin(degrees) * xtmp + cos(degrees) * ytmp;
                        }
                        break;
                    case 1:
                        for (i = 0; i < getsetlength(cg, k); i++)
                        {
                            x[i] -= tx;
                            y[i] -= ty;
                        }
                        break;
                    case 2:
                        for (i = 0; i < getsetlength(cg, k); i++)
                        {
                            x[i] *= sx;
                            y[i] *= sy;
                        }
                        break;
                    } /* end case */
                } /* end for j */
                updatesetminmax(cg, k);
                update_set_status(cg, k);
            } /* end if */
        } /* end for k */
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_ACTIVATE:
        t.tmajor = data1;
        t.tminor = data2;
        break;

    default:
        print_comment(__LINE__, __FILE__, "WARNING: Float command not implemented");
        break;
    }
}

void PlotCommunication::receiveCommand_ValuesMessage(char *message)
{
    // char DataBuffer[MAXDATALEN];
    long data1, data2, data3, data4, data5, data6, data7, data8, data9, data10;
    static int sym, symchar, symskip, symfill, symcolor, symlinew, symlines, cset;
    int line, pen, wid, fill, fillusing, fillpat, fillcol, j, i, set_mode;
    double symsize;
    // char s[30];
    XEvent event;
    event.type = 0; //
    XButtonEvent *b_event = (XButtonEvent *)&event;
    XMotionEvent *m_event = (XMotionEvent *)&event;
    plot_command_type command;
    sscanf(message, "%d %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld", (int *)&command, &data1, &data2, &data3, &data4, &data5, &data6, &data7, &data8, &data9, &data10);
    switch (command)
    {
    case BOXES_DEF_PROC:
        box_color = data1;
        box_loctype = data2;
        box_lines = data3;
        box_linew = data4;
        box_fill = data5;
        box_fillcolor = data6;
        box_fillpat = data7;
        break;
    case LINES_DEF_PROC:
        line_asize = data1 / 50.0;
        line_color = data2;
        line_arrow = data3;
        line_atype = data4;
        line_lines = data5;
        line_linew = data6;
        line_loctype = data7;
        break;
    case DEFINE_STRING_DEFAULTS:
        string_font = data1;
        string_color = data2;
        string_size = data3 / 100.0;
        string_rot = data4;
        string_loctype = data5;
        string_just = data6;
        break;
    case MY_PROC:
        event.type = data1;
        event.xbutton.button = data2;
        event.xmotion.x = data3;
        event.xmotion.y = data4;
        event.xmotion.state = data5;
        b_event->time = data6;
        m_event->x = data7;
        m_event->y = data8;
        my_proc((Widget)data9, (caddr_t)1, &event);
        break;
    case DEFINE_LEGENDS_PROC:
        g[cg].l.active = data2;
        g[cg].l.vgap = data3;
        g[cg].l.len = data4;
        g[cg].l.loctype = data5;
        g[cg].l.font = data8;
        g[cg].l.color = data9;
        break;
    case DEFINE_LEGENDS_PROC2:
        g[cg].l.box = data1;
        g[cg].l.boxfill = data2;
        g[cg].l.boxfillusing = data3;
        g[cg].l.boxfillcolor = data4;
        g[cg].l.boxfillpat = data5;
        g[cg].l.boxlcolor = data6;
        g[cg].l.boxlinew = data7;
        g[cg].l.boxlines = data8;
        update_ledit_items(cg);
        drawgraph();
        break;
    case DEFINE_ERRBAR_PROC:
        g[cg].p[data8].errbarxy = data2;
        g[cg].p[data8].errbar_linew = data3;
        g[cg].p[data8].errbar_lines = data4;
        g[cg].p[data8].errbar_riser = data5;
        g[cg].p[data8].errbar_riser_linew = data6;
        g[cg].p[data8].errbar_riser_lines = data7;
        if (data9)
            drawgraph();
        break;
    case DEFINE_SYMBOLS:
        symskip = data1;
        symfill = data2;
        symcolor = data3;
        symlinew = data4;
        symlines = data5;
        symchar = data6;
        cset = data7;
        break;
    case DEFINE_SYMBOLS2:
        symsize = data1 / 100.0;
        sym = data2;
        pen = data3;
        wid = data4;
        line = data5;
        fill = data6;
        fillusing = data7;
        fillpat = data8;
        fillcol = data9;
        set_mode = data10;
        if (set_mode == 0)
        {
            g[cg].p[cset].symskip = symskip;
            g[cg].p[cset].symsize = symsize;
            g[cg].p[cset].symchar = symchar;
            g[cg].p[cset].symfill = symfill;
            g[cg].p[cset].symlinew = symlinew;
            g[cg].p[cset].symlines = symlines;
            g[cg].p[cset].fill = fill;
            g[cg].p[cset].fillusing = fillusing;
            g[cg].p[cset].fillpattern = fillpat;
            g[cg].p[cset].fillcolor = fillcol;
            set_plotstr_string(&g[cg].l.str[cset], define_symbol_string);
            setplotsym(cg, cset, sym);
            setplotlines(cg, cset, line);
            setplotlinew(cg, cset, wid);
            setplotcolor(cg, cset, pen);
            setplotsymcolor(cg, cset, symcolor);
        }
        else
        {
            for (i = 0; i < g[cg].maxplot; i++)
            {
                if (isactive(cg, i))
                {
                    g[cg].p[i].symskip = symskip;
                    g[cg].p[i].symsize = symsize;
                    g[cg].p[i].symchar = symchar;
                    g[cg].p[i].symfill = symfill;
                    g[cg].p[i].symlinew = symlinew;
                    g[cg].p[i].symlines = symlines;
                    g[cg].p[i].fill = fill;
                    g[cg].p[i].fillusing = fillusing;
                    g[cg].p[i].fillpattern = fillpat;
                    g[cg].p[i].fillcolor = fillcol;
                    setplotsym(cg, i, sym);
                    setplotlines(cg, i, line);
                    setplotlinew(cg, i, wid);
                    setplotcolor(cg, i, pen);
                    setplotsymcolor(cg, i, symcolor);
                }
            }
        }
        updatesymbols(cg, cset);
        drawgraph();
        break;
    case DEFINE_AUTOS:
        define_autos(data1, data2, data3, data4, data5, data6);
        break;
    case DEFINE_ARRANGE:
    case DO_SAMPLE_PROC:
        sdata1 = data1;
        sdata2 = data2;
        sdata3 = data3;
        break;
    case TICKS_DEFINE_NOTIFY_PROC:
        t.tl_flag = data2;
        t.t_flag = data3;
        t.t_drawbar = data4;
        switch (data1)
        {
        case 0: /* current axis */
            set_graph_tickmarks(cg, &t, curaxis);
            break;
        case 1: /* all axes, current graph */
            for (i = 0; i < MAXAXES; i++)
            {
                g[cg].t[i].tl_flag = t.tl_flag;
                g[cg].t[i].t_flag = t.t_flag;
                g[cg].t[i].t_drawbar = t.t_drawbar;
                set_plotstr_string(&g[cg].t[i].label, t.label.s);
                g[cg].t[i].tmajor = t.tmajor;
                g[cg].t[i].tminor = t.tminor;
            }
            break;
        case 2: /* current axis, all graphs */
            for (i = 0; i < maxgraph; i++)
            {
                g[i].t[curaxis].tl_flag = t.tl_flag;
                g[i].t[curaxis].t_flag = t.t_flag;
                g[i].t[curaxis].t_drawbar = t.t_drawbar;
                set_plotstr_string(&g[i].t[curaxis].label, t.label.s);
                g[i].t[curaxis].tmajor = t.tmajor;
                g[i].t[curaxis].tminor = t.tminor;
            }
            break;
        case 3: /* all axes, all graphs */
            for (i = 0; i < maxgraph; i++)
            {
                for (j = 0; j < 6; j++)
                {
                    g[i].t[j].tl_flag = t.tl_flag;
                    g[i].t[j].t_flag = t.t_flag;
                    g[i].t[j].t_drawbar = t.t_drawbar;
                    set_plotstr_string(&g[i].t[j].label, t.label.s);
                    g[i].t[j].tmajor = t.tmajor;
                    g[i].t[j].tminor = t.tminor;
                }
            }
            break;
        }
        drawgraph();
        break;
    case ACCEPT_AXISLABEL_PROC:
        get_graph_tickmarks(cg, &t, curaxis);
        t.label_layout = data1;
        t.label.font = data2;
        t.label.color = data3;
        t.label.linew = data4;
        t.label.charsize = data5 / 100.0;
        set_graph_tickmarks(cg, &t, curaxis);
        drawgraph();
        break;
    case ACCEPT_TICKLABEL_PROC:
        get_graph_tickmarks(data1, &t, data2);
        t.tl_angle = data3;
        t.tl_layout = data4;
        t.tl_sign = data5;
        t.tl_op = data6;
        t.tl_font = data7;
        t.tl_color = data8;
        t.tl_linew = data9;
        t.tl_skip = data10;
        break;
    case ACCEPT_TICKLABEL_PROC2:
        t.tl_prec = data3;
        t.tl_staggered = data4;
        t.tl_starttype = data5;
        t.tl_stoptype = data6;
        t.tl_format = data7;
        t.tl_charsize = data10 / 100.0;
        set_graph_tickmarks(data1, &t, data2);
        break;
    case ACCEPT_TICKMARK_PROC:
        get_graph_tickmarks(data1, &t, data2);
        t.t_inout = data3;
        t.t_op = data4;
        t.t_color = data5;
        t.t_linew = data6;
        t.t_lines = data7;
        t.t_mcolor = data8;
        t.t_mlinew = data9;
        t.t_mlines = data10;
        break;
    case ACCEPT_TICKMARK_PROC2:
        t.t_size = data3 / 100.0;
        t.t_msize = data4 / 100.0;
        t.t_gridflag = data5;
        t.t_mgridflag = data6;
        set_graph_tickmarks(data1, &t, data2);
        break;
    case ACCEPT_AXISBAR_PROC:
        get_graph_tickmarks(cg, &t, curaxis);
        t.t_drawbarcolor = data1;
        t.t_drawbarlinew = data2;
        t.t_drawbarlines = data3;
        set_graph_tickmarks(cg, &t, curaxis);
        drawgraph();
        break;
    case ACCEPT_SPECIAL_PROC:
        t.t_type = data1;
        t.tl_type = data2;
        t.t_spec = data3;
        set_graph_tickmarks(cg, &t, curaxis);
        drawgraph();
        break;
    case GETSELECTEDSETS0:
        if (selecteds != NULL)
            delete (selecteds);
        selecteds = new int[data1 + 10];
        selectedcnt = data1;
        selecteds[0] = data2;
        selecteds[1] = data3;
        selecteds[2] = data4;
        selecteds[3] = data5;
        selecteds[4] = data6;
        selecteds[5] = data7;
        selecteds[6] = data8;
        selecteds[7] = data9;
        selecteds[8] = data10;
        break;
    case GETSELECTEDSETS1:
        selecteds[9] = data1;
        selecteds[10] = data2;
        selecteds[11] = data3;
        selecteds[12] = data4;
        selecteds[13] = data5;
        selecteds[14] = data6;
        selecteds[15] = data7;
        selecteds[16] = data8;
        selecteds[17] = data9;
        selecteds[18] = data10;
        break;
    case GETSELECTEDSETS2:
        selecteds[19] = data1;
        selecteds[20] = data2;
        selecteds[21] = data3;
        selecteds[22] = data4;
        selecteds[23] = data5;
        selecteds[24] = data6;
        selecteds[25] = data7;
        selecteds[26] = data8;
        selecteds[27] = data9;
        selecteds[28] = data10;
        break;
    case GETSELECTEDSETS3:
        selecteds[29] = data1;
        selecteds[30] = data2;
        selecteds[31] = data3;
        selecteds[32] = data4;
        selecteds[33] = data5;
        selecteds[34] = data6;
        selecteds[35] = data7;
        selecteds[36] = data8;
        selecteds[37] = data9;
        selecteds[38] = data10;
        break;
    case GETSELECTEDSET:
        savedretval = data1;
        break;
    case DO_FOURIER_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_fourier(data6, selecteds[i], data1, data2, data3, data4, data5);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_RUNAVG_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            do_runavg(selecteds[i], data1, data2, data3, data4);
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_REGRESS_PROC:
        set_wait_cursor();
        for (i = 0; i < selectedcnt; i++)
        {
            if (data1 == 11)
            {
                for (j = 1; j <= data1 - 1; j++)
                {
                    do_regress(selecteds[j], j, data2, data3, data4);
                }
            }
            else
            {
                do_regress(selecteds[i], data1, data2, data3, data4);
            }
        }
        delete (selecteds);
        selecteds = NULL;
        unset_wait_cursor();
        drawgraph();
        break;
    case DO_XCOR_PROC:
        set_wait_cursor();
        do_xcor(data1, data2, data3, data4);
        unset_wait_cursor();
        break;
    default:
        print_comment(__LINE__, __FILE__, "WARNING: command not implemented");
        break;
    }
}

//==========================================================================
// receive a master message
//==========================================================================
void PlotCommunication::receiveMasterMessage()
{
    ismaster = 1;
    set_toolbars(TOOLBAR, 1);
    set_menus(1);
    // fprintf(stderr,"Master !\n");
    drawgraph();
}

//==========================================================================
// receive a slave message
//==========================================================================
void PlotCommunication::receiveSlaveMessage()
{
    ismaster = 1;
    set_toolbars(TOOLBAR, 0);
    set_menus(0);
    ismaster = 0;
    drawgraph();
}

//==========================================================================
// receive a slave message
//==========================================================================
void PlotCommunication::receiveMasterSlaveMessage()
{
}

//==========================================================================
// delete object
//==========================================================================
void PlotCommunication::deleteObject(char *object)
{
    int i, j;
    for (i = 0; i < maxgraph; i++)
    {
        for (j = 0; j < g[i].maxplot; j++)
        {
            if (strcmp(getcomment(i, j), object) == 0)
            {
                killset(i, j);
                // set_default_graph(i);
            }
        }
    }
}

//==========================================================================
// add set
//==========================================================================
void PlotCommunication::addset(char *object, double *x_d, double *y_d, int no_elems, int doreplace)
{
    int i, j;
    if (doreplace)
    {
        for (i = 0; i < maxgraph; i++)
        {
            for (j = 0; j < g[i].maxplot; j++)
            {
                if (strcmp(getcomment(i, j), object) == 0)
                {
                    softkillset(i, j);
                    activateset(i, j);
                    settype(i, j, XY);
                    setcol(i, x_d, j, no_elems, 0);
                    setcol(i, y_d, j, no_elems, 1);
                    setcomment(i, j, object);
                    log_results(object);
                    updatesetminmax(i, j);
                    parseCommands(commands);
                    drawgraph();
                }
            }
        }
    }
    else
    {
        for (i = 0; i < gno; i++)
        {
            if (strlen(getcomment(i, 0)) == 0)
                break;
        }
        if ((j = nextset(i)) == -1)
        {
            unset_wait_cursor();
            return;
        }
        activateset(i, j);
        settype(i, j, XY);
        setcol(i, x_d, j, no_elems, 0);
        setcol(i, y_d, j, no_elems, 1);
        setcomment(i, j, object);
        log_results(object);
        updatesetminmax(i, j);
        parseCommands(commands);
        drawgraph();
        if (i == gno)
            gno++;
    }
}

//==========================================================================
// parse commands
//==========================================================================
void PlotCommunication::parseCommands(const char *commands)
{
    char *curtok;
    int errpos;
    char buf[500], *tmp_commands;
    double ascan, bscan, cscan, dscan, xscan, yscan;
    if (commands != NULL)
    {
        // do not modify commands directly !!
        tmp_commands = new char[strlen(commands) + 1];
        strcpy(tmp_commands, commands);
        lowtoupper(tmp_commands);
        curtok = strtok(tmp_commands, "\n");
        while (curtok != NULL)
        {
            strcpy(buf, curtok);
            fixupstr(buf);
            scanner(buf, &xscan, &yscan, 1, &ascan, &bscan, &cscan, &dscan, 1, 0, 0, &errpos);
            curtok = strtok(NULL, "\n");
        }
        delete tmp_commands;
    }
}
