/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : This is the communication message handler for the renderer
//                    all messages going through here
//                    comin' from ports or the renderer
//
// * Class(es)      : InvCommunication
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau Uwe Woessner
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
#include <covise/covise.h>
#ifdef DEBUG
#define DBG
#endif

#include <util/coStringTable.h>
using namespace covise;
coStringTable partNames;

//
// ec stuff
//
#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include <covise/covise_process.h>
#include <do/coDoText.h>
#include <do/coDoTexture.h>
#include <do/coDoPixelImage.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <appl/RenderInterface.h>

//
//  class definition
//
#include "InvCommunication.h"
#include "InvRenderManager.h"
#include "InvPort.h"
#include <util/coMaterial.h>

#include "coFeedbackParser.h"
//
// C stuff
//
#include <unistd.h>

//
// Time measurement stuff
//
#ifdef TIMING
#include "InvTimer.h"
#endif

//
// coDoSet handling
//
coMaterialList materialList("metal");

static int anzset = 0;
static char *setnames[MAXSETS];
static int elemanz[MAXSETS];
static char **elemnames[MAXSETS];

static char m_name[100];
static char inst_no[100];
static char h_name[100];

//
// external info
//
extern ApplicationProcess *appmod;
extern int port;
extern char *host;
extern int proc_id;
extern int socket_id;

//
// external functions in render manager
//
extern void rm_receiveCamera(char *message);
extern void rm_switchMaster();
extern void rm_switchSlave();
extern void rm_switchMasterSlave();
extern void rm_setRenderTime(float time);
extern void rm_receiveTransformation(char *message);
extern void rm_receiveTelePointer(char *message);
extern void rm_receiveDrawstyle(char *message);
extern void rm_receiveSyncMode(char *message);
extern void rm_receiveLightMode(char *message);
extern void rm_receiveSelection(char *message);
extern void rm_receiveDeselection(char *message);
extern void rm_receivePart(char *message);
extern void rm_receiveReferencePart(char *message);
extern void rm_receiveResetScene();
extern void rm_receiveTransparency(char *message);
extern void rm_receiveFog(char *message);
extern void rm_receiveAntialiasing(char *message);
extern void rm_receiveBackcolor(char *message);
extern void rm_receiveAxis(char *message);
extern void rm_receiveClippingPlane(char *message);
extern void rm_receiveViewing(char *message);
extern void rm_receiveProjection(char *message);
extern void rm_receiveDecoration(char *message);
extern void rm_receiveHeadlight(char *message);
extern void rm_receiveSequencer(char *message);
extern void rm_receiveColormap(char *message);
//
// external functions in object manager
//
#include "InvObjectManagerExtern.h"

//#########################################################################
// InvCommunication
//#########################################################################

struct Repl
{
    char *oldName, *newName;
    struct Repl *next;
    Repl(const char *oldN, const char *newN)
    {
        oldName = strcpy(new char[strlen(oldN) + 1], oldN);
        newName = strcpy(new char[strlen(newN) + 1], newN);
        next = NULL;
    }
};

static Repl dummy("", "");
static Repl *replace_ = &dummy;

//=========================================================================
// constructor
//=========================================================================
InvCommunication::InvCommunication()
{
    pk_ = 1000;
}

int InvCommunication::getNextPK()
{
    return pk_++;
}

int InvCommunication::getCurrentPK()
{
    return pk_;
}

//=====================================================================
//
//=====================================================================
const char *InvCommunication::getname(const char *file)
{
    char *dirname, *covisepath;
    FILE *fp;
    int i;
    static char *buf = NULL;
    static int buflen;

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        cerr << "ERROR: COVISE_PATH not defined!\n";
        return NULL;
    };
    if ((buf == NULL) || (buflen < (int)(strlen(covisepath) + strlen(file) + 20)))
    {
        buflen = strlen(covisepath) + strlen(file) + 100;
        delete[] buf;
        buf = new char[buflen];
    }
    char *coPath = new char[strlen(covisepath) + 1];
    strcpy(coPath, covisepath);
#ifdef _WIN32
    dirname = strtok(coPath, ";");
#else
    dirname = strtok(coPath, ":");
#endif
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            fclose(fp);
            delete[] coPath;
            return buf;
        }
        for (i = strlen(dirname) - 2; i > 0; i--)
        {
            if (dirname[i] == '/')
            {
                dirname[i] = '\0';
                break;
            }
            else if (dirname[i] == '\\')
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            fclose(fp);
            delete[] coPath;
            return buf;
        }
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    delete[] coPath;
    sprintf(buf, "%s", file);
    fp = ::fopen(buf, "r");
    if (fp != NULL)
    {
        fclose(fp);
        return buf;
    }
    buf[0] = '\0';
    return NULL;
}

//=========================================================================
// parse the message string
//=========================================================================
int InvCommunication::parseMessage(char *line, char *token[], int tmax, char *sep)
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
// send a camera message
//==========================================================================
void InvCommunication::sendCameraMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "CAMERA";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

#ifdef TIMING
    time_str = new char[100];
    sprintf(time_str, "%s: Sending TRANSFORM[%d]", appmod->getName(), transform_send_ctr++);
    covise_time->mark(__LINE__, time_str);
#endif
}

//==========================================================================
// send a VRMLcamera message
//==========================================================================
void InvCommunication::sendVRMLCameraMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "VRMLCAMERA";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

#ifdef TIMING
    time_str = new char[100];
    sprintf(time_str, "%s: Sending TRANSFORM[%d]", appmod->getName(), transform_send_ctr++);
    covise_time->mark(__LINE__, time_str);
#endif
}

//==========================================================================
// send a transform message
//==========================================================================
void InvCommunication::sendTransformMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "TRANSFORM";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

#ifdef TIMING
    time_str = new char[100];
    sprintf(time_str, "%s: Sending TRANSFORM[%d]", appmod->getName(), transform_send_ctr++);
    covise_time->mark(__LINE__, time_str);
#endif

}

//==========================================================================
// send a telepointer message
//==========================================================================
void InvCommunication::sendTelePointerMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "TELEPOINTER";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

#ifdef TIMING
    time_str = new char[100];
    sprintf(time_str, "%s: Sending TELEPOINTER[%d]", appmod->getName(), telePointer_send_ctr++);
    covise_time->mark(__LINE__, time_str);
#endif

}

//==========================================================================
// send a VRML telepointer message
//==========================================================================
void InvCommunication::sendVRMLTelePointerMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "VRML_TELEPOINTER";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

#ifdef TIMING
    time_str = new char[100];
    sprintf(time_str, "%s: Sending VRML_TELEPOINTER[%d]", appmod->getName(), telePointer_send_ctr++);
    covise_time->mark(__LINE__, time_str);
#endif


}

//==========================================================================
// send a drawstyle message
//==========================================================================
void InvCommunication::sendDrawstyleMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "DRAWSTYLE";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunication::sendLightModeMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "LIGHTMODE";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a colormap message
//==========================================================================
void InvCommunication::sendColormapMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "COLORMAP";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunication::sendSelectionMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "SELECTION";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

void
InvCommunication::sendCSFeedback(char *key, char *message)
{
    //     char DataBuffer[MAXDATALEN];
    std::string nl("\n");
    std::string dataBuf(key);
    dataBuf = dataBuf + nl;
    dataBuf = dataBuf + std::string(message);
    dataBuf = dataBuf + nl;
    send_ctl_msg(const_cast<char*>(dataBuf.c_str()), COVISE_MESSAGE_UI);
}

void
InvCommunication::sendAnnotation(char *key, char *message)
{


    std::string nl("\n");
    std::string dataBuf(key);
    dataBuf = dataBuf + nl;
    dataBuf = dataBuf + std::string(message);
    dataBuf = dataBuf + nl;

    send_ctl_msg(const_cast<char*>(dataBuf.c_str()), COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunication::sendDeselectionMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "DESELECTION";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a part switching message
//==========================================================================
void InvCommunication::sendPartMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "PART";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a reference part message
//==========================================================================
void InvCommunication::sendReferencePartMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "REFPART";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

//==========================================================================
// send a reset scene  message
//==========================================================================
void InvCommunication::sendResetSceneMessage()
{
    char DataBuffer[MAXDATALEN];
    const char *key = "RESET";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunication::sendTransparencyMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "TRANSPARENCY";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunication::sendSyncModeMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "SYNC";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a fog mode message
//==========================================================================
void InvCommunication::sendFogMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "FOG";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a aliasing mode message
//==========================================================================
void InvCommunication::sendAntialiasingMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "ANTIALIASING";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a back color message
//==========================================================================
void InvCommunication::sendBackcolorMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "BACKCOLOR";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a axis mode message
//==========================================================================
void InvCommunication::sendAxisMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "AXIS";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a clipping plane message
//==========================================================================
void InvCommunication::sendClippingPlaneMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "CLIP";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunication::sendViewingMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "VIEWING";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunication::sendProjectionMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "PROJECTION";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

//==========================================================================
// send a decoration mode message
//==========================================================================
void InvCommunication::sendDecorationMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "DECORATION";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);

}

//==========================================================================
// send a headlight mode message
//==========================================================================
void InvCommunication::sendHeadlightMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "HEADLIGHT";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);


}

//==========================================================================
// send a sequencer message
//==========================================================================
void InvCommunication::sendSequencerMessage(char *message)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "SEQUENCER";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, message);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_RENDER);
}

//==========================================================================
// send a quit message
//==========================================================================
void InvCommunication::sendQuitMessage()
{
    Message *msg = new Message;
    char DataBuffer[MAXDATALEN];
    const char *key = "";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    msg->type = COVISE_MESSAGE_QUIT;
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_REQ_UI);

    print_comment(__LINE__, __FILE__, "sended quit message");
}

//==========================================================================
// send a quit message
//==========================================================================
void InvCommunication::sendQuitRequestMessage()
{
    char DataBuffer[MAXDATALEN];

    strcpy(DataBuffer, m_name);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, inst_no);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, h_name);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, "QUIT");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_REQ_UI);

    print_comment(__LINE__, __FILE__, "sended quit message");
}

//==========================================================================
// send a finish message
//==========================================================================
void InvCommunication::sendFinishMessage()
{
    char DataBuffer[MAXDATALEN];
    const char *key = "";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_FINISHED);
    print_comment(__LINE__, __FILE__, "sended finished message");
}

//==========================================================================
// send a help message
//==========================================================================
void InvCommunication::sendShowHelpMessage(const char *url)
{
    char DataBuffer[MAXDATALEN];
    const char *key = "SHOW_HELP";

    strcpy(DataBuffer, key);
    strcat(DataBuffer, "\n");
    strcat(DataBuffer, url);
    strcat(DataBuffer, "\n");
    send_ctl_msg(DataBuffer, COVISE_MESSAGE_UI);
}

void InvCommunication::send_ctl_msg(const char* message, int type)
{
    Message msg{ (covise_msg_type)type, DataHandle{(char*)message, strlen(message) + 1, false} };
    appmod->send_ctl_msg(&msg);
}
//==========================================================================
// receive a add object  message
//==========================================================================
void InvCommunication::receiveAddObjectMessage(const coDistributedObject *data_obj,
                                               char *object, int doreplace)
{
    const coDistributedObject *dobj = 0L;
    const coDistributedObject *colors = 0L;
    const coDistributedObject *normals = 0L;
    const coDistributedObject *texture = 0L;
    const coDoGeometry *geometry = 0L;
    const coDoText *Text = 0L;
    char *IvData;
    const char *gtype;
    //char *tstep_attrib = NULL;
    int is_timestep = 0;
    int timestep = -1;

    /////// Either we hand in the object or we create it from the name
    if (!data_obj)
    {
        data_obj = coDistributedObject::createFromShm(object);
    }

    if (data_obj && data_obj->objectOk())
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "GEOMET") == 0)
        {
            geometry = (coDoGeometry *)data_obj;
            if (geometry->objectOk())
            {
                dobj = geometry->getGeometry();
                normals = geometry->getNormals();
                colors = geometry->getColors();
                texture = geometry->getTexture();
                gtype = dobj->getType();
                if (strcmp(gtype, "SETELE") == 0)
                {
                    // We've got a set !
                    if (doreplace) // Check if we have to replace the set. In this
                    { // case we delete the current set and build a new one.
                        receiveDeleteObjectMessage(object);
                        doreplace = 0;
                        addgeometry(object, doreplace, is_timestep, timestep,
                                    NULL, dobj, normals, colors, texture, geometry);
                    }
                    else
                        addgeometry(object, doreplace, is_timestep, timestep, NULL, dobj, normals, colors, texture, geometry);
                }
                else
                    addgeometry(object, doreplace, is_timestep, timestep, NULL, dobj, normals, colors, texture, geometry);
            }
            delete geometry;
        }
        else if (strcmp(gtype, "DOTEXT") == 0)
        {
            Text = (coDoText *)data_obj;
            Text->getAddress(&IvData);
            if (doreplace)
                //
                om_replaceIvCB(object, IvData, Text->getTextLength());
            else
                //
                om_addIvCB(object, NULL, IvData, Text->getTextLength());
        }
        else
        {
            if (strcmp(gtype, "SETELE") == 0)
            { // We've got a set !
                if (doreplace) // Check if we have to replace the set. In this
                { // case we delete the current set and build a new one.
                    receiveDeleteObjectMessage(object);
                    doreplace = 0;
                    addgeometry(object, doreplace, is_timestep, timestep, NULL, data_obj, NULL, NULL, NULL, NULL);
                }
                else
                    addgeometry(object, doreplace, is_timestep, timestep, NULL, data_obj, NULL, NULL, NULL, NULL);
            }
            else
                addgeometry(object, doreplace, is_timestep, timestep, NULL, data_obj, NULL, NULL, NULL, NULL);
        }

        handleAttributes(object, data_obj);
    }
}

void InvCommunication::handleAttributes(const char *name, const coDistributedObject *obj)
{
    if (const char *text = obj->getAttribute("TEXT"))
    {
        const char *font = obj->getAttribute("TEXT_FONT");
        const char *size = obj->getAttribute("TEXT_SIZE");
        float points = size ? atof(size) : 20.;
        if (coviseViewer && coviseViewer->getTextManager())
            coviseViewer->getTextManager()->setAnnotation(name, text, points, font);
    }
}

//======================================================================
// create a color data object for a named color
//======================================================================
FILE *fp = NULL;
int isopen = 0;

static uint32_t
create_named_color(const char *cname)
{
    int r = 255, g = 255, b = 255;
    uint32_t rgba;
    char line[80];
    char *tp, *token[15];
    unsigned char *chptr;
    int count;
    const int tmax = 15;
    char color[80];
    char color_name[80];

    // first check if we get the name of the color or the RGB values
    const char *first_blank = strchr(cname, ' ');
    const char *last_blank = strrchr(cname, ' ');

    if (first_blank && last_blank && (first_blank - cname <= 3)
        && (last_blank - first_blank > 0) && (last_blank - first_blank <= 4))
    {
        int ret = sscanf(cname, "%d %d %d", &r, &g, &b);
        if (ret != 3)
        {
            cerr << "create_named_color: sscanf failed" << endl;
        }
    }

    else
    {
        while (*cname == ' ' || *cname == '\t')
            cname++;
        strncpy(color_name, cname, 80);
        char *tmpptr = &color_name[strlen(color_name)];
        tmpptr--;
        while (*tmpptr == '\0' || *tmpptr == ' ' || *tmpptr == '\n' || *tmpptr == '\t')
        {
            *tmpptr = '\0';
            tmpptr--;
        }

        //   cerr << "in create_named_color: *" << cname << "* ------------------------------" << endl;
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            if (line[0] == '!')
                continue;
            count = 0;
            tp = strtok(line, " \t");
            for (count = 0; count < tmax && tp != NULL;)
            {
                token[count] = tp;
                tp = strtok(NULL, " \t");
                count++;
            }
            token[count] = NULL;
            strcpy(color, token[3]);
            if (count == 5)
            {
                //  cerr << "concatenating " << color << " and " << token[4] << "\n";
                strcat(color, " ");
                strcat(color, token[4]);
            }
            if (count == 6)
            {
                //cerr << "concatenating " << color << ", " << token[4] << " and " <<	token[5] << "\n";
                strcat(color, " ");
                strcat(color, token[4]);
                strcat(color, " ");
                strcat(color, token[5]);
            }
            if (count == 7)
            {
                //cerr << "concatenating " << color << ", " << token[4] << ", " << token[6] << " and " <<	token[5] << "\n";
                strcat(color, " ");
                strcat(color, token[4]);
                strcat(color, " ");
                strcat(color, token[5]);
                strcat(color, " ");
                strcat(color, token[6]);
            }
            tmpptr = color;
            while (*tmpptr != '\0' && *tmpptr != '\n')
                tmpptr++;
            *tmpptr = '\0';
            //       cerr << "count: " << count << " token[3]: " << color << endl;
            //       cerr << "comparing *" << color << "* with *" << color_name << "*\n";
            if (strcmp(color, color_name) == 0)
            {
                //	  cerr << "found it!!! ***********************" << endl;
                r = atoi(token[0]);
                g = atoi(token[1]);
                b = atoi(token[2]);
                fseek(fp, 0L, SEEK_SET);
                break;
            }
        }
    }
    fseek(fp, 0L, SEEK_SET);

    /*
   #ifdef BYTESWAP
      chptr      = (unsigned char *)&rgba;
      *chptr     = (unsigned char)(255); chptr++;
      *(chptr)   = (unsigned char)(b); chptr++;
      *(chptr)   = (unsigned char)(g); chptr++;
      *(chptr)   = (unsigned char)(r);               // no transparency
   #else
   */
    chptr = (unsigned char *)&rgba;
    *chptr = (unsigned char)(r);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(255); // no transparency
    //#endif

    return rgba;
}

//==========================================================================
// receive a add object  message
//==========================================================================
void
InvCommunication::addgeometry(char *object, int doreplace, int is_timestep, int timestep,
                              char *root,
                              const coDistributedObject *geometry,
                              const coDistributedObject *normals,
                              const coDistributedObject *colors,
                              const coDistributedObject *texture,
                              const coDoGeometry *container,
                              char *feedbackInStr)
{
    const coDistributedObject *const *dobjsg = NULL; // Geometry Set elements
    const coDistributedObject *const *dobjsc = NULL; // Color Set elements
    const coDistributedObject *const *dobjsn = NULL; // Normal Set elements
    const coDistributedObject *const *dobjst = NULL; // Texture Set elements
    const coDoVec3 *normal_udata = NULL;
    const coDoVec3 *color_udata = NULL;
    const coDoFloat *color_sdata = NULL;
    const coDoRGBA *color_pdata = NULL;
    const coDoTexture *tex = NULL;
    const coDoPoints *points = NULL;
    const coDoLines *lines = NULL;
    const coDoPolygons *poly = NULL;
    const coDoTriangleStrips *strip = NULL;
    const coDoUniformGrid *ugrid = NULL;
    const coDoRectilinearGrid *rgrid = NULL;
    const coDoStructuredGrid *sgrid = NULL;
    const coDoUnstructuredGrid *unsgrid = NULL;
    const coDoSet *set = NULL;
    const coDoPixelImage *img = NULL;

    // number of elements per geometry,color and normal set
    int no_elems = 0, no_c = 0, no_n = 0, no_t = 0;
    int normalbinding((int)INV_NONE);
    int colorbinding((int)INV_NONE);
    int colorpacking = INV_NONE;
    int vertexOrder = 0;
    int no_poly = 0;
    int no_strip = 0;
    int no_vert = 0;
    int no_points = 0;
    int no_lines = 0;
    int curset;

    bool delete_pc = false;
    int texW = 0, texH = 0; // texture width and height
    unsigned char *texImage = NULL; // texture map
    int pixS = 0; // size of pixels in texture map (= number of bytes per pixel)

    int *v_l = NULL, *l_l = NULL, *el = NULL, *vl = NULL;
    int xsize, ysize, zsize;
    int minTimeStep, maxTimeStep;
    float xmax = FLT_MIN, xmin = FLT_MAX,
          ymax = FLT_MIN, ymin = FLT_MAX,
          zmax = FLT_MIN, zmin = FLT_MAX;
    float *rc = NULL, *gc = NULL, *bc = NULL, *xn = NULL, *yn = NULL, *zn = NULL;
    uint32_t *pc = NULL;
    unsigned char *byteData = NULL;
    float *x_c = NULL, *y_c = NULL, *z_c = NULL;
    float **t_c = NULL;
    float transparency = 0.0;

    // part handling
    int partID;
    SoGroup *objNode = NULL; //  group node belonging to added object

    const char *gtype, *ntype, *ctype, *ttype;
    const char *vertexOrderStr, *transparencyStr;
    const char *bindingType;
    char *objName;
    char buf[300];
    const char *tstep_attrib = 0L;
    const char *colormap_attrib = 0L;
    const char *part_attrib = 0L;
    char *rName = NULL;

    char *feedbackStr = NULL;

    is_timestep = 0;
    int i;
    uint32_t rgba;
    coMaterial *material = NULL;
    curset = anzset;

    char *interactorStr = NULL;

    handleAttributes(object, geometry);

    gtype = geometry->getType();

    //////////////////////////////////////////////////////////////////////////////
    /// Added 'geometry' is a geometry container object

    if (strcmp(gtype, "GEOMET") == 0)
    {
        container = (coDoGeometry *)geometry;
        if (container->objectOk())
        {
            // unpack parts and recursively call addgeometry
            geometry = container->getGeometry();
            normals = container->getNormals();
            colors = container->getColors();
            texture = container->getTexture();
            gtype = geometry->getType();

            addgeometry(object, doreplace, is_timestep, timestep, root, geometry, normals, colors, texture, container);
        }
        delete geometry;
    }

    //////////////////////////////////////////////////////////////////////////////
    /// Added 'geometry' is a SET

    else if (strcmp(gtype, "SETELE") == 0)
    {

        set = (coDoSet *)geometry;
        if (set != NULL)
        {
            // retrieve the whole set
            dobjsg = set->getAllElements(&no_elems);

            // look if it is a timestep series
            tstep_attrib = set->getAttribute("TIMESTEP");
            if (tstep_attrib != NULL)
            {
                int ret = sscanf(tstep_attrib, "%d %d", &minTimeStep, &maxTimeStep);
                if (ret != 2)
                {
                    cerr << "InvCommunication::addgeometry: sscanf1 failed" << endl;
                }
                minTimeStep = 1;
                maxTimeStep = no_elems;
                is_timestep = 1;
            }

            /////////////////////////////////////////////////////////////////////////
            /// If we got Normals in the object
            if (normals != NULL)
            {
                ntype = normals->getType();
                if (strcmp(ntype, "SETELE") != 0) // GEO is set, so this must be set, too...
                {
                    print_comment(__LINE__, __FILE__, "ERROR: ...did not get a normal set");
                }
                else
                {
                    set = (coDoSet *)normals; // aw: removed if (set!=NULL) ... always true
                    // Get Set
                    dobjsn = set->getAllElements(&no_n);
                    if (no_n == no_elems)
                    {
                        print_comment(__LINE__, __FILE__, "... got normal set");
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: number of normalelements does not match geometry set");
                        no_n = 0;
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////
            ///  If we got Colors in the object
            if (colors != NULL)
            {
                ctype = colors->getType();
                if (strcmp(ctype, "SETELE") != 0)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: ...did not get a color set");
                }
                else
                {
                    set = (coDoSet *)colors;
                    // look if there is a colormap inside the set
                    colormap_attrib = set->getAttribute("COLORMAP");
                    // parse the attribute...
                    if (colormap_attrib != NULL)
                        om_addColormapCB(object, colormap_attrib);

                    // Get Set
                    dobjsc = set->getAllElements(&no_c);
                    if (no_c == no_elems)
                    {
                        print_comment(__LINE__, __FILE__, "... got color set");
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: number of colorelements does not match geometry set");
                        no_c = 0;
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////
            ///  If we got Colors in the object
            if (texture != NULL)
            {
                ttype = texture->getType();
                if (strcmp(ttype, "SETELE") != 0)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: ...did not get a texture set");
                }
                else
                {
                    set = (coDoSet *)texture;

                    // look if there is a colormap inside the set
                    colormap_attrib = set->getAttribute("COLORMAP");
                    // parse the attribute...
                    if (colormap_attrib != NULL)
                        om_addColormapCB(object, colormap_attrib);
                    // Get Set
                    dobjst = set->getAllElements(&no_t);
                    if (no_t == no_elems)
                        print_comment(__LINE__, __FILE__, "... got texture set");
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: number of texture-elements does not match geometry set");
                        no_t = 0;
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////
            ////   ?????
            if (!doreplace)
            {
                setnames[curset] = new char[strlen(object) + 1];
                strcpy(setnames[curset], object);
                elemanz[curset] = no_elems;
                elemnames[curset] = new char *[no_elems];
                om_addSeparatorCB(object, root, is_timestep, minTimeStep, maxTimeStep);
                anzset++;
            }
            for (i = 0; i < no_elems; i++)
            {
                if (is_timestep)
                    timestep = i;
                const char *tmp = dobjsg[i]->getAttribute("OBJECTNAME");

                // check for interactor her too
                // and use it
                if (tmp == NULL)
                {
                    strcpy(buf, dobjsg[i]->getName());
                    objName = new char[1 + strlen(buf)]; // we may have to delete it later
                    strcpy(objName, buf);
                }
                else
                {
                    objName = new char[1 + strlen(tmp)];
                    strcpy(objName, tmp);
                }
                char *preFix = NULL;
                char *contNm = NULL;
                if (container)
                {
                    contNm = container->getName();
                    if (!strncmp(contNm, "Collect", 7) || !strncmp(contNm, "Material", 8))
                    {
                        int preFixLen = 1 + strlen(contNm) + strlen(objName);
                        preFix = new char[preFixLen];
                    }
                }
                // we add a prefix to the obj-name if we've got the output of a Collect module
                if (preFix)
                {
                    if (!strncmp(contNm, "Collect", 7))
                    {
                        strcpy(preFix, "C");
                        strcat(preFix, &contNm[7]);
                    }
                    if (!strncmp(contNm, "Material", 8))
                    {
                        strcpy(preFix, "M");
                        strcat(preFix, &contNm[8]);
                    }

                    strcat(preFix, objName);

                    delete[] objName;
                    objName = new char[1 + strlen(preFix)];
                    strcpy(objName, preFix);
                }

                if (!doreplace)
                {
                    elemnames[curset][i] = new char[strlen(objName) + 1];
                    strcpy(elemnames[curset][i], objName);
                }

                const coDistributedObject *dobjPtr = NULL;
                if (no_c > 0)
                {
                    if (no_n > 0)
                    {
                        if (no_t > 0)
                            dobjPtr = dobjst[i];
                        addgeometry(objName, doreplace, is_timestep, timestep, object, dobjsg[i], dobjsn[i], dobjsc[i], dobjPtr, container, interactorStr);
                    }
                    else
                    {
                        dobjPtr = NULL;
                        if (no_t > 0)
                            dobjPtr = dobjst[i];
                        addgeometry(objName, doreplace, is_timestep, timestep, object, dobjsg[i], NULL, dobjsc[i], dobjPtr, container, interactorStr);
                    }
                }
                else
                {
                    dobjPtr = NULL;
                    if (no_n > 0)
                    {
                        if (no_t > 0)
                            dobjPtr = dobjst[i];
                        addgeometry(objName, doreplace, is_timestep, timestep, object, dobjsg[i], dobjsn[i], NULL, dobjPtr, container, interactorStr);
                    }
                    else
                    {
                        dobjPtr = NULL;
                        if (no_t > 0)
                            dobjPtr = dobjst[i];
                        addgeometry(objName, doreplace, is_timestep, timestep, object, dobjsg[i], NULL, NULL, dobjPtr, container, interactorStr);
                    }
                }
            }
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: ...got bad geometry set");
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    /////   Single Geometry, not a set
    else // not a set
    {

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        //// Now the geometrical primitives

        if (strcmp(gtype, "POLYGN") == 0)
        {
            poly = (coDoPolygons *)geometry; // removed test (poly!=NULL)...
            rName = poly->getName();
            no_poly = poly->getNumPolygons();
            no_vert = poly->getNumVertices();
            no_points = poly->getNumPoints();
            poly->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            strip = (coDoTriangleStrips *)geometry; // removed test (strip!=NULL)...
            rName = strip->getName();
            no_strip = strip->getNumStrips();
            no_vert = strip->getNumVertices();
            no_points = strip->getNumPoints();
            strip->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            ugrid = (coDoUniformGrid *)geometry;
            ugrid->getGridSize(&xsize, &ysize, &zsize);
            ugrid->getMinMax(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
            no_points = xsize * ysize * zsize;
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            unsgrid = (coDoUnstructuredGrid *)geometry;
            unsgrid->getGridSize(&xsize, &ysize, &no_points);
            unsgrid->getAddresses(&el, &vl, &x_c, &y_c, &z_c);
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid = (coDoRectilinearGrid *)geometry;
            rgrid->getGridSize(&xsize, &ysize, &zsize);
            rgrid->getAddresses(&x_c, &y_c, &z_c);
            no_points = xsize * ysize * zsize;
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid = (coDoStructuredGrid *)geometry;
            sgrid->getGridSize(&xsize, &ysize, &zsize);
            sgrid->getAddresses(&x_c, &y_c, &z_c);
            no_points = xsize * ysize * zsize;
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            points = (coDoPoints *)geometry;
            no_points = points->getNumPoints();
            points->getAddresses(&x_c, &y_c, &z_c);
        }
        else if (strcmp(gtype, "LINES") == 0)
        {
            lines = (coDoLines *)geometry;
            rName = lines->getName();
            no_lines = lines->getNumLines();
            no_vert = lines->getNumVertices();
            no_points = lines->getNumPoints();
            lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            print_comment(__LINE__, __FILE__, "WARNING: We should never get here");
            return;
            if (delete_pc)
                delete[] pc;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: ...got unknown geometry");
            return;
            if (delete_pc)
                delete[] pc;
        }

        //
        // check for feedback attribute
        // oldstyle
        const char *tmpStr = geometry->getAttribute("FEEDBACK");
        if (tmpStr)
        {
            feedbackStr = new char[1 + strlen(tmpStr)];
            strcpy(feedbackStr, tmpStr);
            // lets see if we have simultaniously an IGNORE attribute (for CuttingSurfaces)
            tmpStr = geometry->getAttribute("IGNORE");
            if (tmpStr)
            {
                char *tmp = new char[18 + strlen(feedbackStr) + strlen(tmpStr)];
                strcpy(tmp, feedbackStr);
                strcat(tmp, "<IGNORE>");
                strcat(tmp, tmpStr);
                strcat(tmp, "<IGNORE>");
                delete[] feedbackStr;
                feedbackStr = tmp;
            }
        }
        else
        {
            // here we translate the new interactor scheme to the oldstyle
            // string which is used by InvPlaneMover
            tmpStr = geometry->getAttribute("INTERACTOR");
            if ((tmpStr == NULL) && (feedbackInStr != NULL))
                tmpStr = feedbackInStr;
            if (tmpStr)
            {
                coFeedbackParser fp(tmpStr);
                if ((fp.moduleName().find("CuttingSurface") != string::npos) || (fp.moduleName().find("CuttingSurfaceComp") != string::npos))
                {
                    float x, y, z;
                    if (fp.getFloatVector("vertex", x, y, z))
                    {
                        ostringstream stream;
                        stream << "C" << fp.moduleName();
                        stream << "\n" << fp.moduleInstance();
                        stream << "\n" << fp.moduleHost();
                        stream << "\n<IGNORE> " << x << " ";
                        stream << y << " ";
                        stream << z << " 0.0001 1 <IGNORE>";
                        if (feedbackStr)
                            delete[] feedbackStr;
                        feedbackStr = new char[stream.str().size() + 1];
                        strcpy(feedbackStr, stream.str().c_str());
                    }
                }
            }
        }

        ////////////////////////////////////
        // check for vertexOrder attribute
        vertexOrderStr = geometry->getAttribute("vertexOrder");
        if (vertexOrderStr == NULL)
            vertexOrder = 2;
        else
            vertexOrder = vertexOrderStr[0] - '0';

        ////////////////////////////////////
        // check for Transparency
        transparencyStr = geometry->getAttribute("TRANSPARENCY");
        transparency = 0.0;
        if (transparencyStr != NULL)
        {
            // sk 21.06.2001
            //	      fprintf(stderr,"Transparency: \"%s\"\n",transparencyStr);
            if ((transparency = atof(transparencyStr)) < 0.0)
                transparency = 0.0;
            if (transparency > 1.0)
                transparency = 1.0;
        }

        ////////////////////////////////////
        /// check for Material
        const char *materialStr = geometry->getAttribute("MATERIAL");
        if (materialStr != NULL)
        {
            if (strncmp(materialStr, "MAT:", 4) == 0)
            {
                char dummy[10];
                char material_name[256];
                float ambientColor[3];
                float diffuseColor[3];
                float specularColor[3];
                float emissiveColor[3];
                float shininess;
                float transparency;

                int ret = sscanf(materialStr, "%s%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f",
                                 dummy, material_name,
                                 &ambientColor[0], &ambientColor[1], &ambientColor[2],
                                 &diffuseColor[0], &diffuseColor[1], &diffuseColor[2],
                                 &specularColor[0], &specularColor[1], &specularColor[2],
                                 &emissiveColor[0], &emissiveColor[1], &emissiveColor[2],
                                 &shininess, &transparency);
                if (ret != 16)
                {
                    cerr << "InvCommunication::addgeometry: sscanf2 failed" << endl;
                }

                const char *mat_colorStr = geometry->getAttribute("MAT_COLOR");
                if (mat_colorStr != NULL)
                { // change base color of material from white to the given color

                    float r, g, b;
                    int ri, gi, bi;
                    int ret = sscanf(mat_colorStr, "%d %d %d", &ri, &gi, &bi);
                    if (ret != 3)
                    {
                        cerr << "InvCommunication::addgeometry: sscanf3 failed" << endl;
                    }
                    r = (float)ri / 255.;
                    g = (float)gi / 255.;
                    b = (float)bi / 255.;

                    diffuseColor[0] = r * diffuseColor[0];
                    diffuseColor[1] = g * diffuseColor[1];
                    diffuseColor[2] = b * diffuseColor[2];
                    specularColor[0] = r * specularColor[0];
                    specularColor[1] = g * specularColor[1];
                    specularColor[2] = b * specularColor[2];
                    ambientColor[0] = r * ambientColor[0];
                    ambientColor[1] = g * ambientColor[1];
                    ambientColor[2] = b * ambientColor[2];
                }
                material = new coMaterial(material_name, ambientColor, diffuseColor, specularColor, emissiveColor, shininess, transparency);
                colorbinding = INV_OVERALL;
                colorpacking = INV_RGBA;
            }
            else
            {
                material = materialList.get(materialStr);
                if (!material)
                {
                    char category[500];
                    int ret = sscanf(materialStr, "%s", category);
                    if (ret != 1)
                    {
                        cerr << "InvCommunication::addgeometry: sscanf4 failed" << endl;
                    }
                    materialList.add(category);
                    material = materialList.get(materialStr);
                    // 		    if(!material) {
                    // 			// sk 21.06.2001
                    // 			//			         fprintf(stderr,"Material %s not found!\n",materialStr);
                    // 		    }
                }
            }
            // take transperancy of Material if no TRANSPARENCY attribute was found
            if (transparencyStr == NULL && material != NULL)
            {
                transparency = material->transparency;
            }
        }

        ////////////////////////////////////
        // check for Part-ID
        part_attrib = geometry->getAttribute("PART");
        if (part_attrib != NULL)
        {

            // partID = atoi(part_attrib);
            if (partNames.isElement(part_attrib))
            {
                partID = partNames[part_attrib];
            }
            else
            {
                partID = getNextPK();
                partNames.insert(partID, part_attrib);
            }
            //cerr << "PART-ID: " << partID << endl;
            if (partID < 0)
                partID = -1;
        }
        else
            partID = -1;

        ////////////////////////////////////
        /// Normals
        //if (normalbinding!=INV_NONE)
        if (normals)
        {
            ntype = normals->getType();

            // Normals in coDoVec3
            if (strcmp(ntype, "USTVDT") == 0)
            {
                normal_udata = (coDoVec3 *)normals;
                no_n = normal_udata->getNumPoints();
                normal_udata->getAddresses(&xn, &yn, &zn);
            }

            // Normals in unknown format
            else
                no_n = 0;

            /// now get this attribute junk done
            if (no_n == no_points || no_n == no_vert)
                normalbinding = INV_PER_VERTEX;
            else if (no_n > 1 && (no_poly == 0 || no_poly == no_n))
                normalbinding = INV_PER_FACE;
            else if (no_n >= 1)
            {
                normalbinding = INV_OVERALL;
                no_n = 1;
            }
            else
                normalbinding = INV_NONE;
        }
        else
        {
            normalbinding = INV_NONE;
            no_n = 0;
        }

        ////////////////////////////////////
        /// Colors / Textures

        if (texture) /// Colors via 'texture' coloring
        {

            colorbinding = INV_PER_VERTEX; //INV_TEXTURE
            colorpacking = INV_TEXTURE;
            colormap_attrib = texture->getAttribute("COLORMAP");

            ttype = texture->getType();
            if (strcmp(ttype, "TEXTUR") == 0)
            {
                tex = (coDoTexture *)texture;
                img = tex->getBuffer();
                texImage = (unsigned char *)(img->getPixels());
                texW = img->getWidth();
                texH = img->getHeight();
                pixS = img->getPixelsize();
                no_t = tex->getNumCoordinates();
                t_c = tex->getCoordinates();
            }
            else
            {
                cerr << "Incorrect data type in data object 'texture'." << endl;
                colorpacking = INV_NONE;
                colorbinding = INV_NONE;
            }
        }

        else if (colors) /// Colors via 'normal' coloring
        {

            ctype = colors->getType();
            if (const coDoByte *byte = dynamic_cast<const coDoByte *>(colors))
            {
                no_c = byte->getNumPoints();
                byteData = byte->getAddress();
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "USTSDT") == 0)
            {
                color_sdata = (coDoFloat *)colors;
                no_c = color_sdata->getNumPoints();
                color_sdata->getAddress(&rc);
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "USTVDT") == 0)
            {
                color_udata = (coDoVec3 *)colors;
                no_c = color_udata->getNumPoints();
                color_udata->getAddresses(&rc, &gc, &bc);
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "RGBADT") == 0)
            {
                color_pdata = (coDoRGBA *)colors;
                no_c = color_pdata->getNumPoints();
                uint32_t *pc_shared = NULL;
                color_pdata->getAddress((int **)(void *)&pc_shared);
                pc = new uint32_t[no_c];
                delete_pc = true;
#ifdef BYTESWAP
                for (int i = 0; i < no_c; i++)
                {
                    pc[i] = ((pc_shared[i] & 0xff) << 24)
                            | ((pc_shared[i] & 0xff00) << 8)
                            | ((pc_shared[i] & 0xff0000) >> 8)
                            | ((pc_shared[i] & 0xff000000) >> 24);
                }
#else
                if (delete_pc)
                {
                    delete[] pc;
                    pc = NULL;
                }
                color_pdata->getAddress((int **)&pc);
#endif
                colorpacking = INV_RGBA;
            }
            else
            {
                colorbinding = INV_NONE;
                colorpacking = INV_NONE;
                print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not jet implemented");
                //   sendError("ERROR: DataTypes other than structured and unstructured are not jet implemented");
            }

            colormap_attrib = colors->getAttribute("COLORMAP");

            /// now get this attribute junk done
            if (no_c == no_points)
                colorbinding = INV_PER_VERTEX;
            else if (no_c > 1 && (no_poly == 0 || no_poly == no_c))
                colorbinding = INV_PER_FACE;
            else if (no_c >= 1)
                colorbinding = INV_OVERALL;
            else
                colorbinding = INV_NONE;
        }

        // Colors not given per object, try to take from attribute
        if ((no_c == 0) && (no_t == 0))
        {

            bindingType = geometry->getAttribute("COLOR");
            if (bindingType != NULL)
            {
                colorbinding = INV_OVERALL;
                colorpacking = INV_RGBA;
                no_c = 1;

                if (*bindingType == '#')
                {
                    bindingType++;
                    int r, g, b, a;
                    sscanf(bindingType, "%02x%02x%02x%02x", &r, &g, &b, &a);

                    unsigned char *chptr = (unsigned char *)&rgba;
                    *chptr = (unsigned char)(r);
                    chptr++;
                    *(chptr) = (unsigned char)(g);
                    chptr++;
                    *(chptr) = (unsigned char)(b);
                    chptr++;
                    *(chptr) = (unsigned char)(a); // no transparency
                    if (delete_pc)
                    {
                        delete[] pc;
                        pc = NULL;
                    }
                    delete_pc = false;
                    pc = &rgba;
                }

                else
                {
                    // open ascii file for color names
                    if (!isopen)
                    {
                        fp = CoviseRender::fopen("share/covise/rgb.txt", "r");
                        if (fp != NULL)
                            isopen = TRUE;
                    }
                    if (isopen)
                    {
                        rgba = create_named_color(bindingType);
                        if (delete_pc)
                        {
                            delete[] pc;
                            pc = NULL;
                        }
                        delete_pc = false;
                        pc = &rgba;
                    }
                }
            }
            else
            {
                // we: do NOT set here, INV_NONE by default and might be
                // colorpacking = INV_NONE;
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

        //
        // add object to inventor scenegraph depending on type
        //
        if (doreplace)
        {
            if (strcmp(gtype, "UNIGRD") == 0)
                objNode = om_replaceUGridCB(object, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                                            no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                            no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "RCTGRD") == 0)
                objNode = om_replaceRGridCB(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                            no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                            no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "STRGRD") == 0)
                objNode = om_replaceSGridCB(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                            no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                            no_n, normalbinding, xn, yn, zn, transparency);

            // texture coordinates?
            if (no_t)
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = om_replacePolygonCB(object, no_poly, no_vert,
                                                  no_points, x_c, y_c, z_c,
                                                  v_l, l_l,
                                                  no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                  no_n, normalbinding, xn, yn, zn, transparency,
                                                  vertexOrder,
                                                  texW, texH, pixS, texImage,
                                                  no_t, t_c[0], t_c[1], material);

                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = om_replaceTriangleStripCB(object, no_strip, no_vert,
                                                        no_points, x_c, y_c, z_c,
                                                        v_l, l_l,
                                                        no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                        no_n, normalbinding, xn, yn, zn, transparency,
                                                        vertexOrder,
                                                        texW, texH, pixS, texImage,
                                                        no_t, t_c[0], t_c[1], material);
            }
            else
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = om_replacePolygonCB(object, no_poly, no_vert,
                                                  no_points, x_c, y_c, z_c,
                                                  v_l, l_l,
                                                  no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                  no_n, normalbinding, xn, yn, zn, transparency,
                                                  vertexOrder,
                                                  texW, texH, pixS, texImage,
                                                  no_t, NULL, NULL, material);

                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = om_replaceTriangleStripCB(object, no_strip, no_vert,
                                                        no_points, x_c, y_c, z_c,
                                                        v_l, l_l,
                                                        no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                        no_n, normalbinding, xn, yn, zn, transparency,
                                                        vertexOrder,
                                                        texW, texH, pixS, texImage,
                                                        no_t, NULL, NULL, material);
            }

            if (strcmp(gtype, "LINES") == 0)
                objNode = om_replaceLineCB(object, no_lines, no_vert,
                                           no_points, x_c, y_c, z_c,
                                           v_l, l_l,
                                           no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                           no_n, normalbinding, xn, yn, zn);

            if (no_points && ((strcmp(gtype, "POINTS") == 0) || (strcmp(gtype, "UNSGRD") == 0)))
                objNode = om_replacePointCB(object, no_points,
                                            x_c, y_c, z_c,
                                            colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc));

            if (colormap_attrib != NULL)
            {

                // cerr << "Replacing non set colormap (calling om_addColormapCB)" << endl;
                om_replaceColormapCB(object, colormap_attrib);
            }

            if (partID > -1 && objNode != NULL)
                om_replacePartCB(object, partID, (SoSwitch *)objNode);

            if (timestep > -1 && partID > -1 && objNode != NULL)
                om_replaceTimePartCB(object, timestep, partID, (SoSwitch *)objNode);
        }

        else // if not (doreplace)
        {

            if (strcmp(gtype, "UNIGRD") == 0)
                objNode = om_addUGridCB(object, root, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                                        no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc), byteData,
                                        no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "RCTGRD") == 0)
                objNode = om_addRGridCB(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                                        no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                        no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "STRGRD") == 0)
                objNode = om_addSGridCB(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                                        no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                        no_n, normalbinding, xn, yn, zn, transparency);

            // texture coordinates ?
            if (t_c)
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = om_addPolygonCB(object, root, no_poly, no_vert,
                                              no_points, x_c, y_c, z_c,
                                              v_l, l_l,
                                              no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                              no_n, normalbinding, xn, yn, zn, transparency,
                                              vertexOrder,
                                              texW, texH, pixS, texImage,
                                              no_t, t_c[0], t_c[1], material,
                                              rName, feedbackStr);
                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = om_addTriangleStripCB(object, root, no_strip, no_vert,
                                                    no_points, x_c, y_c, z_c,
                                                    v_l, l_l,
                                                    no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                    no_n, normalbinding, xn, yn, zn, transparency,
                                                    vertexOrder,
                                                    texW, texH, pixS, texImage,
                                                    no_t, t_c[0], t_c[1], material,
                                                    rName, feedbackStr);
            }
            else
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = om_addPolygonCB(object, root, no_poly, no_vert,
                                              no_points, x_c, y_c, z_c,
                                              v_l, l_l,
                                              no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                              no_n, normalbinding, xn, yn, zn, transparency,
                                              vertexOrder,
                                              texW, texH, pixS, texImage,
                                              no_t, NULL, NULL, material,
                                              rName, feedbackStr);
                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = om_addTriangleStripCB(object, root, no_strip, no_vert,
                                                    no_points, x_c, y_c, z_c,
                                                    v_l, l_l,
                                                    no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                                    no_n, normalbinding, xn, yn, zn, transparency,
                                                    vertexOrder,
                                                    texW, texH, pixS, texImage,
                                                    no_t, NULL, NULL, material,
                                                    rName, feedbackStr);
            }

            if (strcmp(gtype, "LINES") == 0)
                objNode = om_addLineCB(object, root, no_lines, no_vert,
                                       no_points, x_c, y_c, z_c,
                                       v_l, l_l,
                                       no_c, colorbinding, colorpacking, rc, gc, bc, (uint32_t *)(pc),
                                       no_n, normalbinding, xn, yn, zn, material,
                                       rName, feedbackStr);
            if ( // no_points &&     AW: we DO accept 0-sized objects now
                ((strcmp(gtype, "POINTS") == 0) || (strcmp(gtype, "UNSGRD") == 0)))
            {
                const char *ptSizeStr = geometry->getAttribute("POINTSIZE");
                float ptSize = ptSizeStr ? atof(ptSizeStr) : 0.0;

                objNode = om_addPointCB(object, root, no_points,
                                        x_c, y_c, z_c,
                                        no_c, colorbinding, colorpacking,
                                        rc, gc, bc, (uint32_t *)(pc), ptSize);
            }
            if (colormap_attrib != NULL)
            {
                // cerr << "Adding non set colormap (calling om_addColormapCB)" << endl;
                om_addColormapCB(object, colormap_attrib);
            }

            if (partID > -1 && objNode != NULL)
                om_addPartCB(object, partID, (SoSwitch *)objNode);

            if (timestep > -1 && partID > -1 && objNode != NULL)
                om_addTimePartCB(object, timestep, partID, (SoSwitch *)objNode);
        }
    }
    if (delete_pc)
        delete[] pc;
}

//==========================================================================
// receive a object delete message
//==========================================================================
void InvCommunication::receiveDeleteObjectMessage(char *object)
{
    //cerr << "InvCommunication::receiveDeleteObjectMessage("
    //     << object << ")" << endl;
    if (NULL == object)
    {
        return;
    }
    char buffer[512];

    // AW: first look for replacements
    Repl *repl = replace_->next; // 1st is dummy
    Repl *orep = replace_;
    while (repl && strcmp(repl->newName, object) != 0)
    {
        orep = repl;
        repl = repl->next;
    }
    if (repl)
    {
        //cerr << "Using old " << repl->oldName << " instead " << object << endl;
        strcpy(buffer, repl->oldName);
        delete[] repl -> oldName;
        delete[] repl -> newName;
        orep->next = repl->next;
        delete repl;
        object = buffer;
    }

    //cerr << "\n\nReplace list:" << endl;
    //repl = replace->next;
    //while (repl)
    //{
    //   cerr << "   " << repl->oldName << "  -->  " << repl->newName << endl;
    //   repl = repl->next;
    //}
    //cerr << "\n" << endl;

    int i, n;
    for (i = 0; i < anzset; i++)
    {
        if (strcmp(setnames[i], object) == 0)
        {
            for (n = 0; n < elemanz[i]; n++)
            {
                receiveDeleteObjectMessage(elemnames[i][n]);
                delete[] elemnames[i][n];
            }
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
        }
    }
    om_deleteObjectCB(object);
    om_deleteColormapCB(object);
    om_deletePartCB(object);
    om_deleteTimePartCB(object);
    if (coviseViewer && coviseViewer->getTextManager())
        coviseViewer->getTextManager()->removeAnnotation(object);
}

//==========================================================================
// receive delete all message
//==========================================================================
void InvCommunication::receiveDeleteAll(void)
{
    //cerr << "InvCommunication::receiveDeleteObjectMessage("
    //     << object << ")" << endl;
    char *object;

    int i, n;
    for (i = 0; i < anzset; i++)
    {
        object = setnames[i];
        for (n = 0; n < elemanz[i]; n++)
        {
            receiveDeleteObjectMessage(elemnames[i][n]);
            delete[] elemnames[i][n];
        }
        delete[] elemnames[i];
        om_deleteObjectCB(object);
        om_deleteColormapCB(object);
        om_deletePartCB(object);
        om_deleteTimePartCB(object);
    }
}

// AW: save replaced-by-NULL object ne=ames

void InvCommunication::setReplace(char *oldName, char *newName)
{
    Repl *repl = replace_->next; // 1st is dummy
    Repl *orep = replace_;
    while (repl && strcmp(repl->newName, oldName) != 0)
    {
        orep = repl;
        repl = repl->next;
    }
    if (repl) // replace more than the first time
    {
        delete[] repl -> newName;
        repl->newName = strcpy(new char[strlen(newName) + 1], newName);
    }
    else
        orep->next = new Repl(oldName, newName);

    //cerr << "\n\nReplace list:" << endl;
    //repl = replace->next;
    //while (repl)
    //{
    //   cerr << "   " << repl->oldName << "  -->  " << repl->newName << endl;
    //   repl = repl->next;
    //}
    //cerr << "\n" << endl;
}

//==========================================================================
// receive a object delete message
//==========================================================================
void InvCommunication::receiveCameraMessage(char *message)
{
    rm_receiveCamera(message);
}

//==========================================================================
// receive a transform message
//==========================================================================
void InvCommunication::receiveTransformMessage(char *message)
{
    rm_receiveTransformation(message);
}

//==========================================================================
// receive a telepointer message
//==========================================================================
void InvCommunication::receiveTelePointerMessage(char *message)
{
    rm_receiveTelePointer(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunication::receiveDrawstyleMessage(char *message)
{
    rm_receiveDrawstyle(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunication::receiveLightModeMessage(char *message)
{
    rm_receiveLightMode(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunication::receiveSelectionMessage(char *message)
{
    rm_receiveSelection(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunication::receiveDeselectionMessage(char *message)
{
    rm_receiveDeselection(message);
}

//==========================================================================
// receive a part switching message
//==========================================================================
void InvCommunication::receivePartMessage(char *message)
{
    rm_receivePart(message);
}

//==========================================================================
// receive a reference part message
//==========================================================================
void InvCommunication::receiveReferencePartMessage(char *message)
{
    rm_receiveReferencePart(message);
}

//==========================================================================
// receive a reset scene message
//==========================================================================
void InvCommunication::receiveResetSceneMessage()
{
    rm_receiveResetScene();
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunication::receiveTransparencyMessage(char *message)
{
    rm_receiveTransparency(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveSyncModeMessage(char *message)
{
    rm_receiveSyncMode(message);
}

//==========================================================================
// receive a fog mode message
//==========================================================================
void InvCommunication::receiveFogMessage(char *message)
{
    rm_receiveFog(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveAntialiasingMessage(char *message)
{
    rm_receiveAntialiasing(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveBackcolorMessage(char *message)
{
    rm_receiveBackcolor(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveAxisMessage(char *message)
{
    rm_receiveAxis(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveClippingPlaneMessage(char *message)
{
    rm_receiveClippingPlane(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveViewingMessage(char *message)
{
    rm_receiveViewing(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunication::receiveProjectionMessage(char *message)
{
    rm_receiveProjection(message);
}

//==========================================================================
// receive a decoration mode message
//==========================================================================
void InvCommunication::receiveDecorationMessage(char *message)
{
    rm_receiveDecoration(message);
}

//==========================================================================
// receive a headlight mode message
//==========================================================================
void InvCommunication::receiveHeadlightMessage(char *message)
{
    rm_receiveHeadlight(message);
}

//==========================================================================
// receive a sequencer message
//==========================================================================
void InvCommunication::receiveSequencerMessage(char *message)
{
    rm_receiveSequencer(message);
}

//==========================================================================
// receive a colormap message
//==========================================================================
void InvCommunication::receiveColormapMessage(char *message)
{
    rm_receiveColormap(message);
}

//==========================================================================
// receive a master message
//==========================================================================
void InvCommunication::receiveMasterMessage(char *mod, char *inst, char *host)
{
    if (mod != NULL)
        strcpy(m_name, mod);
    if (inst != NULL)
        strcpy(inst_no, inst);
    if (host != NULL)
        strcpy(h_name, host);
    rm_switchMaster();
}

//==========================================================================
// receive an update message
//==========================================================================
void InvCommunication::receiveUpdateMessage(char *mod, char *inst, char *host)
{
    (void)mod;
    (void)inst;
    (void)host;
    rm_updateSlave();
}

//==========================================================================
// receive a slave message
//==========================================================================
void InvCommunication::receiveSlaveMessage(char *mod, char *inst, char *host)
{
    if (mod != NULL)
        strcpy(m_name, mod);
    if (inst != NULL)
        strcpy(inst_no, inst);
    if (host != NULL)
        strcpy(h_name, host);
    rm_switchSlave();
}

//==========================================================================
// receive a slave message
//==========================================================================
void InvCommunication::receiveMasterSlaveMessage(char *mod, char *inst, char *host)
{
    if (mod != NULL)
        strcpy(m_name, mod);
    if (inst != NULL)
        strcpy(inst_no, inst);
    if (host != NULL)
        strcpy(h_name, host);
    rm_switchMasterSlave();
}
