/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QString>
#include <QStringList>

//
// debug stuff (local use)
//
#include <covise/covise.h>

#include <util/coStringTable.h>
covise::coStringTable partNames;

//
// ec stuff
//
#include <covise/covise_process.h>
#include <covise/covise_appproc.h>
#include <covise/covise_msg.h>
#include <do/coDoData.h>
#include <do/coDoText.h>
#include <do/coDoTexture.h>
#include <do/coDoPixelImage.h>
#include <do/coDoPoints.h>
#include <do/coDoSpheres.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoSet.h>

//
//  class definition
//
#include "InvDefs.h"
#include "InvError.h"
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif
#include "InvMsgManager.h"
#ifndef YAC
#include "InvObjectManager.h"
#else
#include "InvObjectManager_yac.h"
#endif
#include "InvTelePointer.h"
#include "SoBillboard.h"
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/nodes/SoScale.h>

#include <util/coMaterial.h>
#include <util/string_util.h>

#include "InvCommunicator.h"
//
// C stuff
//
#ifndef _WIN32
#include <unistd.h>
#endif

using namespace covise;
//
// coDoSet handling
//
coMaterialList materialList("metal");

//
// defines
//
#define MAXDATALEN 500
#define MAXTOKENS 25
#define MAXHOSTLEN 20
#define MAXSETS 8000

static int anzset = 0;
static char *setnames[MAXSETS];
static int elemanz[MAXSETS];
static char **elemnames[MAXSETS];

static QString m_name;
static QString inst_no;
static QString h_name;

//#########################################################################
// InvCommunicator
//#########################################################################

struct Repl
{
    QString oldName;
    QString newName;
    struct Repl *next;
    Repl(QString oldN, QString newN)
    {
        oldName = oldN;
        newName = newN;
        next = NULL;
    }
};

static Repl dummy("", "");
static Repl *replace_ = &dummy;

//=========================================================================
// constructor
//=========================================================================
InvCommunicator::InvCommunicator()
{
    int i;
    pk_ = 1000;
    for (i = 0; i < MAXSETS; i++)
    {
        setnames[i] = NULL;
        elemanz[i] = 0;
        elemnames[i] = NULL;
    }
}

int InvCommunicator::getNextPK()
{
    return pk_++;
}

int InvCommunicator::getCurrentPK()
{
    return pk_;
}

//==========================================================================
// send a message
//==========================================================================
void InvCommunicator::sendMSG(QString message)
{
    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{message.toLatin1().data() ,  message.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);
}

//==========================================================================
// send a camera message
//==========================================================================
void InvCommunicator::sendCameraMessage(const char *message)
{
    buffer = "CAMERA";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a VRMLcamera message
//==========================================================================
void InvCommunicator::sendVRMLCameraMessage(QString message)
{
    buffer = "VRMLCAMERA";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a transform message
//==========================================================================
void InvCommunicator::sendTransformMessage(QString message)
{
    buffer = "TRANSFORM";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a telepointer message
//==========================================================================
void InvCommunicator::sendTelePointerMessage(const char *message)
{
    buffer = "TELEPOINTER";
    buffer.append("\n");
    buffer.append(message);
    ;
    sendMSG(buffer);
}

//==========================================================================
// send a VRML telepointer message
//==========================================================================
void InvCommunicator::sendVRMLTelePointerMessage(QString message)
{
    buffer = "VRML_TELEPOINTER";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a drawstyle message
//==========================================================================
void InvCommunicator::sendDrawstyleMessage(QString message)
{
    buffer = "DRAWSTYLE";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunicator::sendLightModeMessage(QString message)
{
    buffer = "DRAWSTYLE";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a colormap message
//==========================================================================
void InvCommunicator::sendColormapMessage(QString message)
{
    buffer = "COLORMAP";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunicator::sendSelectionMessage(QString message)
{
    buffer = "SELECTION";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

void InvCommunicator::sendCSFeedback(const char *key, QString message)
{
    QString dataBuf(key);
    dataBuf += "\n";
    dataBuf += message;
    dataBuf += "\n";

    Message msg{ COVISE_MESSAGE_UI , DataHandle{dataBuf.toLatin1().data(),  dataBuf.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);
}

void InvCommunicator::sendAnnotation(const char *key, QString message)
{
    QString dataBuf(key);
    dataBuf += "\n";
    dataBuf += message;
    dataBuf += "\n";

    Message msg{ COVISE_MESSAGE_RENDER , DataHandle{dataBuf.toLatin1().data(), dataBuf.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunicator::sendDeselectionMessage(QString message)
{
    buffer = "DESELECTION";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a part switching message
//==========================================================================
void InvCommunicator::sendPartMessage(QString message)
{
    buffer = "PART";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a reference part message
//==========================================================================
void InvCommunicator::sendReferencePartMessage(QString message)
{
    buffer = "REFPART";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a reset scene  message
//==========================================================================
void InvCommunicator::sendResetSceneMessage()
{
    buffer = "RESET";
    buffer.append("\n");
    sendMSG(buffer);
}

//==========================================================================
// send a light mode message
//==========================================================================
void InvCommunicator::sendTransparencyMessage(QString message)
{
    buffer = "TRANSPARENCY";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunicator::sendSyncModeMessage(QString message)
{
    buffer = "SYNC";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a fog mode message
//==========================================================================
void InvCommunicator::sendFogMessage(QString message)
{
    buffer = "FOG";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a aliasing mode message
//==========================================================================
void InvCommunicator::sendAntialiasingMessage(QString message)
{
    buffer = "ANTIALIASING";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a back color message
//==========================================================================
void InvCommunicator::sendBackcolorMessage(QString message)
{
    buffer = "BACKCOLOR";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a axis mode message
//==========================================================================
void InvCommunicator::sendAxisMessage(QString message)
{
    buffer = "AXIS";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a clipping plane message
//==========================================================================
void InvCommunicator::sendClippingPlaneMessage(QString message)
{
    buffer = "CLIP";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunicator::sendViewingMessage(QString message)
{
    buffer = "VIEWING";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a sync mode message
//==========================================================================
void InvCommunicator::sendProjectionMessage(QString message)
{
    buffer = "PROJECTION";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a decoration mode message
//==========================================================================
void InvCommunicator::sendDecorationMessage(QString message)
{
    buffer = "DECORATION";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a headlight mode message
//==========================================================================
void InvCommunicator::sendHeadlightMessage(QString message)
{
    buffer = "HEADLIGHT";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);
}

//==========================================================================
// send a sequencer message
//==========================================================================
void InvCommunicator::sendSequencerMessage(QString message)
{
    buffer = "SEQUENCER";
    buffer.append("\n");
    buffer.append(message);
    sendMSG(buffer);

    receiveSequencerMessage(message);
}

//==========================================================================
// send a quit message
//==========================================================================
void InvCommunicator::sendQuitMessage()
{


    buffer = "";
    buffer.append("\n");
    Message msg{ COVISE_MESSAGE_QUIT , DataHandle{buffer.toLatin1().data(), buffer.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);
    print_comment(__LINE__, __FILE__, "sended quit message");
}

//==========================================================================
// send a quit message
//==========================================================================
void InvCommunicator::sendQuitRequestMessage()
{
    QStringList list;

    list << m_name << inst_no << h_name << "QUIT";
    buffer = list.join("\n");

    Message msg{ COVISE_MESSAGE_REQ_UI , DataHandle{buffer.toLatin1().data(), buffer.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);

    print_comment(__LINE__, __FILE__, "sended quit message");

    list.clear();
}

//==========================================================================
// send a finish message
//==========================================================================
void InvCommunicator::sendFinishMessage()
{
    buffer = "";
    buffer.append("\n");

    Message msg{ COVISE_MESSAGE_FINISHED , DataHandle{buffer.toLatin1().data() ,  buffer.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&msg);
}

//==========================================================================
// receive a add object  message
//==========================================================================
void InvCommunicator::receiveAddObjectMessage(const coDistributedObject *data_obj,
                                              const char *object, int doreplace)
{
    const coDistributedObject *dobj = NULL;
    const coDistributedObject *colors = NULL;
    const coDistributedObject *normals = NULL;
    const coDistributedObject *texture = NULL;
    const coDoGeometry *geometry = NULL;
    const coDoText *Text = NULL;

    char *IvData;
    const char *gtype;
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
        }
        else if (strcmp(gtype, "DOTEXT") == 0)
        {
            Text = (coDoText *)data_obj;
            Text->getAddress(&IvData);
            if (doreplace)
                //
                renderer->om->replaceIvCB(object, IvData, Text->getTextLength());
            else
                //
                renderer->om->addIvCB(object, NULL, IvData, Text->getTextLength());
        }
        else if (strcmp(gtype, "TEXTUR") == 0)
        {
            // texture to be used as default color map for volume rendering
            coDoTexture *tex = (coDoTexture *)data_obj;
            coDoPixelImage *img = tex->getBuffer();
            uchar *texImage = (uchar *)(img->getPixels());
            int texW = img->getWidth();
            int texH = img->getHeight();
            int pixS = img->getPixelsize();

            if (pixS == 4 && texH == 1)
            {
                renderer->viewer->setGlobalLut(texW, texImage);
            }
            else if (pixS == 3)
            {
                fprintf(stderr, "got 3 component (RGB) colormap texture -- set transparentTextures in section Colors to true in covise.config\n");
            }
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
    delete data_obj; //delete object here, not before handiling the attributes
    }
}


void InvCommunicator::handleAttributes(const char *name, const coDistributedObject *obj)
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

static int
create_named_color(const char *cname)
{
    int r = 255, g = 255, b = 255;
    int rgba;
    char line[80];
    char *tp, *token[15];
    unsigned char *chptr;
    int count;
    const int tmax = 15;
    char color[80];
    char color_name[80];

    strncpy(color_name, cname, sizeof(color_name));

    while (*cname == ' ' || *cname == '\t')
        cname++;
    strncpy(color_name, cname, sizeof(color_name) - 1);
    color_name[sizeof(color_name) - 1] = '\0';

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
    fseek(fp, 0L, SEEK_SET);

    /*
   #ifdef BYTESWAP
   chptr      = (unsigned char *)&rgba;
   *(chptr)   = (unsigned char)(255);               // no transparency
   *(chptr)   = (unsigned char)(b); chptr++;
   *(chptr)   = (unsigned char)(g); chptr++;
   *chptr     = (unsigned char)(r); chptr++;
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

#include <Inventor/nodes/SoSphere.h>

//==========================================================================
// receive a add object  message
//==========================================================================
void
InvCommunicator::addgeometry(const char *object, int doreplace, int is_timestep, int timestep, const char *root,
                             const coDistributedObject *geometry, const coDistributedObject *normals,
                             const coDistributedObject *colors, const coDistributedObject *texture,
                             const coDoGeometry *container)
{
    const coDistributedObject *const *dobjsg = NULL; // Geometry Set elements
    const coDistributedObject *const *dobjsc = NULL; // Color Set elements
    const coDistributedObject *const *dobjsn = NULL; // Normal Set elements
    const coDistributedObject *const *dobjst = NULL; // Texture Set elements
    const coDoVec3 *normal_udata = NULL;
    const coDoVec3 *color_udata = NULL;
    const coDoFloat *color_ssdata = NULL;
    const coDoRGBA *color_pdata = NULL;
    const coDoTexture *tex = NULL;
    const coDoPoints *points = NULL;
    const coDoSpheres *spheres = NULL;
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
    int iRenderMethod = 0;
    int no_lines = 0;
    int curset;

    int texW = 0, texH = 0; // texture width and height
    unsigned char *texImage = NULL; // texture map
    int pixS = 0; // size of pixels in texture map (= number of bytes per pixel)

    int *v_l = 0, *l_l = 0, *el, *vl;
    int xsize, ysize, zsize;
    int minTimeStep = 0, maxTimeStep = 0;
    float xmax = 0, xmin = 0, ymax = 0, ymin = 0, zmax = 0, zmin = 0;
    float *rc = NULL, *gc = NULL, *bc = NULL, *xn = NULL, *yn = NULL, *zn = NULL;
    uint32_t *pc = NULL;
    bool delete_pc = false;
    uchar *byteData = NULL;
    float *x_c = NULL, *y_c = NULL, *z_c = NULL;
    float *radii_c = NULL;
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
    const char *tstep_attrib = NULL;
    const char *colormap_attrib = NULL;
    const char *part_attrib = NULL;
    const char *rName = NULL;

    char *feedbackStr = NULL;

    is_timestep = 0;
    int i;
    uint32_t rgba;
    covise::coMaterial *material = NULL;
    curset = anzset;

    handleAttributes(object, geometry);

    gtype = geometry->getType();
    // save this for later, as geometry might get deleted
    const char *labelTmp = geometry->getAttribute("LABEL");
    char *labelStr = NULL;
    if (labelTmp)
    {
        labelStr = new char[strlen(labelTmp) + 1];
        strcpy(labelStr, labelTmp);
    }

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
        // das darf hier nicht geloescht werden, ich denke, es darf ueberhaupt nicht geloescht werden!
        // jetzt darf man's wohl doch wieder loeschen
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
                //int retval;
                //retval=sscanf(tstep_attrib,"%d %d",&minTimeStep,&maxTimeStep);
                //if (retval!=2)
                //{
                //  std::cerr<<"InvCommunicator::addgeometry: sscanf failed"<<std::endl;
                // Sicherheitsabfragen sind schon toll, man sollte nur aufpassen was man hier macht!!!!!return;
                //}
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
                        renderer->om->addColormap(object, colormap_attrib);

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
                        renderer->om->addColormap(object, colormap_attrib);

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
                renderer->om->addSeparatorCB(object, root, is_timestep, minTimeStep, maxTimeStep);
                anzset++;
            }
            for (i = 0; i < no_elems; i++)
            {
                if (is_timestep)
                    timestep = i;
                const char *tmp = dobjsg[i]->getAttribute("OBJECTNAME");

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
                        int preFixLen = int(1 + strlen(contNm) + strlen(objName));
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

                const coDistributedObject *c_ptr = NULL;
                const coDistributedObject *t_ptr = NULL;
                const coDistributedObject *n_ptr = NULL;

                if (no_c > 0)
                    c_ptr = dobjsc[i];
                if (no_t > 0)
                    t_ptr = dobjst[i];
                if (no_n > 0)
                    n_ptr = dobjsn[i];

                addgeometry(objName, doreplace, is_timestep, timestep, object, dobjsg[i], n_ptr, c_ptr, t_ptr, container);
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
        else if (strcmp(gtype, "SPHERE") == 0)
        {
            spheres = (coDoSpheres *)geometry;
            rName = geometry->getName();
            no_points = spheres->getNumSpheres();
            spheres->getAddresses(&x_c, &y_c, &z_c, &radii_c);
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
            if (delete_pc)
                delete[] pc;
            return;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: ...got unknown geometry");
            if (delete_pc)
                delete[] pc;
            return;
        }

        //
        // check for feedback attribute
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

                int retval;
                retval = sscanf(materialStr, "%s%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f",
                                dummy, material_name,
                                &ambientColor[0], &ambientColor[1], &ambientColor[2],
                                &diffuseColor[0], &diffuseColor[1], &diffuseColor[2],
                                &specularColor[0], &specularColor[1], &specularColor[2],
                                &emissiveColor[0], &emissiveColor[1], &emissiveColor[2],
                                &shininess, &transparency);
                if (retval != 16)
                {
                    std::cerr << "InvCommunicator::addgeometry: sscanf1 failed" << std::endl;
                    return;
                }

                material = new covise::coMaterial(material_name, ambientColor, diffuseColor, specularColor, emissiveColor, shininess, transparency);
                colorbinding = INV_OVERALL;
                colorpacking = INV_RGBA;
            }
            else
            {
                material = materialList.get(materialStr);
                if (!material)
                {
                    char category[500];
                    int retval;
                    retval = sscanf(materialStr, "%s", category);
                    if (retval != 1)
                    {
                        std::cerr << "InvCommunicator::addgeometry: sscanf2 failed" << std::endl;
                        return;
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
            if (no_n == no_vert || no_n == no_points)
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
                texImage = reinterpret_cast<unsigned char *>((img->getPixels()));
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

        if (colors) /// Colors via 'normal' coloring
        {
            ctype = colors->getType();
            if (const coDoByte *byte = dynamic_cast<const coDoByte *>(colors))
            {
                no_c = byte->getNumPoints();
                byteData = byte->getAddress();
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "USTVDT") == 0)
            {
                color_udata = (coDoVec3 *)colors;
                no_c = color_udata->getNumPoints();
                color_udata->getAddresses(&rc, &gc, &bc);
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "USTSDT") == 0)
            {
                color_ssdata = (coDoFloat *)colors;
                no_c = color_ssdata->getNumPoints();
                color_ssdata->getAddress(&rc);
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "RGBADT") == 0)
            {
                color_pdata = (coDoRGBA *)colors;
                no_c = color_pdata->getNumPoints();
                colorpacking = INV_RGBA;
#ifdef BYTESWAP
                int *pc_shared = NULL;
                color_pdata->getAddress(&pc_shared);
                if (delete_pc)
                {
                    delete[] pc;
                    pc = NULL;
                }
                pc = new uint32_t[no_c];
                delete_pc = true;

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
            }
            else
            {
                colorbinding = INV_NONE;
                colorpacking = INV_NONE;
                no_c = 0;
                print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not yet implemented");
                //   sendError("ERROR: DataTypes other than structured and unstructured are not yet implemented");
            }

            colormap_attrib = colors->getAttribute("COLORMAP");

            /// now get this attribute junk done
            if (no_c == no_points)
                colorbinding = INV_PER_VERTEX;
            else if (no_c > 1 && (no_poly == 0 || no_poly == no_c))
                colorbinding = INV_PER_FACE;
            else if (no_c >= 1)
            {
                colorbinding = INV_OVERALL;
                no_c = 1;
            }
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

                // use given color if of type #00AAFF
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

                // open ascii file for color names
                else
                {
                    if (!isopen)
                    {
                        const char *fileName = renderer->om->getname("share/covise/rgb.txt");
                        if (fileName)
                        {
                            fp = fopen(fileName, "r");
                        }
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
                objNode = renderer->om->replaceUGridCB(object, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                                                       no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                       no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "RCTGRD") == 0)
                objNode = renderer->om->replaceRGridCB(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                                       no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                       no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "STRGRD") == 0)
                objNode = renderer->om->replaceSGridCB(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                                       no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                       no_n, normalbinding, xn, yn, zn, transparency);

            // texture coordinates?
            if (no_t)
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = renderer->om->replacePolygonCB(object, no_poly, no_vert,
                                                             no_points, x_c, y_c, z_c,
                                                             v_l, l_l,
                                                             no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                             no_n, normalbinding, xn, yn, zn, transparency,
                                                             vertexOrder,
                                                             texW, texH, pixS, texImage,
                                                             no_t, t_c[0], t_c[1], material);

                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = renderer->om->replaceTriangleStripCB(object, no_strip, no_vert,
                                                                   no_points, x_c, y_c, z_c,
                                                                   v_l, l_l,
                                                                   no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                   no_n, normalbinding, xn, yn, zn, transparency,
                                                                   vertexOrder,
                                                                   texW, texH, pixS, texImage,
                                                                   no_t, t_c[0], t_c[1], material);
            }
            else
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = renderer->om->replacePolygonCB(object, no_poly, no_vert,
                                                             no_points, x_c, y_c, z_c,
                                                             v_l, l_l,
                                                             no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                             no_n, normalbinding, xn, yn, zn, transparency,
                                                             vertexOrder,
                                                             texW, texH, pixS, texImage,
                                                             no_t, NULL, NULL, material);

                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = renderer->om->replaceTriangleStripCB(object, no_strip, no_vert,
                                                                   no_points, x_c, y_c, z_c,
                                                                   v_l, l_l,
                                                                   no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                   no_n, normalbinding, xn, yn, zn, transparency,
                                                                   vertexOrder,
                                                                   texW, texH, pixS, texImage,
                                                                   no_t, NULL, NULL, material);
            }

            if (strcmp(gtype, "LINES") == 0)
                objNode = renderer->om->replaceLineCB(object, no_lines, no_vert,
                                                      no_points, x_c, y_c, z_c,
                                                      v_l, l_l,
                                                      no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                      no_n, normalbinding, xn, yn, zn);

            if (no_points && ((strcmp(gtype, "POINTS") == 0) || (strcmp(gtype, "UNSGRD") == 0)))
                objNode = renderer->om->replacePointCB(object, no_points,
                                                       x_c, y_c, z_c,
                                                       colorbinding, colorpacking, rc, gc, bc, pc);

            if (no_points && (strcmp(gtype, "SPHERE") == 0))
            {
                objNode = renderer->om->replaceSphereCB(object, no_points,
                                                        x_c, y_c, z_c, radii_c, no_c,
                                                        colorbinding, colorpacking, rc, gc, bc, pc, iRenderMethod);
            }

            if (colormap_attrib != NULL)
                renderer->om->addColormap(object, colormap_attrib);
        }

        else // if not (doreplace)
        {

            if (strcmp(gtype, "UNIGRD") == 0)
            {
                int no_of_lut_entries = 0;
                uchar *rgbalut = NULL;
                if (pixS == 4 && texH == 1)
                {
                    no_of_lut_entries = texW;
                    rgbalut = texImage;
                }
                else if (pixS == 3)
                {
                    fprintf(stderr, "got 3 component (RGB) colormap texture -- set transparentTextures in section Colors to true in covise.config\n");
                }
                objNode = renderer->om->addUGridCB(object, root, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                                                   no_c, colorbinding, colorpacking, rc, gc, bc, pc, byteData,
                                                   no_n, normalbinding, xn, yn, zn, transparency,
                                                   no_of_lut_entries, rgbalut);
            }
            if (strcmp(gtype, "RCTGRD") == 0)
                objNode = renderer->om->addRGridCB(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                                                   no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                   no_n, normalbinding, xn, yn, zn, transparency);
            if (strcmp(gtype, "STRGRD") == 0)
                objNode = renderer->om->addSGridCB(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                                                   no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                   no_n, normalbinding, xn, yn, zn, transparency);

            // texture coordinates ?
            if (t_c)
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = renderer->om->addPolygonCB(object, root, no_poly, no_vert,
                                                         no_points, x_c, y_c, z_c,
                                                         v_l, l_l,
                                                         no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                         no_n, normalbinding, xn, yn, zn, transparency,
                                                         vertexOrder,
                                                         texW, texH, pixS, texImage,
                                                         no_t, t_c[0], t_c[1], material,
                                                         rName, feedbackStr);
                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = renderer->om->addTriangleStripCB(object, root, no_strip, no_vert,
                                                               no_points, x_c, y_c, z_c,
                                                               v_l, l_l,
                                                               no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                               no_n, normalbinding, xn, yn, zn, transparency,
                                                               vertexOrder,
                                                               texW, texH, pixS, texImage,
                                                               no_t, t_c[0], t_c[1], material,
                                                               rName, feedbackStr);
            }
            else
            {
                if (strcmp(gtype, "POLYGN") == 0)
                    objNode = renderer->om->addPolygonCB(object, root, no_poly, no_vert,
                                                         no_points, x_c, y_c, z_c,
                                                         v_l, l_l,
                                                         no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                         no_n, normalbinding, xn, yn, zn, transparency,
                                                         vertexOrder,
                                                         texW, texH, pixS, texImage,
                                                         no_t, NULL, NULL, material,
                                                         rName, feedbackStr);
                if (strcmp(gtype, "TRIANG") == 0)
                    objNode = renderer->om->addTriangleStripCB(object, root, no_strip, no_vert,
                                                               no_points, x_c, y_c, z_c,
                                                               v_l, l_l,
                                                               no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                               no_n, normalbinding, xn, yn, zn, transparency,
                                                               vertexOrder,
                                                               texW, texH, pixS, texImage,
                                                               no_t, NULL, NULL, material,
                                                               rName, feedbackStr);
            }

            if (strcmp(gtype, "LINES") == 0)
                objNode = renderer->om->addLineCB(object, root, no_lines, no_vert,
                                                  no_points, x_c, y_c, z_c,
                                                  v_l, l_l,
                                                  no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                  no_n, normalbinding, xn, yn, zn, material,
                                                  rName, feedbackStr);
            if ( // no_points &&     AW: we DO accept 0-sized objects now
                ((strcmp(gtype, "POINTS") == 0) || (strcmp(gtype, "UNSGRD") == 0)))
            {
                const char *ptSizeStr = geometry->getAttribute("POINTSIZE");
                float ptSize = ptSizeStr ? atof(ptSizeStr) : 0.0;

                objNode = renderer->om->addPointCB(object, root, no_points,
                                                   x_c, y_c, z_c,
                                                   no_c, colorbinding, colorpacking,
                                                   rc, gc, bc, pc, ptSize);
            }

            if (no_points && (strcmp(gtype, "SPHERE") == 0))
            {
                objNode = renderer->om->addSphereCB(object, root, no_points,
                                                    x_c, y_c, z_c, radii_c,
                                                    no_c, colorbinding, colorpacking,
                                                    rc, gc, bc, pc, iRenderMethod, rName);
            }
            if (colormap_attrib != NULL)
                renderer->om->addColormap(object, colormap_attrib);
        }
    }

    if (labelStr && objNode)
    {
        SbVec2s sz = renderer->viewer->getSize();
        SbViewportRegion vpReg;
        vpReg.setWindowSize(sz);

        vpReg = renderer->viewer->getCamera()->getViewportBounds(vpReg);

        SoGetBoundingBoxAction bBoxAct(vpReg);
        bBoxAct.apply(objNode);
        SbBox3f bb = bBoxAct.getBoundingBox();
        SbVec3f center = bb.getCenter();

        SoSeparator *sep = new SoSeparator;
        SoTranslation *tr = new SoTranslation;
        tr->translation.setValue(center);
        sep->addChild(tr);

        SoMaterial *mat = new SoMaterial();
        mat->emissiveColor.setValue(1, 1, 1);
        objNode->addChild(mat);
        sep->addChild(mat);

        SoFont *font = new SoFont;
        font->name.setValue("Arial");
        font->size.setValue(18.0);
        sep->addChild(font);

        SoBillboard *bboard = new SoBillboard;
        sep->addChild(bboard);

        SoTranslation *tr2 = new SoTranslation;
        tr2->translation.setValue(SbVec3f(0.07f, 0.0f, 0.0f));
        sep->addChild(tr2);

        SoText2 *text = new SoText2;
        text->string.setValue(labelStr);
        sep->addChild(text);

        if (objNode->isOfType(SoSwitch::getClassTypeId()))
        {
            SoNode *n = objNode->getChild(0);
            if (n->isOfType(SoGroup::getClassTypeId()))
            {
                ((SoGroup *)n)->addChild(sep);
            }
            else
            {
                objNode->addChild(sep);
            }
        }
        else
        {
            objNode->addChild(sep);
        }
    }

    delete[] labelStr;

    if (delete_pc)
        delete[] pc;
}

//==========================================================================
// receive a object delete message
//==========================================================================
void InvCommunicator::receiveDeleteObjectMessage(QString object)
{
    QString buffer;

    // AW: first look for replacements
    Repl *repl = replace_->next; // 1st is dummy
    Repl *orep = replace_;

    while (repl && repl->newName != object)
    {
        orep = repl;
        repl = repl->next;
    }

    if (repl)
    {
        //cerr << "Using old " << repl->oldName << " instead " << object << endl;
        buffer = repl->oldName;
        orep->next = repl->next;
        object = buffer;
    }

    int i, n;
    for (i = 0; i < anzset; i++)
    {
        if (setnames[i] == object)
        {
            for (n = 0; n < elemanz[i]; n++)
            {
                receiveDeleteObjectMessage(elemnames[i][n]);
                delete[] elemnames[i][n];
                elemnames[i][n] = NULL;
            }
            delete[] elemnames[i];
            elemnames[i] = NULL;

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
    renderer->om->deleteObjectCB(object.toLatin1());
    renderer->om->deleteColormap(object.toLatin1());
    //renderer->om->deletePartCB(object);
    //renderer->om->deleteTimePartCB(object);

    if (coviseViewer && coviseViewer->getTextManager())
        coviseViewer->getTextManager()->removeAnnotation(object.toStdString().c_str());
}

//==========================================================================
// receive delete all message
//==========================================================================
void InvCommunicator::receiveDeleteAll(void)
{
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
        renderer->om->deleteObjectCB(object);
        renderer->om->deleteColormap(object);
        //renderer->om->deletePartCB(object);
        //renderer->om->deleteTimePartCB(object);
    }
}

void InvCommunicator::setReplace(QString oldName, QString newName)
{
    Repl *repl = replace_->next; // 1st is dummy
    Repl *orep = replace_;

    while (repl && repl->newName != oldName)
    {
        orep = repl;
        repl = repl->next;
    }

    if (repl) // replace more than the first time
    {
        repl->newName = newName;
    }
    else
        orep->next = new Repl(oldName, newName);
}

//==========================================================================
// receive a object delete message
//==========================================================================
void InvCommunicator::receiveCameraMessage(QString list)
{
    float pos[3];
    float ori[4];
    int view;
    float aspect;
    float mynear;
    float myfar;
    float focal;
    float angle;

    const char *msg = list.toLatin1();
    size_t retval;
    retval = sscanf(msg, "%f %f %f %f %f %f %f %d %f %f %f %f %f",
                    &pos[0], &pos[1], &pos[2], &ori[0], &ori[1], &ori[2], &ori[3],
                    &view, &aspect, &mynear, &myfar, &focal, &angle);
    if (retval != 13)
    {
        std::cerr << "InvCommunicator::receiveCameraMessage: sscanf failed" << std::endl;
        return;
    }

    renderer->viewer->setTransformation(pos, ori, view, aspect, mynear, myfar, focal, angle);
}

//==========================================================================
// receive a transform message
//==========================================================================
void InvCommunicator::receiveTransformMessage(QString)
{

    /*float pos[3];
   float ori[4];
   int view;
   float aspect;
   float near;
   float far;
   float focal;
   float angle;

   sscanf(message,"%f %f %f %f %f %f %f %d %f %f %f %f %f",
      &pos[0],&pos[1],&pos[2],&ori[0],&ori[1],&ori[2],&ori[3],
   &view,&aspect,&near,&far,&focal,&angle);

   viewer->setTransformation( pos, ori, view, aspect, near, far,focal, angle );
   */
}

//==========================================================================
// receive a telepointer message
//==========================================================================
void InvCommunicator::receiveTelePointerMessage(QString message)
{
    renderer->viewer->tpHandler->handle(message.toLatin1());
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunicator::receiveDrawstyleMessage(QString)
{
    //viewer->receiveDrawstyle(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunicator::receiveLightModeMessage(QString)
{
    //viewer->receiveLightMode(message);
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunicator::receiveSelectionMessage(QString message)
{
    renderer->viewer->setSelection(message.toLatin1());
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunicator::receiveDeselectionMessage(QString message)
{
    renderer->viewer->setDeselection(message.toLatin1());
}

//==========================================================================
// receive a part switching message
//==========================================================================
void InvCommunicator::receivePartMessage(QString)
{
    //viewer->receivePart(message);
}

//==========================================================================
// receive a reference part message
//==========================================================================
void InvCommunicator::receiveReferencePartMessage(QString)
{
    //viewer->receiveReferencePart(message);
}

//==========================================================================
// receive a reset scene message
//==========================================================================
void InvCommunicator::receiveResetSceneMessage()
{
    //viewer->receiveResetScene();
}

//==========================================================================
// receive a drawstyle message
//==========================================================================
void InvCommunicator::receiveTransparencyMessage(QString)
{
    //viewer->receiveTransparency(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveSyncModeMessage(QString message)
{
    if (renderer->sequencer)
        renderer->sequencer->setSyncMode(message);

    if (message == "LOOSE")
    {
        renderer->setSyncMode(InvMain::SYNC_LOOSE);
    }

    else if (message == "SYNC")
    {
        renderer->setSyncMode(InvMain::SYNC_SYNC);
    }

    else
    {
        renderer->setSyncMode(InvMain::SYNC_TIGHT);
    }
}

//==========================================================================
// receive a fog mode message
//==========================================================================
void InvCommunicator::receiveFogMessage(QString)
{
    //cerr << message << endl;
    //viewer->receiveFog(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveAntialiasingMessage(QString)
{
    //viewer->receiveAntialiasing(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveBackcolorMessage(const char *message)
{
    float r, g, b;

    int retval;
    retval = sscanf(message, "%f %f %f", &r, &g, &b);
    if (retval != 3)
    {
        std::cerr << "InvCommunicator::receiveBackcolorMessage: sscanf failed" << std::endl;
        return;
    }

    // set backcolor in viewer ...
    renderer->viewer->setBackgroundColor(SbColor(r, g, b));
    // keep fog color up to date with bkg color
    //environment->fogColor.setValue(SbColor(r,g,b));
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveAxisMessage(const char *message)
{
    int onoroff;

    int retval;
    retval = sscanf(message, "%d", &onoroff);
    if (retval != 1)
    {
        std::cerr << "InvCommunicator::receiveAxisMessage: sscanf failed" << std::endl;
        return;
    }

    // set antialiasing in viewer ...
    renderer->viewer->setAxis(onoroff);

    renderer->switchAxisMode(onoroff);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveClippingPlaneMessage(const char *message)
{
    int onoroff;
    float val[6];

    int retval;
    retval = sscanf(message, "%d %f %f %f %f %f %f",
                    &onoroff,
                    &val[0], &val[1], &val[2],
                    &val[3], &val[4], &val[5]);
    if (retval != 7)
    {
        std::cerr << "InvCommunicator::receiveClippingPlaneMessage: sscanf failed" << std::endl;
        return;
    }
    SbVec3f normal(val[0], val[1], val[2]);
    SbVec3f point(val[3], val[4], val[5]);

    // set clipping plane in viewer ...
    renderer->viewer->setClipping(onoroff);

    if (onoroff == CO_ON)
        renderer->viewer->setClippingPlane(normal, point);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveViewingMessage(QString)
{
    //viewer->receiveViewing(message);
}

//==========================================================================
// receive a sync mode message
//==========================================================================
void InvCommunicator::receiveProjectionMessage(QString)
{
    //viewer->receiveProjection(message);
}

//==========================================================================
// receive a decoration mode message
//==========================================================================
void InvCommunicator::receiveDecorationMessage(QString)
{
    //viewer->receiveDecoration(message);
}

//==========================================================================
// receive a headlight mode message
//==========================================================================
void InvCommunicator::receiveHeadlightMessage(QString)
{
    //viewer->receiveHeadlight(message);
}

//==========================================================================
// receive a sequencer message
//==========================================================================
void InvCommunicator::receiveSequencerMessage(QString message)
{
    renderer->om->receiveSequencer(message.toLatin1());
}

//==========================================================================
// receive a colormap message
//==========================================================================
void InvCommunicator::receiveColormapMessage(QString)
{
    //viewer->receiveColormap(message);
}

//==========================================================================
// receive a master message
//==========================================================================
//void InvCommunicator::receiveMasterMessage(QString mod, QString inst, QString host)
void InvCommunicator::receiveMasterMessage(QString mod)
{

    m_name = mod;

    renderer->setMaster(true);
}

//==========================================================================
// receive an update message
//==========================================================================
void InvCommunicator::receiveUpdateMessage(QString mod, QString inst, QString host)
{
    cerr << "receiveUpdateMessage____________ not yet handled " << endl;
    m_name = mod;
    inst_no = inst;
    h_name = host;

    renderer->viewer->updateObjectView();

    if (renderer->sequencer)
    {
        char buf[255];
        sprintf(buf, "%d %d %d %d %d %d %d", renderer->sequencer->getValue(),
                renderer->sequencer->getMinimum(), renderer->sequencer->getMaximum(),
                renderer->sequencer->getMinimum(), renderer->sequencer->getMaximum(),
                renderer->sequencer->getSeqState(), 1);
        sendSequencerMessage(buf);
    }
}

//==========================================================================
// receive a slave message
//==========================================================================
void InvCommunicator::receiveSlaveMessage(QString msg)
{
    m_name = msg;

    renderer->setMaster(false);
}

//==========================================================================
// receive a slave message
//==========================================================================
void InvCommunicator::receiveMasterSlaveMessage(QString mod, QString inst, QString host)
{
    m_name = mod;
    inst_no = inst;
    h_name = host;
    //renderer->switchMasterSlave();
}
