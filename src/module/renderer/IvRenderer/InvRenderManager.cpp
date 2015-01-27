/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// *************************************************************************
//
// * Description    : This is the render manager for the renderer
//
//
// * Class(es)      : inv_RM
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.08.93 V 1.0
//
//
//
// *************************************************************************
//
// debug stuff (local use)
//
#include <covise/covise.h>
#ifdef DEBUG
#define DBG
#endif
#include "xpm.h"
#include <unistd.h>

//
// Inventor stuff
//
#include <Inventor/Xt/SoXt.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/nodes/SoGroup.h>

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>
#include <Xm/Xm.h>
#include <Xm/AtomMgr.h>
#include <Xm/VendorSEP.h>
#include <Xm/Protocols.h>
#include <Xm/MainW.h>

//
// C stuff
//
#include <unistd.h>
#include <sys/times.h>

//
// class include file
//
#include "InvRenderManager.h"

//
// the global render manager
//
InvRenderManager *rm = NULL;

//
// the global object manager
//
InvObjectManager *om = NULL;

clock_t tp_lasttime, cam_lasttime; // control rate of telepointer messages
float tp_rate; // minimum difference in seconds between two telepointer messages
struct tms tinfo;
clock_t curr_time;
double diff;

//
// colors used by the renderer
//

Pixel pix_blue;
Pixel pix_green;
Pixel pix_white;
Pixel pix_red;
Pixel pix_gold;
Pixel pix_cyan;

Pixel top_color[3];
Pixel sel_color[3];
Pixel bot_color[3];
Pixel fg_color[3];

//
// fallback resources for the renderer
//
// Problems with resource file COvise ? uncomment this...
//
const char *fallbacks[] = {
    "*background: grey87",
    "*fontList:-*-helvetica-bold-r-*-*-12-*=charset",
    "*TransientShell.ancestorSensitive:TRUE",
    NULL
};

// and comment this out ...
//
// char *fallbacks[] =
//   {
//    "*TransientShell.ancestorSensitive:TRUE",
//    NULL};

//
// NOTE: the following functions have to be friends of inv_RM
// in order to be callable from classes that use the communication
// environment. These classes don't like classes which are using
// Inventor stuff . Future releases of Inventor will make this
// compiler version problem disappear.
//

//=========================================================================
// update slave objects views  (friend of inv_RM)
//=========================================================================
void
rm_updateSlave()
{
    ((InvCoviseViewer *)(rm->r))->updateSlaves();

    if (om->timeStepper != NULL)
    {
        char buf[255];
        InvSequencer *seq = om->timeStepper;
        sprintf(buf, "%d %d %d %d %d %d %d", seq->getValue(), seq->getMinimum(), seq->getMaximum(), seq->getSliderMinimum(), seq->getSliderMaximum(), seq->getSeqAct(), seq->getSeqState());
        rm_sendSequencer(buf);
    }
}

//=========================================================================
// switch renderer to master (friend of inv_RM)
//=========================================================================
void rm_switchMaster()
{
    rm->r->setMaster();

    if (om->timeStepper != NULL)
        om->timeStepper->setActive();
}

//=========================================================================
// switch renderer to slave mode (friend of inv_RM)
//=========================================================================
void rm_switchSlave()
{
    rm->r->setSlave();

    if (om->timeStepper != NULL)
        om->timeStepper->setInactive();
}

//=========================================================================
// switch between master and slave (friend of inv_RM)
//=========================================================================
void rm_switchMasterSlave()
{
    rm->r->setMasterSlave();
}

//=========================================================================
// test if master (friend of inv_RM)
//=========================================================================
int rm_isMaster()
{
    return rm->r->isMaster();
}

//=========================================================================
// test if master (friend of inv_RM)
//=========================================================================
int rm_isSynced()
{
    return rm->r->isSynced();
}

//=========================================================================
// set the actual rendering time (friend of inv_RM)
//=========================================================================
void rm_setRenderTime(float time)
{
    rm->r->setRenderTime(time);
}

//=========================================================================
// called from communication manager when a new camera position has arrived
//=========================================================================
void rm_receiveCamera(char *message)
{
    //
    // parse string and set values
    //
    float pos[3];
    float ori[4];
    int view;
    float aspect;
    float near;
    float far;
    float focal;
    float angle;

    int ret = sscanf(message, "%f %f %f %f %f %f %f %d %f %f %f %f %f",
                     &pos[0], &pos[1], &pos[2], &ori[0], &ori[1], &ori[2], &ori[3],
                     &view, &aspect, &near, &far, &focal, &angle);
    if (ret != 13)
    {
        fprintf(stderr, "rm_receiveCamera: sscanf failed\n");
    }

    rm->r->setTransformation(pos, ori, view, aspect, near, far, focal, angle);
}

//=========================================================================
//  send a new camera position to the controller (friend of inv_RM)
//=========================================================================

void rm_sendCamera(char *message)
{
#ifdef __linux__
    static double clocks_per_sec = sysconf(_SC_CLK_TCK);
#else
    static double clocks_per_sec = CLK_TCK;
#endif

    curr_time = times(&tinfo);
    diff = ((double)curr_time - (double)cam_lasttime) / clocks_per_sec;

    if (diff > tp_rate)
    {
        InvCommunication *cm = new InvCommunication;
        cm->sendCameraMessage(message);
        cam_lasttime = curr_time;
        delete cm;
    }
}

//=========================================================================
//  send a new VRMLcamera position to the controller (friend of inv_RM)
//=========================================================================

void rm_sendVRMLCamera(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendVRMLCameraMessage(message);

    delete cm;
}

//=========================================================================
//  send a new transformation to the controller (friend of inv_RM)
//=========================================================================

void rm_sendTransformation(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendTransformMessage(message);

    delete cm;
}

//=========================================================================
//  send a quit request to the controller (friend of inv_RM)
//=========================================================================

void rm_sendQuitRequest()
{
/*
     InvCommunication *cm = new InvCommunication;

     cm->sendQuitRequestMessage();

     delete cm;
   */

#ifdef _COLLAB_RENDERER

    InvCommunication *cm = new InvCommunication;

    cm->sendQuitMessage();

    delete cm;
#endif
}

//=========================================================================
//  send a new drawstyle to the controller (friend of inv_RM)
//=========================================================================

void rm_sendDrawstyle(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendDrawstyleMessage(message);

    delete cm;
}

//=========================================================================
//  send a new light mode to the controller (friend of inv_RM)
//=========================================================================
void rm_sendLightMode(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendLightModeMessage(message);

    delete cm;
}

//=========================================================================
//  send a new light mode to the controller (friend of inv_RM)
//=========================================================================
void rm_sendSelection(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendSelectionMessage(message);

    delete cm;
}

//=========================================================================
//  send a new light mode to the controller (friend of inv_RM)
//=========================================================================
void rm_sendDeselection(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendDeselectionMessage(message);

    delete cm;
}

//=========================================================================
//  send a new part switching to the controller (friend of inv_RM)
//=========================================================================
void rm_sendPart(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendPartMessage(message);

    delete cm;
}

//=========================================================================
//  send a new reference part to the controller (friend of inv_RM)
//=========================================================================
void rm_sendReferencePart(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendReferencePartMessage(message);

    delete cm;
}

//=========================================================================
//  send a new reset scene to the controller (friend of inv_RM)
//=========================================================================
void rm_sendResetScene()
{
    InvCommunication *cm = new InvCommunication;

    cm->sendResetSceneMessage();

    delete cm;
}

//=========================================================================
//  send a new transparency level to the controller (friend of inv_RM)
//=========================================================================
void rm_sendTransparency(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendTransparencyMessage(message);

    delete cm;
}

//=========================================================================
//  send a new sync mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendSyncMode(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendSyncModeMessage(message);

    // now set the sequencer according to the global sync mode
    if (om->timeStepper != NULL)
        om->timeStepper->setSyncMode(message, rm_isMaster());

    delete cm;
}

//=========================================================================
//  send a new fog mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendFog(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendFogMessage(message);

    delete cm;
}

//=========================================================================
//  send a new aliasing mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendAntialiasing(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendAntialiasingMessage(message);

    delete cm;
}

//=========================================================================
//  send a new camera position to the controller (friend of inv_RM)
//=========================================================================

void rm_sendColormap(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendColormapMessage(message);

    delete cm;
}

//=========================================================================
//  send a new back color to the controller (friend of inv_RM)
//=========================================================================

void rm_sendBackcolor(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendBackcolorMessage(message);

    delete cm;
}

//=========================================================================
//  send a new axis mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendAxis(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendAxisMessage(message);

    delete cm;
}

//=========================================================================
//  send a new clipping plane to the controller (friend of inv_RM)
//=========================================================================

void rm_sendClippingPlane(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendClippingPlaneMessage(message);

    delete cm;
}

//=========================================================================
//  send a new viewing mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendViewing(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendViewingMessage(message);

    delete cm;
}

//=========================================================================
//  send a new projection mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendProjection(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendProjectionMessage(message);

    delete cm;
}

//=========================================================================
//  send a new decoration mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendDecoration(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendDecorationMessage(message);

    delete cm;
}

//=========================================================================
//  send a new headlight mode to the controller (friend of inv_RM)
//=========================================================================

void rm_sendHeadlight(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendHeadlightMessage(message);

    delete cm;
}

//=========================================================================
//  send a new sequencer to the controller (friend of inv_RM)
//=========================================================================

void rm_sendSequencer(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendSequencerMessage(message);

    delete cm;
}

//=========================================================================
// called from communication manager when a new telepointer has arrived
//=========================================================================
void rm_receiveTelePointer(char *message)
{

    rm->r->receiveTelePointer(message);
}

//=========================================================================
// called from communication manager when a new drawstyle has arrived
//=========================================================================
void rm_receiveDrawstyle(char *message)
{

    rm->r->receiveDrawstyle(message);
}

//=========================================================================
// called from communication manager when a new light model has arrived
//=========================================================================
void rm_receiveLightMode(char *message)
{

    rm->r->receiveLightMode(message);
}

//=========================================================================
// called from communication manager when a new selection has arrived
//=========================================================================
void rm_receiveSelection(char *message)
{

    rm->r->receiveSelection(message);
}

//=========================================================================
// called from communication manager when a new deselection has arrived
//=========================================================================
void rm_receiveDeselection(char *message)
{

    rm->r->receiveDeselection(message);
}

//=========================================================================
// called from communication manager when new part switching  has arrived
//=========================================================================
void rm_receivePart(char *message)
{

    rm->r->receivePart(message);
}

//=========================================================================
// called from communication manager when new reference part has arrived
//=========================================================================
void rm_receiveReferencePart(char *message)
{

    rm->r->receiveReferencePart(message);
}

//=========================================================================
// called from communication manager when new reset scene has arrived
//=========================================================================
void rm_receiveResetScene()
{

    rm->r->receiveResetScene();
}

//=========================================================================
// called from communication manager when a new drawstyle has arrived
//=========================================================================
void rm_receiveTransparency(char *message)
{

    rm->r->receiveTransparency(message);
}

//=========================================================================
// called from communication manager when a new sync mode has arrived
//=========================================================================
void rm_receiveSyncMode(char *message)
{

    rm->r->receiveSyncMode(message);
    if (om->timeStepper != NULL)
        om->timeStepper->setSyncMode(message, rm_isMaster());
}

//=========================================================================
// called from communication manager when a new fog mode has arrived
//=========================================================================
void rm_receiveFog(char *message)
{

    rm->r->receiveFog(message);
}

//=========================================================================
// called from communication manager when a new aliasing mode has arrived
//=========================================================================
void rm_receiveAntialiasing(char *message)
{

    rm->r->receiveAntialiasing(message);
}

//=========================================================================
// called from communication manager when a new back color mode has arrived
//=========================================================================
void rm_receiveBackcolor(char *message)
{

    rm->r->receiveBackcolor(message);
}

//=========================================================================
// called from communication manager when a new axis mode has arrived
//=========================================================================
void rm_receiveAxis(char *message)
{

    rm->r->receiveAxis(message);
}

//=========================================================================
// called from communication manager when a new clipping plane has arrived
//=========================================================================
void rm_receiveClippingPlane(char *message)
{

    rm->r->receiveClippingPlane(message);
}

//=========================================================================
// called from communication manager when a new viewing mode has arrived
//=========================================================================
void rm_receiveViewing(char *message)
{

    rm->r->receiveViewing(message);
}

//=========================================================================
// called from communication manager when a new projection mode has arrived
//=========================================================================
void rm_receiveProjection(char *message)
{

    rm->r->receiveProjection(message);
}

//=========================================================================
// called from communication manager when a new decoration mode has arrived
//=========================================================================
void rm_receiveDecoration(char *message)
{

    rm->r->receiveDecoration(message);
}

//=========================================================================
// called from communication manager when a new projection mode has arrived
//=========================================================================
void rm_receiveHeadlight(char *message)
{

    rm->r->receiveHeadlight(message);
}

//=========================================================================
// called from communication manager when a new sequencer info has arrived
//=========================================================================
void rm_receiveSequencer(char *message)
{

    om->receiveSequencer(message);
}

//=========================================================================
// called from communication manager when a new colormap info has arrived
//=========================================================================
void rm_receiveColormap(char *message)
{

    rm->r->receiveColormap(message);
}

void
rm_sendCSFeedback(char *key, char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendCSFeedback(key, message);

    delete cm;
}

void
rm_sendAnnotation(char *key, char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendAnnotation(key, message);

    delete cm;
}

//=========================================================================
//  send a new telepointer to the controller (friend of inv_RM)
//=========================================================================

void rm_sendTelePointer(char *message)
{

// SGI searches old-style - must work for both
#ifdef __linux__
    static double clocks_per_sec = sysconf(_SC_CLK_TCK);
#else
    static double clocks_per_sec = CLK_TCK;
#endif

    curr_time = times(&tinfo);
    diff = ((double)curr_time - (double)tp_lasttime) / clocks_per_sec;

    // some messages MUST be send like those with CO_RMV content (remove telepointer)
    // therefore we have to check its content
    int state;
    float dummy;
    char tpname[32];
    int ret = sscanf(message, "%s %d %f %f %f %f", &tpname[0], &state, &dummy, &dummy, &dummy, &dummy);
    if (ret != 6)
    {
        fprintf(stderr, "rm_sendTelePointer: sscanf failed\n");
    }

    if ((diff > tp_rate) || (state == CO_RMV))
    {
        InvCommunication *cm = new InvCommunication;
        cm->sendTelePointerMessage(message);
        tp_lasttime = curr_time;
        delete cm;
    }
}

//=========================================================================
//  send a new VRMLtelepointer to the controller (friend of inv_RM)
//=========================================================================

void rm_sendVRMLTelePointer(char *message)
{
    InvCommunication *cm = new InvCommunication;

    cm->sendVRMLTelePointerMessage(message);

    delete cm;
}

//=========================================================================
// called from communication manager when a new transformation has arrived
//=========================================================================
void rm_receiveTransformation(char *message)
{

    rm->r->receiveTransformation(message);
}

//=========================================================================
// define the colors for the renderer (friend of inv_RM)
//=========================================================================
void rm_defineColors(Widget Toplevel)
{
    Colormap cmap;
    XColor red;
    XColor blue;
    XColor green;
    XColor white;
    XColor gold;
    XColor cyan;
    XColor unused;

    XtVaGetValues(Toplevel, XmNcolormap, &cmap, NULL);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "chocolate", &gold, &unused);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "orchid", &blue, &unused);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "indigo", &green, &unused);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "crimson", &red, &unused);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "cyan4", &white, &unused);
    XAllocNamedColor(XtDisplay(Toplevel), cmap, "gold3", &cyan, &unused);
    pix_blue = blue.pixel;
    pix_green = green.pixel;
    pix_white = white.pixel;
    pix_red = red.pixel;
    pix_gold = gold.pixel;
    pix_cyan = cyan.pixel;
    /*
     XmGetColors(XtScreen(Toplevel), cmap, pix_gold,
     &fg_color[0], &top_color[0],
     &bot_color[0], &sel_color[0]);
     XmGetColors(XtScreen(Toplevel), cmap, pix_white,
     &fg_color[1], &top_color[1],
     &bot_color[1], &sel_color[1]);
     XmGetColors(XtScreen(Toplevel), cmap, pix_blue,
     &fg_color[2], &top_color[2],
     &bot_color[2], &sel_color[2]);
   */
}

//=========================================================================
// startup for the render manager (friend of inv_RM)
//=========================================================================
void rm_startup(int argc, char *argv[])
{
    tp_rate = 0.01;
    tp_lasttime = cam_lasttime = 0;

    //
    // create a render manager
    //
    rm = new InvRenderManager(argc, (char **)argv, (char **)fallbacks);

    //
    // create an explorer style renderer (type 0)
    //
    rm->createRenderer(0, (char *)"Renderer");

    //
    // start an object manager for the renderer
    //
    om = new InvObjectManager();

    //
    // show the renderer widget
    //
    rm->showRenderer();

    //
    // X event main loop, never return from here
    //
    rm->mainLoop();
}

//================================================================
// class InvRenderManager
//================================================================

//================================================================
// constructor for the render manager
//================================================================
InvRenderManager::InvRenderManager(int argc, char *argv[], char *fallbacks[])
{
    argcount = argc;

    strcpy(render_name, argv[0]);
    strcat(render_name, "_");
    strcat(render_name, argv[4]);
    strcat(render_name, "@");
    strcat(render_name, appmod->get_hostname());
    strcat(render_name, " (Inv. 2.1)");
    // set the correct display
    if (getenv((char *)"DISPLAY") == NULL)
        putenv((char *)"DISPLAY=:0");

    // create toplevel shell
    //
    toplevel = XtVaAppInitialize(&app_context, "COvise", NULL,
                                 0, &argcount, argv, fallbacks, NULL);
    screen = XtScreen(toplevel);
    disp = XtDisplay(toplevel);

    rm_defineColors(toplevel);

    tp_rate = 0.02;
    tp_lasttime = cam_lasttime = 0;
}

//================================================================
// create the renderer
//================================================================
void
InvRenderManager::createRenderer(int type, char *name)
{
    //Arg args[2];
    //int err;
    //Pixmap mask;
    Atom WM_DELETE_WINDOW;

    strcpy(title, name);
    shell = XtAppCreateShell(name, NULL,
                             applicationShellWidgetClass,
                             disp, NULL, 0);

    // set icon
    pixmap = XmGetPixmap(screen,
                         (char *)"wingdogs",
                         BlackPixelOfScreen(screen),
                         WhitePixelOfScreen(screen));
    //    err = XpmCreatePixmapFromData(disp,(Drawable)XtWindow(toplevel),
    //eurologo_xpm,&pixmap,&mask,NULL);
    //err = XpmReadFileToPixmap(disp,(Drawable)XtWindow(toplevel),
    //"/visin1/people/0125/zrfv/covise/pixmaps/test.xpm",&pixmap,&mask,NULL);
    //    fprintf(stderr,"Pixmap-Fehler %d\n",err);
    // popup as  icon
    XtVaSetValues(shell,
                  XmNiconic, TRUE,
                  XmNiconName, "Renderer",
                  XmNiconPixmap, pixmap,
                  XmNtitle, render_name,
                  //		XmNiconWindow,	XtWindow(toplevel),
                  //		XmNiconMask,	mask,
                  //		XmNinitialState, IconicState,
                  XmNdeleteResponse, XmDO_NOTHING,
                  NULL);
    WM_DELETE_WINDOW = XmInternAtom(disp, (char *)"WM_DELETE_WINDOW", FALSE);

    XmAddProtocolCallback(shell,
                          XmInternAtom(XtDisplay(shell), (char *)"WM_PROTOCOLS", FALSE),
                          WM_DELETE_WINDOW,
                          (XtCallbackProc)rm_sendQuitRequest,
                          (XtPointer)NULL);

    // that's the default size
    //
    // XtSetArg( args[0], XmNwidth,  550 );
    // XtSetArg( args[1], XmNheight, 640 );
    // XtSetValues(shell, args, 2);

    SoXt::init(shell);
    if (type == 0)
    {
        r = new InvCoviseViewer();
        appWindow = r->buildWidget(shell, render_name);
        // create a connection to the environment
    }
    else
    {
        print_comment(__LINE__, __FILE__, "ERROR :Sorry no renderer of this type");
        print_comment(__LINE__, __FILE__, "Creating interactive renderer instead");
        r = new InvCoviseViewer();
        appWindow = r->buildWidget(shell, render_name);
    }

    p = new InvPort();
    p->setConnection(app_context, NULL);
}

//================================================================
// set size of renderer
//================================================================
void
InvRenderManager::setSize(int nx, int ny)
{
    int sx, sy;
    int offset = 0;

#ifdef _AIRBUS
    offset = 30;
#endif

#ifdef _COLLAB_VIEWER
    sx = DisplayWidth(disp, DefaultScreen(disp));
    sy = DisplayHeight(disp, DefaultScreen(disp));
#else
    sx = nx + offset;
    sy = ny;
#endif

    if (nx == 0 && ny == 0)
    {
        float pal_ratio = 720. / 576.;
        int bx = 60, by = 199; //border size
        sy = DisplayHeight(disp, DefaultScreen(disp)) - 100;
        sx = (int)((float)(sy - by) * pal_ratio + (float)bx + 0.5);
    }
    if (shell != NULL)
    {
        Arg args[2];

        XtSetArg(args[0], XmNwidth, sx);
        XtSetArg(args[1], XmNheight, sy);
        XtSetValues(shell, args, 2);
    }
    else
    {
        print_comment(__LINE__, __FILE__, "ERROR :sorry no renderer created ");
    }
}

//================================================================
// set a new root node , camera set to see everything
//================================================================
void
InvRenderManager::setSceneGraph(SoNode *root)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->setSceneGraph(root);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a root node to the renderer, camera set to see everything
//================================================================
void
InvRenderManager::addToSceneGraph(SoGroup *child, const char *name, SoGroup *root)
{
    if (toplevel != NULL && shell != NULL)
    {
#ifdef TIMING
#endif
        r->addToSceneGraph(child, name, root);
#ifdef TIMING
#endif
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a texture node to the renderer
//================================================================
void
InvRenderManager::addToTextureList(SoTexture2 *tex)
{
    if (toplevel != NULL && shell != NULL)
    {
#ifdef TIMING
#endif
        r->addToTextureList(tex);
#ifdef TIMING
#endif
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// remove a texture node from the renderer
//================================================================
void
InvRenderManager::removeFromTextureList(SoTexture2 *tex)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->removeFromTextureList(tex);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a colormap to the renderer
//================================================================
void InvRenderManager::addColormap(const char *name, const char *colormap)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->addColormap(name, colormap);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a part to the renderer
//================================================================
void
InvRenderManager::addPart(const char *name, int partId, SoSwitch *s)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->addPart(name, partId, s);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// replace the part
//================================================================
void
InvRenderManager::replacePart(const char *name, int partId, SoSwitch *s)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->replacePart(name, partId, s);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// remove a part from the renderer
//================================================================
void
InvRenderManager::deletePart(const char *name)
{
    if (NULL == name)
    {
        return;
    }
    if (toplevel != NULL && shell != NULL)
    {
        r->deletePart(name);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a time-part to the renderer
//================================================================
void
InvRenderManager::addTimePart(const char *name, int timeStep, int partId, SoSwitch *s)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->addTimePart(name, timeStep, partId, s);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// replace a time-part to the renderer
//================================================================
void
InvRenderManager::replaceTimePart(const char *name, int timeStep, int partId, SoSwitch *s)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->replaceTimePart(name, timeStep, partId, s);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// remove a time-part from the renderer
//================================================================
void
InvRenderManager::deleteTimePart(const char *name)
{
    if (NULL == name)
    {
        return;
    }
    if (toplevel != NULL && shell != NULL)
    {
        r->deleteTimePart(name);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// add a root node to the renderer, camera set to see everything
//================================================================
void
InvRenderManager::removeFromSceneGraph(SoGroup *root, const char *name)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->removeFromSceneGraph(root, name);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// remove a colormap from the renderer
//================================================================
void
InvRenderManager::deleteColormap(const char *name)
{
    if (NULL == name)
    {
        return;
    }
    if (toplevel != NULL && shell != NULL)
    {
        r->deleteColormap(name);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// replace the root , no camera change
//================================================================
void
InvRenderManager::replaceSceneGraph(SoNode *root)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->replaceSceneGraph(root);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// replace the colormap
//================================================================
void
InvRenderManager::replaceColormap(const char *name, const char *colormap)
{
    if (toplevel != NULL && shell != NULL)
    {
        r->replaceColormap(name, colormap);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// show the renderer
//================================================================
void
InvRenderManager::showRenderer()
{
    if (toplevel != NULL && shell != NULL)
    {
        r->show();
        SoXt::show(shell);
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR :you have to build a renderer before");
}

//================================================================
// go in X event loop
//================================================================
void
InvRenderManager::mainLoop()
{
    if (toplevel != NULL && shell != NULL)
        SoXt::mainLoop();
    else
        print_comment(__LINE__, __FILE__,
                      "ERROR :you have to build a renderer before");
}

//================================================================
//  destructor
//================================================================
InvRenderManager::~InvRenderManager()
{
    delete r;
    p->removeConnection();
    delete p;
}
