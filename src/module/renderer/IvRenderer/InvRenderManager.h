/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_RENDER_MANAGER_H
#define _INV_RENDER_MANAGER_H

/* $Id: InvRenderManager.h /main/vir_main/1 19-Nov-2001.15:25:40 sasha_te $ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// *************************************************************************
//
// * Description    :  the render manager for the renderer
//
//
// * Class(es)      : InvRenderManager
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
// *************************************************************************
//
//
//

//
// other classes
//
#include "InvRenderer.h"
#include "InvCoviseViewer.h"
#include "InvObjectManager.h"
#include "InvDefs.h"
#include "InvPort.h"
#include "InvCommunication.h"

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>

//
// global externals
//
extern ApplicationProcess *appmod;

//
// prototypes
//
void rm_startup(int argc, char *argv[]);

//
// friends
//
void rm_updateSlave();
void rm_switchMaster();
void rm_switchSlave();
void rm_switchMasterSlave();
int rm_isMaster();
int rm_isSynced();
void rm_startup();
void rm_sendCamera(char *message);
void rm_sendQuitRequest();
void rm_receiveCamera(char *message);
void rm_sendTransformation(char *message);
void rm_receiveTransformation(char *message);
void rm_sendTelePointer(char *message);
void rm_receiveTelePointer(char *message);
void rm_sendDrawstyle(char *message);
void rm_receiveDrawstyle(char *message);
void rm_sendLightMode(char *message);
void rm_receiveLightMode(char *message);
void rm_sendSelection(char *message);
void rm_receiveSelection(char *message);
void rm_sendDeselection(char *message);
void rm_receiveDeselection(char *message);
void rm_sendPart(char *message);
void rm_receivePart(char *message);
void rm_sendReferencePart(char *message);
void rm_receiveReferencePart(char *message);
void rm_sendResetScene();
void rm_receiveResetScene();
void rm_sendTransparency(char *message);
void rm_receiveTransparency(char *message);
void rm_sendSyncMode(char *message);
void rm_receiveSyncMode(char *message);
void rm_sendFog(char *message);
void rm_receiveFog(char *message);
void rm_sendAntialiasing(char *message);
void rm_receiveAntialiasing(char *message);
void rm_sendBackcolor(char *message);
void rm_receiveBackcolor(char *message);
void rm_sendAxis(char *message);
void rm_receiveAxis(char *message);
void rm_sendClippingPlane(char *message);
void rm_receiveClippingPlane(char *message);
void rm_sendViewing(char *message);
void rm_receiveViewing(char *message);
void rm_sendProjection(char *message);
void rm_receiveProjection(char *message);
void rm_sendSequencer(char *message);
void rm_receiveSequencer(char *message);
void rm_defineColors(Widget Toplevel);
void rm_setRenderTime(float time);
void rm_sendColormap(char *message);
void rm_receiveColormap(char *message);
void rm_addColormap(const char *name, const char *colormap);
void rm_replaceColormap(const char *name, const char *colormap);
void rm_deleteColormap(const char *name);
void rm_addPart(char *name, int part_id, SoSwitch *s);
void rm_replacePart(char *name, int part_id, SoSwitch *s);
void rm_deletePart(char *name);
void rm_addTimePart(char *name, int timeStep, int part_id, SoSwitch *s);
void rm_replaceTimePart(char *name, int timeStep, int part_id, SoSwitch *s);
void rm_deleteTimePart(char *name);

//
// CLASSES
//
class InvRenderManager;

//================================================================
// InvRenderManager
//================================================================

class InvRenderManager
{
private:
    XtAppContext app_context;
    Widget appWindow;
    Widget toplevel;
    Widget shell;

    int argcount;

    Pixmap pixmap;
    Screen *screen;
    Display *disp;
    InvRenderer *r;
    InvPort *p;
    char title[50];
    char render_name[100];

    friend void rm_updateSlave();
    friend void rm_switchMaster();
    friend void rm_switchSlave();
    friend void rm_switchMasterSlave();
    friend int rm_isMaster();
    friend int rm_isSynced();
    friend void rm_sendQuitRequest();
    friend void rm_sendVRMLCamera(char *message);
    friend void rm_sendCamera(char *message);
    friend void rm_receiveCamera(char *message);
    friend void rm_sendTransformation(char *message);
    friend void rm_receiveTransformation(char *message);
    friend void rm_sendVRMLTelePointer(char *message);
    friend void rm_sendTelePointer(char *message);
    friend void rm_receiveTelePointer(char *message);
    friend void rm_sendDrawstyle(char *message);
    friend void rm_receiveDrawstyle(char *message);
    friend void rm_sendLightMode(char *message);
    friend void rm_receiveLightMode(char *message);
    friend void rm_sendSelection(char *message);
    friend void rm_receiveSelection(char *message);
    friend void rm_sendDeselection(char *message);
    friend void rm_receiveDeselection(char *message);
    friend void rm_sendPart(char *message);
    friend void rm_receivePart(char *message);
    friend void rm_sendReferencePart(char *message);
    friend void rm_receiveReferencePart(char *message);
    friend void rm_sendResetScene();
    friend void rm_receiveResetScene();
    friend void rm_sendTransparency(char *message);
    friend void rm_receiveTransparency(char *message);
    friend void rm_sendSyncMode(char *message);
    friend void rm_receiveSyncMode(char *message);
    friend void rm_sendFog(char *message);
    friend void rm_receiveFog(char *message);
    friend void rm_sendAntialiasing(char *message);
    friend void rm_receiveAntialiasing(char *message);
    friend void rm_sendBackcolor(char *message);
    friend void rm_receiveBackcolor(char *message);
    friend void rm_sendAxis(char *message);
    friend void rm_receiveAxis(char *message);
    friend void rm_sendClippingPlane(char *message);
    friend void rm_receiveClippingPlane(char *message);
    friend void rm_sendViewing(char *message);
    friend void rm_receiveViewing(char *message);
    friend void rm_sendProjection(char *message);
    friend void rm_receiveProjection(char *message);
    friend void rm_sendDecoration(char *message);
    friend void rm_receiveDecoration(char *message);
    friend void rm_sendHeadlight(char *message);
    friend void rm_receiveHeadlight(char *message);
    friend void rm_sendSequencer(char *message);
    friend void rm_receiveSequencer(char *message);
    friend void rm_sendColormap(char *message);
    friend void rm_receiveColormap(char *message);
    friend void rm_addColormap(const char *name, const char *colormap);
    friend void rm_replaceColormap(const char *name, const char *colormap);
    friend void rm_deleteColormap(const char *name);
    friend void rm_defineColors(Widget Toplevel);
    friend void rm_setRenderTime(float time);
    friend void rm_addPart(char *name, int partId, SoSwitch *s);
    friend void rm_replacePart(char *name, int partId, SoSwitch *s);
    friend void rm_deletePart(char *name);
    friend void rm_addTimePart(char *name, int timeStep, int partId, SoSwitch *s);
    friend void rm_replaceTimePart(char *name, int timeStep, int partId, SoSwitch *s);
    friend void rm_deleteTimePart(char *name);
    friend void rm_sendCSFeedback(char *key, char *message);
    friend void rm_sendAnnotation(char *key, char *message);

public:
    InvRenderManager(int argc, char *argv[], char *fallbacks[]);
    void createRenderer(int type, char *name);
    void setSize(int nx, int ny);
    void setSceneGraph(SoNode *node);
    void addToSceneGraph(SoGroup *child, const char *name, SoGroup *node);
    void removeFromSceneGraph(SoGroup *root, const char *name);
    void replaceSceneGraph(SoNode *node);
    void addToTextureList(SoTexture2 *tex);
    void removeFromTextureList(SoTexture2 *tex);
    void addColormap(const char *name, const char *colormap);
    void replaceColormap(const char *name, const char *colormap);
    void deleteColormap(const char *name);
    void addPart(const char *name, int partId, SoSwitch *s);
    void replacePart(const char *name, int partId, SoSwitch *s);
    void deletePart(const char *name);
    void addTimePart(const char *name, int timeStep, int partId, SoSwitch *s);
    void replaceTimePart(const char *name, int timeStep, int partId, SoSwitch *s);
    void deleteTimePart(const char *name);
    void showRenderer();
    void mainLoop();
    void addInteractor(coDistributedObject *obj);

    ~InvRenderManager();
};

/* Icon Pixmap */
/*static char * eurologo_xpm[] =
{
   "97 97 115 2",
   "  	c #B6B6DADAFFFF",
   ". 	c #9191DADAFFFF",
   "X 	c #9191B6B6FFFF",
   "o 	c #B6B6FFFFFFFF",
   "O 	c #DADAFFFFFFFF",
   "+ 	c #9191B6B6AAAA",
   "@ 	c #48486D6DAAAA",
   "# 	c #6D6D6D6DAAAA",
   "$ 	c #000024240000",
   "% 	c #000000005555",
   "& 	c #000000000000",
   "* 	c #000024245555",
   "= 	c #48486D6D5555",
   "- 	c #00000000AAAA",
   "; 	c #6D6D9191FFFF",
   ": 	c #24242424AAAA",
   "> 	c #DADADADAFFFF",
   ", 	c #6D6D9191AAAA",
   "< 	c #FFFFFFFFFFFF",
   "1 	c #91919191AAAA",
   "2 	c #484848485555",
   "3 	c #91919191FFFF",
   "4 	c #B6B6B6B6FFFF",
   "5 	c #FFFFFFFF0000",
   "6 	c #B6B6B6B65555",
   "7 	c #48484848AAAA",
   "8 	c #B6B6B6B6AAAA",
   "9 	c #DADADADAAAAA",
   "0 	c #242424245555",
   "q 	c #6D6D6D6D5555",
   "w 	c #919191915555",
   "e 	c #00000000FFFF",
   "r 	c #DADADADA0000",
   "t 	c #6D6D6D6DFFFF",
   "y 	c #24244848FFFF",
   "u 	c #DADADADA5555",
   "i 	c #24242424FFFF",
   "p 	c #FFFFFFFF5555",
   "a 	c #48484848FFFF",
   "s 	c #B6B6B6B60000",
   "d 	c #48486D6DFFFF",
   "f 	c #00002424FFFF",
   "g 	c #00002424AAAA",
   "h 	c #B6B691915555",
   "j 	c #00004848FFFF",
   "k 	c #91916D6D5555",
   "l 	c #24244848AAAA",
   "z 	c #919191910000",
   "x 	c #6D6D6D6D0000",
   "c 	c #24246D6DFFFF",
   "v 	c #FFFFDADAFFFF",
   "b 	c #6D6DB6B6FFFF",
   "n 	c #48489191FFFF",
   "m 	c #DADAB6B6AAAA",
   "M 	c #FFFFFFFFAAAA",
   "N 	c #DADAB6B6FFFF",
   "B 	c #DADAFFFFAAAA",
   "V 	c #B6B6DADA5555",
   "C 	c #B6B6DADAAAAA",
   "Z 	c #9191DADAAAAA",
   "A 	c #48489191AAAA",
   "S 	c #DADAFFFF5555",
   "D 	c #6D6D4848AAAA",
   "F 	c #48482424AAAA",
   "G 	c #B6B6FFFFAAAA",
   "H 	c #6D6DB6B6AAAA",
   "J 	c #B6B69191FFFF",
   "K 	c #9191B6B65555",
   "L 	c #91916D6DFFFF",
   "P 	c #91916D6DAAAA",
   "I 	c #DADAFFFF0000",
   "U 	c #FFFFDADA0000",
   "Y 	c #B6B691910000",
   "T 	c #242448485555",
   "R 	c #DADAB6B60000",
   "E 	c #6D6D4848FFFF",
   "W 	c #DADA91910000",
   "Q 	c #B6B69191AAAA",
   "! 	c #FFFFB6B60000",
   "~ 	c #6D6D48485555",
   "^ 	c #DADAB6B65555",
   "/ 	c #FFFFB6B65555",
   "( 	c #DADA91915555",
   ") 	c #DADA6D6D0000",
   "_ 	c #919148480000",
   "` 	c #FFFF91910000",
   "' 	c #B6B66D6D5555",
   "] 	c #FFFF6D6D0000",
   "[ 	c #B6B648480000",
   "{ 	c #DADA00000000",
   "} 	c #6D6D24245555",
   "| 	c #B6B64848AAAA",
   " .	c #FFFF48480000",
   "..	c #DADA48485555",
   "X.	c #919148485555",
   "o.	c #FFFF24240000",
   "O.	c #FFFF00000000",
   "+.	c #B6B66D6DAAAA",
   "@.	c #919124245555",
   "#.	c #6D6DDADAFFFF",
   "$.	c #B6B600005555",
   "%.	c #919100005555",
   "&.	c #6D6D2424AAAA",
   "*.	c #FFFF00005555",
   "=.	c #DADA00005555",
   "-.	c #6D6D00005555",
   ";.	c #91914848AAAA",
   ":.	c #48480000AAAA",
   ">.	c #B6B600000000",
   ",.	c #484800005555",
   "<.	c #242400005555",
   "1.	c #919100000000",
   "2.	c #6D6D00000000",
   "3.	c #242400000000",
   "4.	c #484800000000",
   "  .   X   .   X   .   X   .   X   .   X   .   X   .   X   .   X   . o   O o O +   @ # $ % & % & % & % & % * # =   + O o O   o .   .   X   .   X   .   X   .   X   .   X   .   X   .   X   .   X   ",
   "    X       X       X       X       X       X       X       .   o O   X @ % & & & % % % % % % % % - % % % % % % & & & % @ X   O o   .       X       X       X       X       X       X       X     ",
   "  X   X   X   X   X   X   X   X   X   X   X   X   X   X     O   ; % % & % % - % % % - % - % - % - % - % - % - % - % - % % & % % ;   O     X   X   X   X   X   X   X   X   X   X   X   X   X   X   ",
   "X   X   X   X   X . X   X   X   X   X   X . X   X     o   X : % % % % - % - % - > > % - % - - - - - - - % - - - % - % % % - % % % % : X         X   X   X   X   X . X   X   X   X   X   X . X   X ",
   "  X   X   X   X   X   X   X   X   X   X   X   X     o , : % - % - % - % - % - # < < % % - % - % 1 2 - % - % - % - % < < 3 % - % - % - % : , o     X   X   X   X   X   X   X   X   X   X   X   X   ",
   "X . X X X . X X X . X X X . X X X . X X X   X     X - % - - - - - > < > % - % 4 < < % - - - - - 5 6 - - - - - - - - < < 1 - - - - - - - - - - X     X X X . X X X . X X X . X X X . X X X . X X X ",
   "  X   X   X   X   X   X   X   X   X   X   X o X : % - - - - - - - 1 < < 7 % - 8 < 9 - - - 0 8 q 5 5 1 q - - - - - % < < - % - - - - - - - % - % : X o X   X   X   X   X   X   X   X   X   X   X   ",
   "X X X X X X X X X X X X X X X X X X X .     @ - - - - - - - - - - - > < < - % 4 < 4 - - - - w 5 5 5 5 : - - - - - 3 < < - - - - - - - - - - - - - - @     X X X X X X X X X X X X X X X X X X X X ",
   "  X X X   X X X   X X X   X X X   X   . X - - - e - - - e - - - e - - < < # - 8 < # - - e - e r 5 5 6 - e - - - - > < 8 - - - - e - - - e - e - e - - - X     X   X X X   X X X   X X X   X X X   ",
   "X X X X X X X X X X X X X X X X X   . t - - - - - e - - - - - - - - - 3 < < - 3 < 3 - - - e - 5 # 8 5 - - - - - - < < 7 - - - - - - - e - : - - - - - - - ; . . X X X X X X X X X X X X X X X X X ",
   "X 3 X X X 3 X X X ; X X X 3 X X   X y - e - e - e - # - e - e - e - e - < < t # < 7 e - e - 1 q e - 8 7 e - e - e < < - e - e - e w 1 e u w e - e - e - e - y X   X X X X 3 X X X ; X X X ; X X X ",
   "X X X X X X X X X X X X X X X   X i - e - e - e - e u 5 - u 7 e - e - - - < < > < t - e - e - e - e - e - e - - 3 < < - - e - e - e 5 5 5 7 e e - e - e - - - i X   X X X X X X X X X X X X X X X ",
   "X ; X ; X ; X ; X ; X ; X X   ; e - e - e - e - e e i 5 5 5 e - e - e - e # < < < - e - e - e - e - e - e - e - < < 4 - e - e - e e p 5 5 6 e - e - e - e - e - e ;   ; X ; X ; X ; X ; X ; X ; X ",
   "X X 3 X X X 3 X X X 3 X X   ; - - e - e - e - e e 1 5 5 5 5 : e e e - e - e < < < e - e e e - e e e - e e e - e < < 7 - e e - e 7 5 5 5 5 u 6 : e e - e e e - e - e ; X X X 3 X X X 3 X X X 3 X X ",
   "X ; X ; X ; X ; X ; X ;   ; e - e - e - e - e - e w 8 5 5 5 5 6 e - e - e - i > 3 - e - e - e - e - e - e - e - t 8 e - e - e - 1 7 e r 5 e e - e - e - e - e - e - e ;   ; X ; X ; X ; X ; X ; X ",
   "; 3 ; X ; 3 ; X ; 3 3 X ; e - e - a > < 4 a e e e e e 5 6 e e i e e e e e e e e - e e e e e e e e e e e e e e e - e e e e e e e e e e 1 5 e e e e e e e 3 > 3 e - e - e ; X 3 X ; X ; X ; X 3 X ; ",
   "X ; X ; X ; X ; X ; X ; e - e - < < < < < < 4 - e - e 6 t e e - e - e - e - e - e e e e e e e e e e e e e e e - e - e - e - e - e - e - 8 e e - e - e 4 < < < < a - e - e ; X ; X ; X ; X ; X ; X ",
   "; X ; ; ; X ; ; ; X X i e e e > < < t t > < < 4 e e e a e e e e e e e e e e e e e e - e - e - e - e - - - e - e e e e e e e e e e e e e e e e e e e t < < t 4 < < e e e e i X X ; 3 ; ; ; 3 ; ; ; ",
   "3 ; X ; 3 ; X ; X X a - e - i < < - e - e a < < 4 - e e e - e e e - e e e - e e e s 9 6 9 6 9 6 6 6 9 6 9 6 u - e - e e e - e e e - e e e - e e e - 4 < < - e t < > e e e - a X X ; X ; X ; X ; X ",
   "; ; ; 3 ; ; ; 3 X d e e e e 4 < 4 e e e e e 3 < < e e e e e e e e e e e e e e e e 6 1 q q q q 8 w 1 q q q q 6 e e e e e e e e e e e e e e e e e > e e < < i e e < < e e e e e d X 3 ; ; ; ; ; ; ; ",
   "3 ; ; t 3 ; ; ; X e e e e e > < 4 e e - e e e < < e e - e e e - e e e e e e e e e 6 8 1 4 8 8 1 1 8 8 8 4 8 9 - e e e e e e e - e e e e e e e 4 < 4 e 4 < < e : < < e e e e e - X ; 3 ; 3 ; 3 ; 3 ",
   "; ; ; ; ; ; ; 3 e e e e e e 3 < < e e e e e e < < i e e e e e e e e e e e e e e e 6 1 q q 1 q 1 q 1 q # q 1 6 e e e e e e e e e e e e e e e e < < 3 e e < < t e > 4 e e e e e e e X ; ; ; ; ; ; ; ",
   "; t ; t ; t X a e e e e e e e < < 4 e e e e e < < e e e e e e e e e e e e e e e e 6 8 w 8 w 8 1 1 1 8 1 8 w 9 - e e e e e e e e e e e e e e e 4 < 3 e e t < < e e e e e e e e e e a X ; ; t ; t ; ",
   "; ; ; ; ; ; ; e e e e e e e e t < < > e e i < < 3 e e e e e e e e e e e e e e e e 6 1 1 1 1 1 8 q 8 1 1 1 1 6 e e e e e e e e e e e e e e e e e < < i e e < < e e e e e e e e e e e ; ; ; ; ; ; ; ",
   "; t ; d ; ; y e e e e e e e e e 4 < < < < < < > e e e e e e e e e e e e e e e e e 6 8 w 8 1 8 w 1 8 8 1 8 1 9 - e e e e e e e e e e e e e e e e 3 < < > < < < e e e e e e e e e e e i ; ; t ; d ; ",
   "t ; t ; ; t e e f i e f f i e e e a < < < < 3 e e f e f f i e i i i e i i i f f e 6 1 1 1 1 w 1 q 8 q 1 1 1 6 e e i f i i i f i i i f i i i i e e t < < < < e e i i e a e e i i i i e t ; ; t ; ; ",
   "; d ; d ; i i i y i y f f e i i f e e e e e e e i i y i y i y i y f y i y f y f e 6 8 w 1 w 8 1 1 1 1 w 1 w 9 - i i y i y f y i y i y i y i y i e e e a t e e e y f e 6 d e e e i i i i ; d ; d ; ",
   "d t t ; d i i y i f e e y t e y i y i f e i i y i y i y i y i y i y i y i y i y e 6 8 1 1 1 1 8 q 8 1 1 1 1 6 e i y i y i y i y i y i y i y i y i i f e e f i i e e e p 5 t 5 t i y i i d ; t t d ",
   "; d ; d a i a i a i 8 e p 6 e e i i a i a i a i a i a i a i a i a i a i a i a i e 6 8 w 8 1 8 w 1 1 1 w 8 1 8 - a y a i a y a i a y a i a y a i a y a i a y a i t 5 5 5 5 5 9 e a y a i a d ; d ; ",
   "t t d t y a a a a i 5 5 5 5 u u d a a a a a y a a a a a a a a a a a y a a a y a e 6 1 1 1 1 w 1 q 8 w 1 1 1 6 f a d y a a a y a a a y a a d y a a a y a a a y y y 3 u 5 5 5 e e a a y a y t d d d ",
   "t a t a d a d a d e f 5 5 5 5 6 t y d a d a d a d a d a d a d a d a d a d a d a f w 8 q 1 q 1 1 1 w 1 q 1 w 8 g d a d a d a d a d a d a d a d a d a d a d a d a a e e 5 p 5 8 i d a d a d a ; d t ",
   "d d d d a d d d a f 1 5 5 5 e e y d d d d d d d d d d d d d d d d d d d d d d d f 6 8 8 1 8 8 8 q 8 1 8 8 8 6 i d d d d d d d d d d d d d d d d d d d d d d d d d a d 5 i e t a a d d d a d d d d ",
   "t a d a t d t a t a 9 # a 5 t y t d t a t d d a t d d a t d t a t d d a t d d a y h 1 2 1 q # w 1 w # q 1 q 8 g t d d a t d d a t d t a t d d a t d d a t d t a t a d # j y d a t d t a t d d a t ",
   "d a d t d t d d d d y y f 8 t d d t d d d t d d d t d d d t d d d t d d d t d t y 6 8 8 8 8 8 8 1 8 8 8 8 8 6 y d t d d d t d d d t d d d t d d d t d d d t d d d t d d d t d d d t d d d t d a d ",
   "d y ; d t y y d ; y ; d d a t d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d d w 8 q 1 q 1 1 1 w 1 q 1 k 8 l ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d ; d d y d d ; d ; y d ",
   "a d ; ; y t O < < d d d d d d ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; a z w w w w w w x w w w w w z d ; ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; t ; d d y j y 3 4 d d ; ; d y ",
   "a d ; c ; < < <   y d t < ; d d ; t ; d ; t ; d ; t ; d ; t ; d ; t ; d ; t ; d ; l d y d d d l d d d l d a d d ; t ; d ; t ; d ; t ; d ; t ; d ; t ; d ; t ; d ; d d j d t 4 > < < < d ; t ; d a ",
   "i ; ; c > < > X d ; d   < < d d ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; t > d ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ; d ; X O < < < < < < < X d ; ; ; i ",
   "i ; ; d < < d c ; ; ; ; < < > d 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; ; 7 3 < 4 @ X ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; 3 ; X < < < < < < > d ; < > c ; X ; y ",
   "y X ; ; < < c ; ; 3 ; d ; < < d ; ; ; 3 ; 3 ; 3 ; 3 ; 3 ; ; ; 3 ; 3 ; 3 ; ; d < < ; f ; ; 3 ; 3 ; 3 ; 3 ; ; ; 3 ; 3 ; 3 ; 3 ; 3 ; ; ; 3 ; 3 ; 3 ; 3 ; 3 ; ; ; ; d < < > ; ; < < j d < < d ; ; X y ",
   "d X ; d < < ; d X ; X d X < < c X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X d 4 < < < v # b ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; n > < ; c c < < ; c < < X ; X X d ",
   "; X ; ; < < < X d n d d > < > ; 3 X ; X 3 X ; X X X ; X 3 X ; X X X ; X X d 8 < < < < > d X ; X X X ; X 3 X ; X 3 X ; X X X ; X 3 X ; X 3 X ; X 3 X ; X 3 X ; X ; 4 < > d n < < X c   <   ; ; X ; ",
   "; X X ; X < < < < > O < < < X ; X ; X ; X ; X ; X ; X ; X ; X X X @ ; ; X @ 3 < < < 4 c X ; X ; X ; X ; X ; X ; X ; X X X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; X ; < < ; d > < O c   < < ; X X ; ",
   "; X X X ; X < < < < < < <   ; X X X X X X X X X X X X X X X ; X n > > ; ; X c v - y ; X X X X X X X X X X X X X X X b d ; X X X X X X X X X X X X X X X X X X X X ; < < X ; X < > ; b 4 ; b X X ; ",
   "  X X X X ; X X O < <     ; X X . X X X . X X X X X X X   b 3 1 d m < M d @ 3 M < l ; X   X X X 4 X X X   X X X X , v # . X X X 4 X X X 4 X X 3 X X X X X X X X X n < < > ; X ; X ; X ; X X X X   ",
   "X X X X X X X b ; b ; b X X X X X X X X X X X X X X X X X 3 < < < < < < 4 X 8 < < < # X X X X X X X X X X X X X - # < ; . . X X X X X X X X X X X X X X X X X X X X 4 < 4 X X X X X X X X X X X X ",
   "  X   X   X   X   X   X   X   X   X   X   X   X   X   X   @ 3 < < < < l ; X b 1 < < 4 ;   X   X   X   X   X   b 4 < < 7 ; X   X   X   X   X   X   X   X   X   X   X X b X X   X   X . X   X   X   ",
   "X . X . X X X X X X X . X . X . X . X . X . X   X . X ; @ ; 4 < < < v ; .   b 4 < < < ; X . X   X . X   X . . b > < N t b   X   X . X   X . X   X . X   X . X   X . X . X . X X X X 9 X X . X   X ",
   "  X   X   X B V X C B X   X   X   X   X   X   X   X   X 3 < < < < < 4 b . , c m < < < 8 X .   X   X   X   X   , < > n X   X   X   X   X   X   X   X   X   X   X   X   X   X   + X C 5 b   X   X   ",
   "    .   . . C 5 5 5 C .     .       .       .           1 < m 4 1 4 b   ; < > < < < < 1 b           .         . 4 > b               .       .       .       .       .         5 5 5 5 b .   .     ",
   "  X   X B 5 5 5 5 C . X   X   .   X   .   X   .   X   . X , X b   X o b ; < < < < < < , . .   .   X   .   . o . 3 < ; b . b X X   .   .   X   .   X   X   X   .   X   .   X .   5 5 5 5 B X   X   ",
   "          9 5 5 5 p X                                                 3 < < < < < < < 4 1 X           X b b 1 b # < < 3 1 4 > 4 X                                           X B 5 5 5 9           ",
   "O     Z   X O 5 O 5 M X       C       Z       Z       C       C O   . A y : 4 < < < < < < 1   C O , ; 8 < > < 9 < < < < < < < < X     Z       Z       C       Z       Z     B 5 M S 5 b       C O ",
   "  >         9 9 X   C                                           O X N < < < < < < < < # 1 X     , 4 D > < < < < < < < < < < < < , O                                         C   .   S             ",
   "X C O C > C O     C   C > C O C > C O C > C O C > C O C > C O C 7 F v v > 1 4 4 < < < #   G O , 3 < < < < < < < < < < < < < < 8 X C O C > C O C > C O C > C O C > C O C > C   C         > C O C X ",
   "X O C   C       C   C   C   C   C   C   C   C   C   C   C     O C o Z X +   C X , 3 8 . + 3 @ N < < < < < < < < < < < < < < < < , O     C       C       C     > C     > C     > C     > C     O X ",
   "t B O C O C O C O C O C O C O C O C O C O C O C O C > C O C O C O C O C O 9 O 9 O C o H > < < < < < < < < < < < < < < < < < < < 4 C O C O C > C O C O C O C O C O C O C O C O C O C O C O C O B t ",
   "i M C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 B C + C B   7 C C 3 1 > < < < < < < < < < < < < < < < < < < < < J C C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C 9 9 O C M i ",
   "e B O C O C O C O C O C O C O C O C O C O C O C O C O C O C O C O # < 1 C K L 8 # > < < < < < < < < < < < < < < < < < < < > 4 1 , C O C O C O C O C O C O C O C O C O C O C O C O C O C O C O B e ",
   "e B B B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 3 < < > N > < < < < < < < < < < < < < < < < < < < < < # , B B B B 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 9 B 9 9 B B e ",
   "e 8 M C O 9 B C O 9 B C O 9 M B < B < B < B < B < B < B < B M 9 < C + 8 < < < < < < < < < < < < < < < < < < < < < < < < 3 V M C O 9 B 9 < B < B < B < B < B < B < B < B < 9 B C O 9 B C O 9 M 8 e ",
   "e t p B B B 9 B B B 9 B B B K C V C 6 C V C 6 C V C 6 C V C 6 B B B u X < < < < < < < < < < < < < < < < < < < < < < < < < 8 S B B B B B 6 C 6 C V C 6 C V C 6 C V C 6 C 6 B 9 B B B 9 B B B p t e ",
   "e e M S B 9 B 9 B 9 M 9 M 6 P 1 1 q P q P q P q P q P q 1 1 # u M 9 M K v < < < < < < < < < < < < < < < < < < < < < < < < < 4 S M 9 M 6 P 1 1 q P q P q P q P q P q 1 1 # V M 9 B 9 B 9 B S M e e ",
   "e e 9 M B B B B 9 B B B B u 1 1 9 > 9 > 9 > 9 > 9 > 9 > 8 1 1 B B B B S 1 < < < < < < < < < < < < < < < > < < < < < < < < 8 # B B B B 9 1 1 8 > 9 > 9 > 9 > 9 > 9 > 8 1 1 B B B 9 B B B 9 M 9 e e ",
   "e e 3 p M u M u M u M u M V 1 8 < < < < < < < < < < < < < 1 1 u M u M S + < < < < < < < < < < < < v 4 w 8 w 3 4 v < < < 3 S M S M u M V 1 8 < < < < < < < < < < < < < 1 1 u M u M S M u M p 3 e e ",
   "e e e p p M S M M M S M M u # 9 < < > < < < > < < < > < < 1 # p M M S p 1 < < < < < < < < < < < < 8 5 p p p p S r 6 6 V 6 M S M B M p S # 4 < < < < > < < < > < < < < 1 # p p M B M S M p p e e e ",
   "e e e u p S M S M u M S M r # 8 < 9 < > < 9 < > < 9 < > < w 1 S M u M I 7 > < < < < < < < < < < 1 z V w 8 r S 6 4 1 8 6 S I M u M u M u # 8 < > < 9 < > < 9 < > < 9 < 1 P S M u M u M S p u e e e ",
   "e e e i 5 p p p p p p p p u # > < < < < < < < < < < < < < 1 # p p p p V > < < < < < < < < < < < > 4 v < < 4 4 < < < < < > 1 p p p p p S # 8 < < < < < < < < < < < < < 1 # p p p p p p p 5 t e e e ",
   "e e e e p 5 p S M S p S M r # 8 < > < 9 < > < 9 < > < 9 < 1 P 5 M 5 u 8 < < < < < < < < < < < < < < < < < < < < < < < < < 6 p S M S p r # 8 < 9 < > < 9 < > < 9 < > < 1 1 5 p S M S p 5 p e e e e ",
   "e e e e # 5 p p p p p p p 5 7 8 < < > < < < > < < < > < < 1 # 5 p p 3 < < < < < < < < < < < < < < < < < < < < < < < < # 8 5 p p p p p 5 # 4 < < < < > < < < > < < < < 1 # 5 p p p p p 5 # e e e e ",
   "e e e e e 5 5 5 p 5 p 5 p r # 8 < 9 < > < 9 < > < 9 < > < 1 1 5 p 5 u > < < < < < < < < < < < < < < < < < < < < < < < w 5 5 p 5 p 5 p r # 8 < > < 9 < > < 9 < > < 9 < 1 # 5 p 5 p 5 5 5 e e e e e ",
   "e e e e e t 5 5 5 p 5 p 5 5 # > < < < < < < < < < < < < < 1 # 5 5 5 5 u 4 > > < < < < 4 8 < < < < < < 4 w 8 4 < < < < 6 5 p 5 p 5 p 5 5 # 8 < < < < < < < < < < < < < 8 # 5 5 p 5 5 5 t e e e e e ",
   "e e e e e e 5 5 5 5 5 5 5 r 3 w < < < < < < < < < < < < < 1 1 5 p 5 5 5 5 s s 3 < > 6 5 5 s 4 > > 3 6 s 5 5 r 4 < < < w 5 5 5 5 p 5 5 r 3 1 < < < < < < < < < < < < < q 3 5 5 5 5 5 5 e e e e e e ",
   "e e e e e e i 5 5 5 5 5 5 r t 1 1 1 # , # 1 # , # 1 # 1 1 4 # 5 5 5 5 5 5 5 5 5 s 1 I 5 5 5 5 r # r 5 5 5 5 5 4 < < < > 6 5 5 5 5 5 5 I t 4 1 1 , 1 # , , 1 # , # 1 1 4 # 5 5 5 5 5 7 e e e e e e ",
   "e e e e e e e 6 5 5 5 U 5 U r x Y z s z s z s z s z s z z z r 5 5 U 5 U 5 U 5 5 5 5 5 5 5 U 5 5 5 5 5 r 4 w 5 w < < < < < 1 5 5 5 U 5 5 r x Y z s z s z s z s z s z z z r 5 5 5 5 6 e e e e e e e ",
   "e e e e e e e e 5 5 U 5 U 5 5 * & % & & & % & & & & T 0 & % 5 5 U 5 U 5 U 5 U 5 U 5 U 5 U 5 U 5 U 5 5 6   3 U U 4 < < < < 4 R 5 U 5 U 5 5 0 & % & & & % & & & & 0 0 & % 5 5 5 5 5 e e e e e e e e ",
   "e e e e e e e e E 5 5 U 5 U 9 T & & & & & & & & & & < 8 & & R U 5 U U U 5 U U U 5 U U U 5 U U U 5 U 5 R   R 5 U R 8 < < < < N W U U 5 U 9 7 & & & & & & & & & & < 8 & & R 5 5 5 3 e e e e e e e e ",
   "e e e e e e e e e Q U U U U     * 7 T T T l T T * , < < % g 8 U U U ! U U U ! U U U ! U U U ! U U U ! ! ~ U U U U U h   < < <   3 Y U U > > % T T l T T T l * # < < & * 4 U 5 ^ e e e e e e e e e ",
   "e - e - e - e - e e / U U W 1 w h w 8 w 8 w 8 w Q w 8 w w w 1 W U ! U ! U ! U ! U ! U ! U ! U ! U ! L   < Q ! ! U ! U W 8 < < < < 1 ( ) w h Q w 8 w 8 w 8 w 8 w 8 w h w Q ! U e e - e e e - e e e ",
   "e e e e e e e e e e e ! ! ! ! ! ! U ! U ! U ! U ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! _ < < O W ! ! ! ! ! ` Q 3 < < < < O W U ! U ! U ! U ! U ! U ! ! ! U U U e e e e e e e e e e e ",
   "e - e - e - e - e - e - ! ! ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` ' < < 1 ` ` ! ` ! ` ! ` ] 8 < < J o X ` ! ` ! ` ! ` ! ` ! ` ! ` ! ` U D e - e - e - e - e - e ",
   "- e - e - e - e - e - e e ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` 1 < O ) ` ` ` ` ` ` ` ` ` [ 4 < { ] ' ` ` ` ` ` ` ` ` ` ` ` ` ` ` ! D e - e - e - e - e - e e ",
   "e - e - e - e - e - e - e } ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` [ b # |  .` ] ` ] ` ] ` ] ` ] ..< Q ] ` ] ` ] ` ] ` ] ` ] ` ] ` ] ` X.e - e - e - e - e - e - e ",
   "- - - - - - - - - - - - - e F ] ] ] ] ]  .] ] ]  .] ] ]  .] ] ] ] ] ] ] ] ] ] ]  .]  .] ] ] ] ] ] ]  . . .] ] ] ] ] ] ] ] ] ] ] { O 8  .] ] ] ] ] ] ] ] ] ] ] ] ] ] ..e - - - - - - - - - - - - - ",
   "e - - - e - - - e - - - e - e -  . . .o. . . .o. . . .o. . . .o. . . .o. . . .o. . . .o.]  . . . . . . .]  . . .] o.o.O.O.O.o.O.+.. o.o.]  .]  .]  .]  .]  . . .] @.e - e - e - e - e - e - e - e ",
   "- - - - - - - - - - - - - - - - - o.O.o.o.o.O.o.o.o.O.o.o.o.O.o.o.o.o.o.o.o.O.o.o.o.o.o.o.o.o.o.o.o.o.o.o.o.o.o.o.O.#.4 $.$.X %.c ..O.o.o.o.o.o.o.o.o.o.o. .o.o.&.e - - - - - - - - - - - - - - - ",
   "- - - - - - - - - - - - - - - - e - *.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.=.O < < O { O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.-.e - - - - - - - - - - - - - - - - ",
   "- - - - - - - - - - - - - - - - - - - $.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.;.P O O { O.O.O.O.O.O.O.O.O.O.O.O.O.O.- - - - - - - - - - - - - - - - - - - ",
   "- % - % - % - % - % - % - % - % - % - - :.O.O.O.O.O.O.{ O.O.O.{ O.O.O.{ O.O.O.{ O.O.O.{ O.O.O.{ O.{ O.{ O.O.O.{ O.O.O.O.O.{ ;.{ O.O.O.{ O.O.O.{ O.O.O.O.=.- - % - % - % - % - % - % - % - % - % - ",
   "% - % - - - % - % - % - % - % - % - % - - - >.O.O.O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.O.O.{ O.O.O.{ O.O.O.O.O.O.O.{ O.O.O.{ O.O.O.O.,.- - % - - - % - - - % - - - % - - - % - - ",
   "- % - % - % - % - % - % - % - % - % - % - % - <.O.O.O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.{ O.O.O.1.- % - % - % - % - % - % - % - % - % - % - % - ",
   "% % % % % % % % % % % % % % % % % % % % % % % - % -.{ O.{ { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { { O.O.$.% - % % % % % % % % % % % % % % % % % % % % % % % ",
   "- % % % - % % % - % % % - % % % - % % % - % % % - % - 2.{ { { >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ >.{ { O.>.% % - % % % - % % % - % % % - % % % - % % % - % % % - ",
   "% % % % % % % % % % % % % % % % % % % % % % % % % % % % % -.>.{ { { >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.{ { { %.% - % % % % % % % % % % % % % % % % % % % % % % % % % % % ",
   "% & % & % & % & % & % & % & % & % & % % % & % & % & % % % % % 3.%.>.{ >.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.1.>.>.{ >.$.4.% % % & % & % & % & % & % & % & % & % & % & % & % & % & % & % ",
   "& % & % & % & % & % & % & % & % & % & % & % & % & % & % & % % % % % 3.2.1.>.1.>.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.>.>.>.1.2.4.% % % % % & % & % & % & % & % & % & % & % & % & % & % & % & % & % & ",
   "% & & & % & & & % & & & % & & & % & & & % & & & % & & & % & & & % & % & % & 3.3.2.4.2.4.2.4.2.4.2.4.2.4.2.4.2.4.2.4.3.& % & % & % & & & % & & & % & & & % & & & % & % & % & % & % & % & % & % & % "
};
*/
#endif
