/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_COMMUNICATION_H
#define _INV_COMMUNICATION_H

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  the communication message handler  for the renderer
//
//
// * Class(es)      : InvCommunication
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
#include <Xm/Xm.h>

//
// ec stuff
//
#include <covise/covise_process.h>
#include <do/coDoGeometry.h>

// Open Inventor
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSwitch.h>

//
// CLASSES
//
class InvCommunication;

//
// other classes
//
#include "InvDefs.h"
#include "InvError.h"
using namespace covise;

//
// defines
//
#define MAXDATALEN 500
#define MAXTOKENS 25
#define MAXHOSTLEN 20
#define MAXSETS 8000

//================================================================
// InvCommunication
//================================================================

class InvCommunication
{
private:
    int getNextPK();
    int getCurrentPK();
    int pk_;
    void send_ctl_msg(const char* message, int type);

public:
    InvCommunication();
    const char *getname(const char *file);
    int parseMessage(char *line, char *token[], int tmax, char *sep);
    void sendTelepointerMessage(char *message);
    void sendTransformMessage(char *message);
    void sendTelePointerMessage(char *message);
    void sendVRMLTelePointerMessage(char *message);
    void sendDrawstyleMessage(char *message);
    void sendLightModeMessage(char *message);
    void sendSelectionMessage(char *message);
    void sendDeselectionMessage(char *message);
    void sendPartMessage(char *message);
    void sendReferencePartMessage(char *message);
    void sendResetSceneMessage();
    void sendTransparencyMessage(char *message);
    void sendSyncModeMessage(char *message);
    void sendFogMessage(char *message);
    void sendAntialiasingMessage(char *message);
    void sendBackcolorMessage(char *message);
    void sendAxisMessage(char *message);
    void sendClippingPlaneMessage(char *message);
    void sendViewingMessage(char *message);
    void sendProjectionMessage(char *message);
    void sendDecorationMessage(char *message);
    void sendHeadlightMessage(char *message);
    void sendVRMLCameraMessage(char *message);
    void sendCameraMessage(char *message);
    void sendColormapMessage(char *message);
    void sendQuitMessage();
    void sendSequencerMessage(char *message);
    void sendQuitRequestMessage();
    void sendFinishMessage();
    void sendCSFeedback(char *key, char *message);
    void sendAnnotation(char *key, char *message);
    void sendShowHelpMessage(const char *url);

    void receiveAddObjectMessage(const coDistributedObject *obj, char *message, int doreplace);
    void addgeometry(char *object, int doreplace, int is_timestep, int timestep,
                     char *root,
                     const coDistributedObject *geometry,
                     const coDistributedObject *normals,
                     const coDistributedObject *colors,
                     const coDistributedObject *textures,
                     const coDoGeometry *container,
                     char *feedbackInStr = NULL);
    void receiveCameraMessage(char *message);
    void receiveTransformMessage(char *message);
    void receiveTelePointerMessage(char *message);
    void receiveDrawstyleMessage(char *message);
    void receiveLightModeMessage(char *message);
    void receiveSelectionMessage(char *message);
    void receiveDeselectionMessage(char *message);
    void receivePartMessage(char *message);
    void receiveReferencePartMessage(char *message);
    void receiveResetSceneMessage();
    void receiveTransparencyMessage(char *message);
    void receiveSyncModeMessage(char *message);
    void receiveFogMessage(char *message);
    void receiveAntialiasingMessage(char *message);
    void receiveBackcolorMessage(char *message);
    void receiveAxisMessage(char *message);
    void receiveClippingPlaneMessage(char *message);
    void receiveViewingMessage(char *message);
    void receiveProjectionMessage(char *message);
    void receiveHeadlightMessage(char *message);
    void receiveDecorationMessage(char *message);
    void receiveSequencerMessage(char *message);
    void receiveDeleteObjectMessage(char *message);
    void receiveDeleteAll(void);
    void receiveColormapMessage(char *message);
    void receiveUpdateMessage(char *, char *, char *);
    void receiveMasterMessage(char *, char *, char *);
    void receiveSlaveMessage(char *, char *, char *);
    void receiveMasterSlaveMessage(char *, char *, char *);
    void handleAttributes(const char *name, const coDistributedObject *obj);
    ~InvCommunication(){};
    // AW: replacing by empty objects
    void setReplace(char *oldName, char *newName);
};
#endif
