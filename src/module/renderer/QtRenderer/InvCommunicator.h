/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_COMMUNICATION_H
#define _INV_COMMUNICATION_H

//

//
// ec stuff
//
#include <covise/covise_process.h>
#include <do/coDoGeometry.h>

// Open Inventor
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSwitch.h>

using namespace covise;

//================================================================
// InvCommunicator
//================================================================

class InvCommunicator
{
private:
    int getNextPK();
    int getCurrentPK();
    int pk_;

    QString buffer;
    void sendMSG(QString);

public:
    InvCommunicator();
    void sendTransformMessage(QString);
    void sendTelePointerMessage(const char *);
    void sendVRMLTelePointerMessage(QString);
    void sendDrawstyleMessage(QString);
    void sendLightModeMessage(QString);
    void sendSelectionMessage(QString);
    void sendDeselectionMessage(QString);
    void sendPartMessage(QString);
    void sendReferencePartMessage(QString);
    void sendResetSceneMessage();
    void sendTransparencyMessage(QString);
    void sendSyncModeMessage(QString);
    void sendFogMessage(QString);
    void sendAntialiasingMessage(QString);
    void sendBackcolorMessage(QString);
    void sendAxisMessage(QString);
    void sendClippingPlaneMessage(QString);
    void sendViewingMessage(QString);
    void sendProjectionMessage(QString);
    void sendDecorationMessage(QString);
    void sendHeadlightMessage(QString);
    void sendVRMLCameraMessage(QString);
    void sendCameraMessage(const char *);
    void sendColormapMessage(QString);
    void sendQuitMessage();
    void sendSequencerMessage(QString);
    void sendQuitRequestMessage();
    void sendFinishMessage();
    void sendCSFeedback(const char *key, QString);
    void sendAnnotation(const char *key, QString);

    void addgeometry(const char *object, int doreplace, int is_timestep, int timestep, const char *root,
                     const coDistributedObject *geometry, const coDistributedObject *normals,
                     const coDistributedObject *colors, const coDistributedObject *textures, const coDoGeometry *container);

    void receiveAddObjectMessage(const coDistributedObject *obj, const char *object, int doreplace);
    void receiveCameraMessage(QString);
    void receiveTransformMessage(QString);
    void receiveTelePointerMessage(QString);
    void receiveDrawstyleMessage(QString);
    void receiveLightModeMessage(QString);
    void receiveSelectionMessage(QString);
    void receiveDeselectionMessage(QString);
    void receivePartMessage(QString);
    void receiveReferencePartMessage(QString);
    void receiveResetSceneMessage();
    void receiveTransparencyMessage(QString);
    void receiveSyncModeMessage(QString);
    void receiveFogMessage(QString);
    void receiveAntialiasingMessage(QString);
    void receiveBackcolorMessage(const char *);
    void receiveAxisMessage(const char *);
    void receiveClippingPlaneMessage(const char *);
    void receiveViewingMessage(QString);
    void receiveProjectionMessage(QString);
    void receiveHeadlightMessage(QString);
    void receiveDecorationMessage(QString);
    void receiveSequencerMessage(QString);
    void receiveDeleteObjectMessage(QString);
    void receiveDeleteAll();
    void receiveColormapMessage(QString);
    void receiveUpdateMessage(QString, QString, QString);
    void receiveMasterMessage(QString);
    //void receiveMasterMessage(QString, QString, QString);
    void receiveSlaveMessage(QString);
    void receiveMasterSlaveMessage(QString, QString, QString);
    ~InvCommunicator(){};

    // AW: replacing by empty objects
    void setReplace(QString oldName, QString newName);
};
#endif
