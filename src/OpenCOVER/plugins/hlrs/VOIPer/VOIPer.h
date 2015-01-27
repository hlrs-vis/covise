/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef VOIPER_H
#define VOIPER_H
/****************************************************************************\
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: VOIPer Plugin                                               **
 **                                                                          **
 **                                                                          **
 ** Author: Frank Naegele                                                    **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#define HAVE_OPAL

#ifdef HAVE_OPAL

#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>

#include <string>

// OPAL //
//
#include <opal.h>

namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
}

using namespace covise;
using namespace opencover;

class VOIPer : public coVRPlugin, public coTUIListener
{

    //!##########################//
    //! Functions                //
    //!##########################//

public:
    VOIPer();
    virtual ~VOIPer();

    bool init();

    void preFrame();

protected:
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);

private:
    void initUI();
    void deleteUI();

    OpalMessage *sendCommandAndCheck(OpalMessage *command, const char *errorMessage);

    void handleMessages(unsigned timeout);

    bool makeACall(const char *to);
    void hangUpCall();

    void handleIncomingCall(OpalMessage *message);
    void acceptIncomingCall();
    void rejectIncomingCall();

    //!##########################//
    //! Members                  //
    //!##########################//

private:
    // OPAL //
    //
    OpalHandle hOPAL_; // typedef OpalHandleStruct * OpalHandle
    char *currentCallToken_;

    // TUI //
    //
    coTUITab *tuiTab;

    // Status message //
    //
    coTUILabel *tuiMessageLabel_;

    // Make a call //
    //
    coTUIButton *tuiCallButton_;
    coTUIEditField *tuiCallAddress_;

    // Incoming call //
    //
    char *currentIncomingCallToken_;

    coTUIButton *tuiAcceptCallButton_;
    coTUIButton *tuiRejectCallButton_;
    coTUILabel *tuiIncomingCallLabel_;

    // Open call //
    //
    coTUIButton *tuiHangUpButton_;
    coTUILabel *tuiOpenCallLabel_;
};

#endif
#endif