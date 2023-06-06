/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
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

#include "VOIPer.h"

#ifdef HAVE_OPAL

#include <config/CoviseConfig.h>

#define QT_CLEAN_NAMESPACE

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRTui.h>

#include <iostream>

#include <ptlib/sound.h>

using namespace std;

//!##########################//
//! VOIPer                   //
//!##########################//

VOIPer::VOIPer()
: coVRPlugin(COVER_PLUGIN_NAME)
, hOPAL_(NULL)
, currentCallToken_(NULL)
, currentIncomingCallToken_(NULL)
{
}

VOIPer::~VOIPer()
{
    cout << "VOIPer: Shutting down... ";

    OpalShutDown(hOPAL_);

    cout << "done!" << endl;

    deleteUI();
}

bool
VOIPer::init()
{
    initUI();

    // OPAL //
    //
    cout << "\nVOIPer: Initialize OPAL..." << endl;

    unsigned version = OPAL_C_API_VERSION;

    hOPAL_ = OpalInitialise(&version,
                            OPAL_PREFIX_H323 " "
                            //                                  OPAL_PREFIX_SIP   " "
                            //                                  OPAL_PREFIX_IAX2  " "
                            OPAL_PREFIX_PCSS
                                             " TraceLevel=2");
    if (hOPAL_ == NULL)
    {
        cerr << "VOIPer: Could not initialize OPAL!" << endl;
        return false;
    }

    // Settings //
    //
    OpalMessage command;
    OpalMessage *response;

    // General options //
    //
    memset(&command, 0, sizeof(command));
    command.m_type = OpalCmdSetGeneralParameters;
    //command.m_param.m_general.m_audioRecordDevice = "";
    command.m_param.m_general.m_autoRxMedia = command.m_param.m_general.m_autoTxMedia = "audio";
    //  command.m_param.m_general.m_stunServer = "stun.ekiga.net";
    //  command.m_param.m_general.m_stunServer = "stun.voxgratia.org";
    command.m_param.m_general.m_mediaMask = "RFC4175*";

    response = sendCommandAndCheck(&command, "Error VOIPer: Could not set general options");
    if (!response)
        return false;

    OpalFreeMessage(response);

    // Options across all protocols //
    //
    memset(&command, 0, sizeof(command));
    command.m_type = OpalCmdSetProtocolParameters;

    command.m_param.m_protocol.m_userName = "tabletUI";
    command.m_param.m_protocol.m_displayName = "OpenCOVER tabletUI VOIPer";
    command.m_param.m_protocol.m_interfaceAddresses = "*";

    response = sendCommandAndCheck(&command, "Error VOIPer: Could not set protocol options");
    if (!response)
        return false;

    OpalFreeMessage(response);

// Registration test //
//
#if 0
   memset(&command, 0, sizeof(command));
   command.m_type = OpalCmdRegistration;

   command.m_param.m_registrationInfo.m_protocol = "h323";
   command.m_param.m_registrationInfo.m_identifier = "00497116858721517";
   command.m_param.m_registrationInfo.m_hostName = "192.108.36.253";
   command.m_param.m_registrationInfo.m_password = "";
   command.m_param.m_registrationInfo.m_timeToLive = 300;

   cout << "\nVOIPer: Registration... ";
   response = sendCommandAndCheck(&command, "ERROR VOIPer: Registration failed.");
   if (response != NULL && response->m_type == OpalCmdRegistration)
   {
      cout << "successful." << endl;
      cout << "  AddressOfRecord: " << response->m_param.m_registrationInfo.m_identifier << "\n" << endl;
//       m_AddressOfRecord = response->m_param.m_registrationInfo.m_identifier
   }
   else
   {
      cout << "failed." << endl;
   }

   OpalFreeMessage(response);
#endif

    cout << "\nVOIPer: Initialize OPAL... done!" << endl;

    // Print devices //
    //
    PStringArray names = PSoundChannel::GetDeviceNames(PSoundChannel::Player, NULL);
    cout << "\nDEVICES" << endl;
    for (int i = 0; i < names.GetSize(); ++i)
    {
        cout << names[i] << endl;
    }

    // Start Listening //
    //
    cout << "\nVOIPer: Start listening..." << endl;
    handleMessages(5);

    return true;
}

void
VOIPer::preFrame()
{
    handleMessages(0);
}

//!##########################//
//! TABLET UI                //
//!##########################//

void
VOIPer::initUI()
{
    int row = -1;

    // Tab //
    //
    tuiTab = new coTUITab("VOIPer", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, ++row);

    (new coTUILabel("Last Message: ", tuiTab->getID()))->setPos(1, ++row);
    tuiMessageLabel_ = new coTUILabel("...", tuiTab->getID());
    tuiMessageLabel_->setPos(2, row);
    ++row;

    // Make a call //
    //
    tuiCallButton_ = new coTUIButton("Call", tuiTab->getID());
    tuiCallButton_->setEventListener(this);
    tuiCallButton_->setPos(0, ++row);

    tuiCallAddress_ = new coTUIEditField("Call Address", tuiTab->getID());
    tuiCallAddress_->setText("h323:");
    tuiCallAddress_->setPos(1, row);
    ++row;

    // Open call //
    //
    tuiHangUpButton_ = new coTUIButton("Hang up", tuiTab->getID());
    tuiHangUpButton_->setEventListener(this);
    tuiHangUpButton_->setPos(1, ++row);

    tuiOpenCallLabel_ = new coTUILabel("No open call", tuiTab->getID());
    tuiOpenCallLabel_->setPos(2, row);

    // Incoming call //
    //
    tuiAcceptCallButton_ = new coTUIButton("Accept", tuiTab->getID());
    tuiAcceptCallButton_->setEventListener(this);
    tuiAcceptCallButton_->setPos(0, ++row);

    tuiRejectCallButton_ = new coTUIButton("Reject", tuiTab->getID());
    tuiRejectCallButton_->setEventListener(this);
    tuiRejectCallButton_->setPos(1, row);

    tuiIncomingCallLabel_ = new coTUILabel("", tuiTab->getID());
    tuiIncomingCallLabel_->setLabel("No incoming call");
    tuiIncomingCallLabel_->setPos(2, row);
}

void
VOIPer::deleteUI()
{
    // UI //
    //
    delete tuiCallAddress_;
    delete tuiCallButton_;

    delete tuiMessageLabel_;

    delete tuiAcceptCallButton_;
    delete tuiRejectCallButton_;
    delete tuiIncomingCallLabel_;

    delete tuiHangUpButton_;
    delete tuiOpenCallLabel_;

    delete tuiTab;
}

void
VOIPer::tabletEvent(coTUIElement *tUIItem)
{
}

void
VOIPer::tabletPressEvent(coTUIElement *tUIItem)
{
}

void
VOIPer::tabletReleaseEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiAcceptCallButton_)
    {
        acceptIncomingCall();
    }

    else if (tUIItem == tuiRejectCallButton_)
    {
        rejectIncomingCall();
    }

    else if (tUIItem == tuiHangUpButton_)
    {
        hangUpCall();
    }

    else if (tUIItem == tuiCallButton_)
    {
        makeACall(tuiCallAddress_->getText().c_str());
    }
}

void
VOIPer::handleMessages(unsigned timeout)
{
    OpalMessage *message;

    //    while ((message = OpalGetMessage(hOPAL_, timeout)) != NULL)
    if ((message = OpalGetMessage(hOPAL_, timeout)) != NULL) // only one message per frame
    {
        switch (message->m_type)
        {
        case OpalIndRegistration:
            tuiMessageLabel_->setLabel("OpalIndRegistration");
            cout << "VOIPer: OpalIndRegistration." << endl;
            switch (message->m_param.m_registrationStatus.m_status)
            {
            case OpalRegisterRetrying:
                cout << "Trying registration to: " << message->m_param.m_registrationStatus.m_serverName << endl;
                break;
            case OpalRegisterRestored:
                cout << "Registration restored: " << message->m_param.m_registrationStatus.m_serverName << endl;
                break;
            case OpalRegisterSuccessful:
                cout << "Registration successful: " << message->m_param.m_registrationStatus.m_serverName << endl;
                break;
            case OpalRegisterRemoved:
                cout << "Unregistered: " << message->m_param.m_registrationStatus.m_serverName << endl;
                break;
            case OpalRegisterFailed:
                if (message->m_param.m_registrationStatus.m_error == NULL || message->m_param.m_registrationStatus.m_error[0] == '\0')
                {
                    cout << "Registration failed: " << message->m_param.m_registrationStatus.m_serverName << endl;
                }
                else
                {
                    cout << "Registration of %s error: " << message->m_param.m_registrationStatus.m_serverName << ", " << message->m_param.m_registrationStatus.m_error << endl;
                }
            }
            break;

        case OpalIndLineAppearance:
            tuiMessageLabel_->setLabel("OpalIndLineAppearance");
            cout << "VOIPer: OpalIndLineAppearance." << endl;
            switch (message->m_param.m_lineAppearance.m_state)
            {
            case OpalLineIdle:
                cout << "Line available: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineTrying:
                cout << "Line in use: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineProceeding:
                cout << "Line calling: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineRinging:
                cout << "Line ringing: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineConnected:
                cout << "Line connected: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineSubcribed:
                cout << "Line subscription successful: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            case OpalLineUnsubcribed:
                cout << "Unsubscribed line: " << message->m_param.m_lineAppearance.m_line << endl;
                break;
            }
            break;

        case OpalIndIncomingCall:
            tuiMessageLabel_->setLabel("OpalIndIncomingCall");
            handleIncomingCall(message);
            break;

        case OpalIndProceeding:
            tuiMessageLabel_->setLabel("OpalIndProceeding");
            cout << "VOIPer: OpalIndProceeding." << endl;
            break;

        case OpalIndAlerting:
            tuiMessageLabel_->setLabel("OpalIndAlerting");
            cout << "VOIPer: OpalIndAlerting." << endl;
            break;

        case OpalIndEstablished:
            tuiMessageLabel_->setLabel("OpalIndEstablished");
            tuiOpenCallLabel_->setLabel("Open call");
            cout << "VOIPer: OpalIndEstablished." << endl;
            break;

        case OpalIndMediaStream:
            cout << "VOIPer: OpalIndMediaStream.\n"
                 << message->m_param.m_mediaStream.m_type << " "
                 << message->m_param.m_mediaStream.m_format << " ";
            if ((message->m_param.m_mediaStream.m_state == OpalMediaStateOpen))
            {
                tuiMessageLabel_->setLabel("OpalIndMediaStream (opened)");
                cout << " (opened)" << endl;
            }
            else
            {
                tuiMessageLabel_->setLabel("OpalIndMediaStream (closed)");
                cout << " (closed)" << endl;
            }
            break;

        case OpalIndUserInput:
            tuiMessageLabel_->setLabel("OpalIndUserInput");
            cout << "VOIPer: OpalIndUserInput.\n"
                 << "User Input: " << message->m_param.m_userInput.m_userInput << endl;
            break;

        case OpalIndCallCleared:
            tuiMessageLabel_->setLabel("OpalIndCallCleared");
            tuiOpenCallLabel_->setLabel("No open call");
            cout << "VOIPer: OpalIndCallCleared." << endl;
            cout << "  " << message->m_param.m_callCleared.m_callToken << " " << currentCallToken_ << endl;

            delete currentCallToken_;
            currentCallToken_ = NULL;

            if (message->m_param.m_callCleared.m_reason == NULL)
                cout << "Call cleared." << endl;
            else
                cout << "Call cleared: " << message->m_param.m_callCleared.m_reason << endl;

            break;

        default:
            tuiMessageLabel_->setLabel("default");
            break;
        }

        OpalFreeMessage(message);
    }
}

void
VOIPer::handleIncomingCall(OpalMessage *message)
{
    printf("Incoming call from \"%s\", \"%s\" to \"%s\", handled by \"%s\". Token: %s.\n",
           message->m_param.m_incomingCall.m_remoteDisplayName,
           message->m_param.m_incomingCall.m_remoteAddress,
           message->m_param.m_incomingCall.m_calledAddress,
           message->m_param.m_incomingCall.m_localAddress,
           message->m_param.m_incomingCall.m_callToken);

    if (!currentCallToken_ && !currentIncomingCallToken_)
    {
        // No open call, not ringing => accept by pressing button //
        //
        tuiIncomingCallLabel_->setLabel("INCOMING Call!");

        OpalMessage command;
        OpalMessage *response;
        memset(&command, 0, sizeof(command));
        command.m_type = OpalCmdAlerting;
        command.m_param.m_callToken = message->m_param.m_incomingCall.m_callToken;
        if ((response = sendCommandAndCheck(&command, "VOIPer Error: Could not set state to alerting!")) != NULL)
            OpalFreeMessage(response);

        currentIncomingCallToken_ = strdup(message->m_param.m_incomingCall.m_callToken);
    }
    else
    {
        // Open call => refuse //
        //
        tuiIncomingCallLabel_->setLabel("INCOMING Call: rejected due to open call.");

        OpalMessage command;
        OpalMessage *response;
        memset(&command, 0, sizeof(command));
        command.m_type = OpalCmdClearCall;
        command.m_param.m_clearCall.m_callToken = message->m_param.m_incomingCall.m_callToken;
        command.m_param.m_clearCall.m_reason = OpalCallEndedByLocalBusy;
        if ((response = sendCommandAndCheck(&command, "VOIPer Error: Could not refuse call!")) != NULL)
            OpalFreeMessage(response);
    }
}

/*! Accept an incoming call.
*
* There must not be an open call.
* The currentIncomingCallToken_ will be accepted.
*/
void
VOIPer::acceptIncomingCall()
{
    if (!currentCallToken_ && currentIncomingCallToken_)
    {
        tuiIncomingCallLabel_->setLabel("INCOMING Call: accepting.");

        OpalMessage command;
        OpalMessage *response;
        memset(&command, 0, sizeof(command));
        command.m_type = OpalCmdAnswerCall;
        command.m_param.m_callToken = strdup(currentIncomingCallToken_);
        if ((response = sendCommandAndCheck(&command, "VOIPer Error: Could not answer call!")) != NULL)
            OpalFreeMessage(response);

        tuiIncomingCallLabel_->setLabel("No incoming call");

        currentCallToken_ = strdup(currentIncomingCallToken_);
        currentIncomingCallToken_ = NULL;
    }
}

/*! Reject an incoming call.
*
* There must not be an open call.
* The currentIncomingCallToken_ will be rejected.
*/
void
VOIPer::rejectIncomingCall()
{
    if (!currentCallToken_ && currentIncomingCallToken_)
    {
        tuiIncomingCallLabel_->setLabel("INCOMING Call: rejecting.");

        OpalMessage command;
        OpalMessage *response;
        memset(&command, 0, sizeof(command));
        command.m_type = OpalCmdClearCall;
        command.m_param.m_clearCall.m_callToken = currentIncomingCallToken_;
        command.m_param.m_clearCall.m_reason = OpalCallEndedByLocalUser; //OpalCallEndedByRemoteUser;
        if ((response = sendCommandAndCheck(&command, "VOIPer Error: Could not refuse call!")) != NULL)
            OpalFreeMessage(response);

        tuiIncomingCallLabel_->setLabel("No incoming call");
        currentIncomingCallToken_ = NULL;
    }
}

/*! Make a call.
*
*/
bool
VOIPer::makeACall(const char *to)
{
    // example: "h323:168.192.0.2"
    if (currentCallToken_ || currentIncomingCallToken_)
    {
        return 0;
    }

    OpalMessage command;
    OpalMessage *response;

    tuiOpenCallLabel_->setLabel("Calling...");
    cout << "VOIPer: Calling " << to << endl;

    memset(&command, 0, sizeof(command));
    command.m_type = OpalCmdSetUpCall;
    command.m_param.m_callSetUp.m_partyA = NULL; // = me
    command.m_param.m_callSetUp.m_partyB = to;
    if ((response = sendCommandAndCheck(&command, "Could not make call")) == NULL)
        return 0;

    currentCallToken_ = strdup(response->m_param.m_callSetUp.m_callToken);
    OpalFreeMessage(response);

    return 1;
}

/*! Hang up current call.
*
*/
void
VOIPer::hangUpCall()
{
    cerr << "FUNCTION: hang up " << currentCallToken_ << endl;

    if (currentCallToken_)
    {
        cout << "hang up " << currentCallToken_ << endl;

        OpalMessage command;
        OpalMessage *response;
        memset(&command, 0, sizeof(command));
        command.m_type = OpalCmdClearCall;
        command.m_param.m_clearCall.m_callToken = currentCallToken_;
        command.m_param.m_clearCall.m_reason = OpalCallEndedByLocalUser; //OpalCallEndedByRemoteUser;
        if ((response = sendCommandAndCheck(&command, "VOIPer Error: Could not hang up call!")) != NULL)
            OpalFreeMessage(response);
    }
}

OpalMessage *
VOIPer::sendCommandAndCheck(OpalMessage *command, const char *errorMessage)
{
    OpalMessage *response = OpalSendMessage(hOPAL_, command);
    if (!response)
    {
        return NULL; // response NULL => bad
    }

    if (response->m_type != OpalIndCommandError)
    {
        return response; // response OK
    }

    // Print Error Message //
    //
    if (response->m_param.m_commandError == NULL || *response->m_param.m_commandError == '\0')
    {
        cout << errorMessage << endl;
    }
    else
    {
        cout << errorMessage << " " << response->m_param.m_commandError;
    }
    OpalFreeMessage(response);
    return NULL;
}

COVERPLUGIN(VOIPer)

#endif
