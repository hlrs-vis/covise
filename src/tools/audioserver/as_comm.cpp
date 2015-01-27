/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_comm.cpp
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                           Date: 05/05/2002
 *
 *  Purpose : Communication processing
 *
 *********************************************************************
 */

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <windows.h>

#include <windows.h>
#include <assert.h>
#include <errno.h>

#include <iostream>
#include <string>
#include <stdio.h>

#include <mmsystem.h>

#include "common.h"

#include "as_comm.h"
#include "as_cache.h"

#include "as_ServerSocket.h"

#include "as_client_api.h"

static unsigned long cmdCtr = 0;

as_Comm::as_Comm(HWND hWnd)
{
    int i;
    this->dataExpected = 0;
    this->dataReceived = 0;

    // allocate command parameter memory
    for (i = 0; i < MAX_PARAMS; i++)
    {
        params[i] = (char *)malloc(sizeof(char) * MAX_BUFLEN);
        if (NULL == params[i])
        {
            AddLogMsg("Out of memory error!");
            return;
        }
    }
}

as_Comm::~as_Comm()
{
    int i;
    // deallocate command parameter memory
    for (i = 0; i < MAX_PARAMS; i++)
    {
        free(params[i]);
    }
}

long as_Comm::decodeCommand(char *cmd, int numParams)
{
    char msg[MSG_LEN];
    as_Sound *pSound = NULL;
    int soundParamCount = numParams - 1;
    long handle;

#ifdef DEBUG
    int u;
    // print command and parameters
    sprintf(msg, "-> Command '%s'", cmd);
    AddLogMsg(msg);
    if (0 < numParams)
    {
        sprintf(msg, "  %d parameter(s):", numParams);
        AddLogMsg(msg);
        for (u = 0; u < numParams; u++)
        {
            sprintf(msg, "  %2d: %s", u, params[u]);
            AddLogMsg(msg);
        }
    }
#endif

    if (numParams > 0)
    {
        // read handle number
        if (EOF == sscanf(params[0], "%ld", &handle))
        {
            sprintf(msg, "  # SetSoundParameter: invalid handle parameter");
            AddLogMsg(msg);
            return -1;
        }
    }
    // --- set sound parameter ------------------------------------------------------------
    /*
      // decode first because of probable high repetiveness of such commands
      if ( 0 == strcmp(CMD_SET_SOUND, cmd)) {
         long handle;
   //		char * soundParamName;
   //		int soundParamCount = numParams-2;

         if (0 >= soundParamCount) {
            sprintf(msg, "  # SetSoundParameter: wrong params, must be: <handle> <parameter name> <parameter>...");
            AddLogMsg(msg);
            return -1;
   }

   if ( EOF == sscanf(params[0], "%ld", &handle))
   {
   sprintf(msg, "  # SetSoundParameter: invalid handle parameter");
   AddLogMsg(msg);
   return -1;
   }

   //		soundParamName = params[1];

   }
   */
    // set sound volume and source direction relative to listener position (VRML)
    if ((0 == strcmp(SOUND_REL_DIRECTION_VOLUME, cmd)) || (0 == strcmp(SOUND_DIRECTION_VOLUME, cmd)))
    {

        if (2 == soundParamCount)
        {
            float angle;
            float volume;

            if ((EOF == sscanf(params[1], "%f", &angle)))
            {
                sprintf(msg, "   # SOUND_REL_DIRECTION_VOLUME: wrong params");
                AddLogMsg(msg);
                return -1;
            }
            if ((EOF == sscanf(params[2], "%f", &volume)))
            {
                sprintf(msg, "   # SOUND_REL_DIRECTION_VOLUME: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
            {
                pSound->SetGain(volume);
                pSound->SetDirectionRelative(angle, AS_Control->getHandleColor(handle));
                //					pSound->SetDirectionRelative(angle);
            }

            return 0;
        }

        if (4 == soundParamCount)
        {
            float x, y, z;
            float volume;

            if ((EOF == sscanf(params[1], "%f", &x))
                || (EOF == sscanf(params[2], "%f", &y))
                || (EOF == sscanf(params[3], "%f", &z)))
            {
                sprintf(msg, "    # SOUND_REL_DIRECTION_VOLUME: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            if ((EOF == sscanf(params[4], "%f", &volume)))
            {
                sprintf(msg, "   # SOUND_REL_DIRECTION_VOLUME: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
            {
                pSound->SetGain(volume);
                pSound->SetDirectionRelative(x, y, z, AS_Control->getHandleColor(handle));
                //					pSound->SetDirectionRelative(x, y, z);
            }

            return 0;
        }

        // wrong param count
        sprintf(msg, "    # SetSoundDirectionRelative: wrong param count");
        AddLogMsg(msg);
        return -1;
    }

    // set sound source position
    if (0 == strcmp(SOUND_POSITION, cmd))
    {
        float x, y, z;

        if (3 != soundParamCount)
        {
            sprintf(msg, "    # SetSoundPos: wrong param count");
            AddLogMsg(msg);
            return -1;
        }

        if ((EOF == sscanf(params[1], "%f", &x))
            || (EOF == sscanf(params[2], "%f", &y))
            || (EOF == sscanf(params[3], "%f", &z)))
        {
            sprintf(msg, "    # SetSoundPos: wrong params");
            AddLogMsg(msg);
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
        {
            pSound->SetPosition(x, y, z);
            //updateGridColored(x, y, z, AS_Control->getHandleColor(handle));
            pSound->updateGridColored();
        }
        return 0;
    }

    // set sound source gain
    if (0 == strcmp(SOUND_VOLUME, cmd))
    {
        float gain;

        if (1 != soundParamCount)
        {
            sprintf(msg, "    # SetSoundGain: wrong param count");
            AddLogMsg(msg);
            return -1;
        }

        if ((EOF == sscanf(params[1], "%f", &gain)))
        {
            sprintf(msg, "    # SetSoundGain: wrong params");
            AddLogMsg(msg);
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
            pSound->SetGain(gain);

        return 0;
    }

    // set sound source velocity
    if (0 == strcmp(SOUND_VELOCITY, cmd))
    {
        float velocity;

        if (1 != soundParamCount)
        {
            sprintf(msg, "    # SetSoundVelocity: wrong param count");
            AddLogMsg(msg);
            return -1;
        }

        if ((EOF == sscanf(params[1], "%f", &velocity)))
        {
            sprintf(msg, "    # SetSoundVelocity: wrong params");
            AddLogMsg(msg);
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
            pSound->SetVelocity(velocity);

        return 0;
    }

    // set sound source direction
    if (0 == strcmp(SOUND_DIRECTION, cmd))
    {

        if (1 == soundParamCount)
        {
            float angle;

            if ((EOF == sscanf(params[1], "%f", &angle)))
            {
                sprintf(msg, "   # SetSoundDirection: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
            {
                pSound->SetDirection(angle);
            }
            return 0;
        }

        if (3 == soundParamCount)
        {
            float x, y, z;

            if ((EOF == sscanf(params[1], "%f", &x))
                || (EOF == sscanf(params[2], "%f", &y))
                || (EOF == sscanf(params[3], "%f", &z)))
            {
                sprintf(msg, "    # SetSoundDirection: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
                pSound->SetDirection(x, y, z);

            return 0;
        }

        // wrong param count
        sprintf(msg, "    # SetSoundDirection: wrong param count");
        AddLogMsg(msg);
        return -1;
    }

    // set sound source direction relative to listener position (VRML)
    if (0 == strcmp(SOUND_REL_DIRECTION, cmd))
    {

        if (1 == soundParamCount)
        {
            float angle;

            if ((EOF == sscanf(params[1], "%f", &angle)))
            {
                sprintf(msg, "   # SetSoundDirectionRelative: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
            {
                pSound->SetDirectionRelative(angle, AS_Control->getHandleColor(handle));
                //					pSound->SetDirectionRelative(angle);
            }

            return 0;
        }

        if (3 == soundParamCount)
        {
            float x, y, z;

            if ((EOF == sscanf(params[1], "%f", &x))
                || (EOF == sscanf(params[2], "%f", &y))
                || (EOF == sscanf(params[3], "%f", &z)))
            {
                sprintf(msg, "    # SetSoundDirectionRelative: wrong params");
                AddLogMsg(msg);
                return -1;
            }

            pSound = AS_Control->getSoundByHandle(handle);
            if (pSound)
            {
                pSound->SetDirectionRelative(x, y, z, AS_Control->getHandleColor(handle));
                //					pSound->SetDirectionRelative(x, y, z);
            }

            return 0;
        }

        // wrong param count
        sprintf(msg, "    # SetSoundDirectionRelative: wrong param count");
        AddLogMsg(msg);
        return -1;
    }

    // set looping on / off
    if (0 == strcmp(SOUND_LOOPING, cmd))
    {
        int looping;

        if (1 != soundParamCount)
        {
            sprintf(msg, "   # SetSoundLooping: wrong param count");
            AddLogMsg(msg);
            return -1;
        }

        if ((EOF == sscanf(params[1], "%d", &looping)))
        {
            sprintf(msg, "   # SetSoundLooping: wrong params");
            AddLogMsg(msg);
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);

        if (pSound)
        {
            if (1 == looping)
            {
                pSound->SetLooping(true);
            }
            else
            {
                pSound->SetLooping(false);
            }
        }

        return 0;
    }

    // set pitch
    if (0 == strcmp(SOUND_PITCH, cmd))
    {
        float pitch;

        if (1 != soundParamCount)
        {
            sprintf(msg, " SetSoundPitch: wrong param count");
            AddLogMsg(msg);
            return -1;
        }

        if ((EOF == sscanf(params[1], "%f", &pitch)))
        {
            sprintf(msg, "   # SetSoundPitch: wrong params");
            AddLogMsg(msg);
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
            pSound->SetPitch(pitch);

        return 0;
    }

    // -- Sound Test ---
    if (0 == strcmp(CMD_TEST, cmd))
    {

        if (1 == numParams)
        {
            AS_Control->test(atol(params[0]));
        }
        else
        {
            AS_Control->test(0);
        }
        return 0;
    }

    // set volume
    if (0 == strcmp(CMD_SET_VOLUME, cmd))
    {
        switch (numParams)
        {
        case 2:
            AS_Control->setVolume(atol(params[0]), atol(params[1]));
            return 0;
        default:
            AddLogMsg("Set volume: wrong parameters");
            return -1;
        }
    }

    // --- get file handle ---
    if (0 == strcmp(CMD_GET_HANDLE, cmd))
    {
        switch (numParams)
        {
        case 1:
        {
            long handle = -1;
            handle = AS_Control->newHandle(params[0]);
            AS_Communication->sendHandle(handle);
#ifdef DEBUG
            sprintf(msg, "   * GetFileHandle -> %d", handle);
            AddLogMsg(msg);
#endif
            return 0;
        }
        default:
            AddLogMsg("  # GetFileHandle: wrong parameters");
            return -1;
        }
        return 0;
    }

    // --- release handle ---
    if (0 == strcmp(CMD_RELEASE_HANDLE, cmd))
    {
        long handle;

        if (0 == numParams)
        {
            AddLogMsg("  # ReleaseHandle: wrong parameters");
            return -1;
        }
        handle = atol(params[0]);

        if (-1 == handle)
        {
            AddLogMsg("  # ReleaseHandle: invalid handle");
            return -1;
        }
        AS_Control->releaseHandle(handle);
        return 0;
    }

    // --- start playing sound ---
    if (0 == strcmp(CMD_PLAY, cmd))
    {
        long handle;
        float starttime;
        unsigned long starttime_ms = 0;
        float stoptime;
        unsigned long stoptime_ms = 0;

        handle = atol(params[0]);
        if (-1 == handle)
        {
            AddLogMsg("  # Play: invalid handle");
            return -1;
        }

        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
        {
            if (2 < numParams)
            {
                starttime = atof(params[1]);
                stoptime = atof(params[2]);
                pSound->Play(starttime, stoptime);
            }
            else if (2 == numParams)
            {
                starttime = atof(params[1]);
                pSound->Play(starttime);
            }
            else if (1 == numParams)
            {
                pSound->Play();
            }
            else
            {
                AddLogMsg("  # Play: wrong parameters");
                return -1;
            }
        }
        return 0;
    }

    // --- stop playing sound ---
    if (0 == strcmp(CMD_STOP, cmd))
    {
        long handle;

        if (0 == numParams)
        {
            AddLogMsg("  # Stop: wrong parameters");
            return -1;
        }
        handle = atol(params[0]);

        if (-1 == handle)
        {
            AddLogMsg("  # Stop: invalid handle");
            return -1;
        }
        pSound = AS_Control->getSoundByHandle(handle);
        if (pSound)
            pSound->Stop();
        return 0;
    }

    // --- get sound file ---
    if (0 == strcmp(CMD_PUT_FILE, cmd))
    {

        unsigned long size;

        if (2 != numParams)
        {
            sprintf(msg, "  # PutFile: wrong params!");
            AddLogMsg(msg);
            return -1;
        }

        if ((0 == strcmp("", params[0]))
            || (EOF == sscanf(params[1], "%lu", &size)))
        {
            sprintf(msg, "  # PutFile: wrong params!");
            AddLogMsg(msg);
            return -1;
        }

        if (dataExpected != 0)
        {
            AddLogMsg("Already waiting for file data!");
            return -1;
        }
        this->dataExpected = size;
        this->dataReceived = 0;

        sprintf(msg, "  - PutFile params: filename=%s size=%ld", params[0], size);
        AddLogMsg(msg);
        SetStatusMsg("Receiving file data...");

        return AS_Cache->newFile(params[0], size);
    }

    // sync command
    if (0 == strcmp(CMD_SYNC, cmd))
    {
        AddLogMsg("  Sync");
        AS_Communication->sendSync();
        return 0;
    }

    /*
      // --- set EAX parameter -------------------------------------------------
      if ( 0 == strcmp(CMD_SET_EAX, cmd)) {
         char * EAXParamName;
         int EAXParamCount = numParams-1;

         if (0 >= EAXParamCount) {
            sprintf(msg, "Set EAX <parameter name> <parameter>...");
            AddLogMsg(msg);
            return -1;
         }

   EAXParamName = params[0];

   sprintf(msg, "Set EAX: %s", EAXParamName);
   AddLogMsg(msg);

   // set environment
   if ( 0 == strcmp(EAX_ENVIRONMENT, EAXParamName)) {
   unsigned int numEAXEnv;

   if (1 != EAXParamCount) {
   sprintf(msg, "Set EAX environment: wrong param count");
   AddLogMsg(msg);
   return -1;
   }

   if (( EOF == sscanf(params[1], "%d", &numEAXEnv)))
   {
   sprintf(msg, "Set EAX environment: wrong params");
   AddLogMsg(msg);
   return -1;
   }

   if (numEAXEnv > EAX_ENVIRONMENT_COUNT) {
   sprintf(msg, "Set EAX environment: invalid value %d (must be < %d)",
   numEAXEnv, EAX_ENVIRONMENT_COUNT);
   AddLogMsg(msg);
   return -1;
   }
   sprintf(msg, "Set EAX environment %d", numEAXEnv);
   AddLogMsg(msg);

   if (g_paudio) g_paudio->SetEnvironment(numEAXEnv);

   return 0;
   }

   // no known parameter
   sprintf(msg, "Unknown EAX parameter: %s", EAXParamName);
   AddLogMsg(msg);

   return -1;
   }

   // --- set Sound EAX parameter -------------------------------------------------
   if ( 0 == strcmp(CMD_SET_SOUND_EAX, cmd)) {
   long handle;
   char * EAXParamName;
   int EAXParamCount = numParams-2;

   if (0 >= EAXParamCount) {
   sprintf(msg, "Set EAX <handle> <parameter name> <parameter>...");
   AddLogMsg(msg);
   return -1;
   }

   if ( EOF == sscanf(params[0], "%ld", &handle))
   {
   sprintf(msg, "Set EAX: invalid handle parameter");
   AddLogMsg(msg);
   return -1;
   }

   EAXParamName = params[1];

   sprintf(msg, "Set Sound EAX: %s", EAXParamName);
   AddLogMsg(msg);

   // set parameters here ...

   // no known parameter
   sprintf(msg, "Unknown EAX sound parameter: %s", EAXParamName);
   AddLogMsg(msg);

   return -1;
   }
   */

    // Client exits
    if (0 == strcmp(CMD_QUIT, cmd))
    {
        AddLogMsg("Quit: Closing connection by request.");
        AS_Server->disconnectClient();
        return -1;
    }

    // no known command
    sprintf(msg, "Unknown command: %s", cmd);
    AddLogMsg(msg);

    return 0;
}

long as_Comm::processData(char *buf, long bufLen)
{
    char msg[MSG_LEN];
    //	char seps[];
    long dataRemaining;

    long dataRest;
    long dataToWrite;

    if (0 >= bufLen)
    {
        sprintf(msg, "No data!");
        AddLogMsg(msg);
        return -1;
    }

    //	sprintf(msg, "Data block received, %d blocks", bufLen);
    //	AddLogMsg(msg);

    dataRest = this->dataExpected - this->dataReceived;

    //	sprintf(msg, "Data block belongs to file");
    //	AddLogMsg(msg);

    if (bufLen <= dataRest)
    {
        // got buffer with data only
        dataReceived += bufLen;
        dataToWrite = bufLen;
    }
    else
    {
        // bufLen > dataRest
        // got last data block, but buffer contains other stuff, maybe a command
        dataReceived += dataRest;
        dataToWrite = dataRest;
    }
    //	sprintf(msg, "Write %d blocks to file", dataToWrite);
    //	AddLogMsg(msg);

    dataRemaining = AS_Cache->writeToFile(buf, dataToWrite);

    if (dataReceived == this->dataExpected)
    {
        SetStatusMsg("");
        AddLogMsg("* File data complete");
        this->dataExpected = 0;
        this->dataReceived = 0;
    }

    //	sprintf(msg, " - %d blocks remaining", dataRemaining);
    //	AddLogMsg(msg);

    return dataRemaining;
}

long as_Comm::tokenizeMessage(char *buf, long bufLen)
{
    char msg[MSG_LEN];

    char cmd[32]; // buffer for command
    char *token;
    char *lastChar;
    int lastPos = 0;
    int len = 0;
    int isString = 0;
    int isFloat = 0;
    char arg[256] = "";
    int numArgs = 0;
    long u;
    int paramCount = 0;
    int rc;

#ifdef LOGFILE
    REFERENCE_TIME time0 = 0;
    REFERENCE_TIME time1 = 0;
    REFERENCE_TIME time2 = 0;
    REFERENCE_TIME diffTime0 = 0;
    REFERENCE_TIME diffTime1 = 0;

    g_pReferenceClock->GetTime(&time0);
#endif

    cmdCtr++;

    // start at pos 1
    // search command string for next space, comma or <cr>
    // first keyword is command
    // go to next char which is not a space
    // if next char is " then this is the begin of a string,
    //  go until next " which marks the end of the string

    if (0 >= bufLen)
    {
        sprintf(msg, "No message!");
        AddLogMsg(msg);
        return -1;
    }

    //	sprintf(msg, "Command message received, %d blocks", bufLen);
    //	AddLogMsg(msg);

    // --- command processing -----------------
    token = buf;

    //get command and params
    len = strlen(token);

    // remove tailing '\n'
    if (buf[len] == '\n')
        buf[len] = '\0';

    lastChar = token;
    lastPos = 0;
    isString = 0;
    isFloat = 0;
    numArgs = 0;
    paramCount = 0;
    for (u = 0; u < len; u++)
    {
        // string delimiter
        if ('"' == token[u])
        {
            if (1 == isString)
            {
                // end of string
                isString = 0;
            }
            else
            {
                // begin of string
                isString = 1;
                lastChar = &(token[u]);
                lastPos = u;
            }
        }

        // parameter delimiter
        if ((' ' == token[u]) || (',' == token[u]) || (u == len - 1))
        {

            if (1 == isString)
            {
                // parameter delimiter not valid inside string
                break;
            }

            if (u == len - 1)
            {
                // end of line
                //strncpy(arg, lastChar, u-lastPos+1);
                sprintf(arg, lastChar);
                //buf[u-lastPos+1] = '\0';
            }
            else
            {
                token[u] = '\0';
                sprintf(arg, lastChar);
            }

            lastChar = &(token[u + 1]);
            lastPos = u + 1;

            if (0 == numArgs)
            {
                // first token is command
                strncpy(cmd, token, lastPos);
                cmd[lastPos] = '\0';
            }
            else
            {
                // following tokens are parameters
                sprintf(this->params[paramCount], "%s", arg);
                paramCount++;
            }
            numArgs++;
        }
    }

#ifdef LOGFILE
    g_pReferenceClock->GetTime(&time1);

    diffTime0 = (time1 - time0) / 10;
#endif

    rc = decodeCommand(cmd, paramCount);

#ifdef LOGFILE
    g_pReferenceClock->GetTime(&time2);

    diffTime1 = (time2 - time0) / 10;

    fprintf(logfile, "tokenize: %08ld\n", diffTime0);
    fprintf(logfile, "                decodeCommand: %08ld\n", diffTime1);
#endif

    return rc;
}

/*
void as_Comm::eventHandler( WPARAM wParam, LPARAM lParam)
{
// AS_Server->eventHandler(wParam, lParam);
}
*/
int as_Comm::sendHandle(long handle)
{
    char msg[MSG_LEN];
    char buf[256];

    sprintf(buf, "%ld\n", handle);

    sprintf(msg, "Sending handle: %d", handle);
    AddLogMsg(msg);

    AS_Server->sendMessage(buf);
    return 0;
}

bool as_Comm::sendSync()
{
    char msg[MSG_LEN];
    char buf[256];

    sprintf(buf, "SYNC\n");
    AS_Server->sendMessage(buf);
    sprintf(msg, "Sent SYNC");
    AddLogMsg(msg);
    return 0;
}
