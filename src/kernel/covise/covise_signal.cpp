/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Signal manager class for the COVISE                       **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                               (C) 1995                                 **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author: D. Rantzau, A. Wierse                                          **
 **                                                                        **
 ** Date:  12.12.95  V1.0                                                  **
 ** Last:                                                                  **
\**************************************************************************/

#include "covise.h"

#include "covise_signal.h"

using namespace covise;

void *SignalHandler::handler[65] = {
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL
};

void *SignalHandler::userdata[65] = {
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL
};

SignalHandler::SignalHandler()
{
#ifndef _WIN32
    sigemptyset(&newmask);

    tmp_env = getenv("COVISE_SIGNAL_ALL");
    if (tmp_env && strcasecmp(tmp_env, "1") == 0)
        use_signals = 1;
    else if (tmp_env && strcasecmp(tmp_env, "OFF") == 0)
        use_signals = 0;
    else
        use_signals = 0;
#else
    use_signals = 0;
#endif
}

SignalHandler::~SignalHandler()
{
}

#ifndef _WIN32
void *SignalHandler::doHandle(int sig)
{

    //((SigFunctVoid*)(handler[sig]))(sig,userdata[sig]);
    ((void (*)(int, void *))(handler[sig]))(sig, userdata[sig]);

    return handler[sig];
}

void SignalHandler::addSignal(int sig, void *myHandler, void *udata)
{

    if (use_signals)
    {
        sigNr = sig;
        oldData = userdata[sig];
        userdata[sig] = udata;
        handler[sig] = myHandler;
        sigaddset(&newmask, sig);
        my_signal(sig, (void (*)(int))myHandler);
    }
}

void SignalHandler::removeSignal(int sig)
{

    if (use_signals)
    {
        signal(sig, NULL);
    }
}

void SignalHandler::blockSignals(void)
{

    if (use_signals)
    {
        sigprocmask(SIG_BLOCK, &newmask, &oldmask);
    }
}

void SignalHandler::unblockSignals(void)
{

    if (use_signals)
    {
        sigprocmask(SIG_SETMASK, &oldmask, NULL);
    }
}

int SignalHandler::isPending(int sig)
{

    if (use_signals)
    {
        if (sigpending(&pendmask) < 0)
        {
            return 0;
        }
        else
        {
            if (sigismember(&pendmask, sig))
                return 1;
            else
                return 0;
        }
    }
    else
        return 0;
}

namespace covise
{
// SigFunctVoid *my_signal(int signo, SigFunct *func)
//void (*my_signal (int signo, void (*func)(...)))(...)
void (*my_signal(int signo, void (*func)(int)))(int)
{
    struct sigaction act, old_act;

#ifdef _SX
    act.sa_handler = (void (*)())func;
#else
    act.sa_handler = func;
#endif
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
#if !defined(__hpux) && !defined(_SX)
    act.sa_flags |= SA_RESTART;
#endif
#if defined(_SX) || defined(__hpux)
    if (sigaction(signo, &act, &old_act) < 0)
        return (void (*)(int))SIG_ERR;
    return (void (*)(int))(old_act.sa_handler);
#else
    if (sigaction(signo, &act, &old_act) < 0)
        return SIG_ERR;
    return (old_act.sa_handler);
#endif
}
}
#endif
