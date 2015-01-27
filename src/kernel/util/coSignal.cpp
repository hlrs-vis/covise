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

#ifdef _AIX
#include <strings.h>
#endif

#include <stdlib.h>
#include <string.h>

#undef VERBOSE

#include "coSignal.h"

#include <iostream>
using std::cerr;
using std::endl;

using namespace covise;

/// Singleton with late construction: construct with first getHandler() call
coSignal *coSignal::s_instance = NULL;

/// static variables
coSignal::handlerRec *coSignal::s_handler[NSIG];

#ifdef USE_BSD_SIGNALS
int coSignal::s_blockMask[NSIG];
#else
sigset_t coSignal::s_sysMask;
#endif

bool coSignal::s_callSys[NSIG];

// list all signal names here. For signals with multiple names
// separate with blank
#if defined(__hpux) || defined(__sun) || defined(_WIN32) || defined(__APPLE__)
#ifndef _WIN32
#include <sys/signal.h>
#endif
const char *coSignal::sigName[NSIG] = {
    "SIG0",
    //  1 -  5
    "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP",
    //  6 -  9
    "SIGIOT SIGABRT", "SIGEMT", "SIGFPE", "SIGKILL",
    // 10 - 14
    "SIGBUS", "SIGSEGV", "SIGSYS", "SIGPIPE", "SIGALRM",
    // 15 - 18
    "SIGTERM", "SIGUSR1", "SIGUSR2", "SIGCLD SIGCHLD",
    // 19 - 22
    "SIGPWR", "SIGWINCH", "SIGURG", "SIGPOLL SIGIO"
#ifndef _WIN32
    ,
    // 23 - 27
    "SIGSTOP", "SIGTSTP", "SIGCONT", "SIGTTIN", "SIGTTOU",
    "SIGVTALRM", "SIGPROF", "SIGXCPU", "SIGXFSZ" // 28 - 32
#endif
};
#else
const char *coSignal::sigName[NSIG] = {
    "SIG0",
    //  1 -  5
    "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP",
    //  6 -  9
    "SIGIOT SIGABRT", "SIGEMT", "SIGFPE", "SIGKILL",
    // 10 - 14
    "SIGBUS", "SIGSEGV", "SIGSYS", "SIGPIPE", "SIGALRM",
    // 15 - 18
    "SIGTERM", "SIGUSR1", "SIGUSR2", "SIGCLD SIGCHLD",
    // 19 - 22
    "SIGPWR", "SIGWINCH", "SIGURG", "SIGPOLL SIGIO",
    // 23 - 27
    "SIGSTOP", "SIGTSTP", "SIGCONT", "SIGTTIN", "SIGTTOU",
    // 28 - 32
    "SIGVTALRM", "SIGPROF", "SIGXCPU", "SIGXFSZ", "SIG32",
    "SIG33", "SIG34", "SIG35", "SIG36", "SIG37", "SIG38",
    "SIG39", "SIG40", "SIG41", "SIG42", "SIG43", "SIG44",
    "SIG45", "SIG46", "SIG47", "SIG48", "SIGRTMIN"
                                        "SIG50",
    "SIG51", "SIG52", "SIG53", "SIG54", "SIG55",
    "SIG56", "SIG57", "SIG58", "SIG59", "SIG60", "SIG61",
    "SIG62", "SIG63", "SIGRTMAX"
};
#endif

coSignal::coSignal()
{
#ifdef VERBOSE
    cerr << "coSignal::coSignal()" << endl;
#endif

    // Initialize the handler field
    int sig;
    for (sig = 0; sig < NSIG; sig++)
    {
        s_handler[sig] = NULL;
        s_callSys[sig] = true;
#ifdef USE_BSD_SIGNALS
        s_blockMask[sig] = 0;
#endif
    }

#ifndef USE_BSD_SIGNALS
    // save the system's blocking mask
    sigemptyset(&s_sysMask);
    sigprocmask(SIG_SETMASK, NULL, &s_sysMask);
#endif
}

coSignal::~coSignal()
{

    int sig;
    for (sig = 0; sig < NSIG; sig++)
    {
        // if we installed out handler, we have to cancel it
        if (s_handler[sig])
        {
#ifdef USE_BSD_SIGNALS
            signal(sig, SIG_DFL);
#else
            struct sigaction defaultAct;
            defaultAct.sa_handler = SIG_DFL;
            defaultAct.sa_flags = 0;
            sigemptyset(&defaultAct.sa_mask);
            if (sigaction(sig, &defaultAct, NULL))
                cerr << "Errors re-installing default handler for "
                     << sigName[sig] << endl;
#ifdef VERBOSE
            else
                cerr << "Re-installed default handler for " << sigName[sig] << endl;
#endif
#endif

            // remove all handler Records from list
            struct handlerRec *hRec, *next;
            hRec = s_handler[sig];
            while (hRec)
            {
                next = hRec->next;
                delete hRec;
                hRec = next;
            }
        }
#ifndef USE_BSD_SIGNALS
        // restore the system's blocking mask
        sigprocmask(SIG_SETMASK, &s_sysMask, NULL);
#endif
    }
}

void coSignal::addSignal(int sig, coSignalHandler &myHandler,
                         enum ReplaceOption sysOpt, bool atEnd)
{
    if (!s_instance)
        s_instance = new coSignal;

    if (sysOpt == REPLACE_DFL)
        s_callSys[sig] = false;

    // create a new handler record
    handlerRec *hRec = new handlerRec;
    hRec->handler = &myHandler;

    if (s_handler[sig]) /// there ate handlers for this signal
    {
        if (atEnd) /// chain behind
        {
            handlerRec *last = s_handler[sig];
            while (last->next)
                last = last->next;
            last->next = hRec;
            hRec->next = NULL;
        }
        else /// chain before
        {
            hRec->next = s_handler[sig];
            s_handler[sig] = hRec;
        }
    }
    else /// this is the first handler
    {
        s_handler[sig] = hRec;
        hRec->next = NULL;
    }

// now install the signal handler
#ifdef USE_BSD_SIGNALS
    if (signal(sig, coSignal::doHandle) == SIG_ERR)
        cerr << "Errors installing coSignal BSD handler for"
             << sigName[sig] << endl;
#else
    struct sigaction action;
    action.sa_handler = coSignal::doHandle;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);
    if (sigaction(sig, &action, NULL))
        cerr << "Errors installing coSignal POSIX handler for "
             << sigName[sig] << endl;
#endif
}

///////////////////////////////////////////////////////////////////////////////////
void coSignal::doHandle(int sig)
{
    if (!s_instance)
        s_instance = new coSignal;

#ifdef USE_BSD_SIGNALS
    signal(sig, coSignal::doHandle); // re-install handler
    if (s_blockMask[sig])
    {
        s_blockMask[sig] = -1; // signal is blocked -> set pending and return
        return;
    }
#endif

    handlerRec *hrec = s_handler[sig];
    while (hrec)
    {
#ifdef VERBOSE
        cerr << "Calling user handler " << hrec->handler->sigHandlerName()
             << " for Signal " << sigName[sig] << endl;
#endif
        hrec->handler->sigHandler(sig);
        hrec = hrec->next;
    }

    // We might want to call the original system handler now
    if (s_callSys[sig])
    {
// install the signal handler
#ifdef USE_BSD_SIGNALS
        signal(sig, SIG_DFL);
#else
        struct sigaction action;
        action.sa_handler = SIG_DFL;
        action.sa_flags = 0;
        sigemptyset(&action.sa_mask);
        if (sigaction(sig, &action, NULL))
            cerr << "Errors re-installing default handler for "
                 << sigName[sig] << endl;

        sigset_t myMask;
        sigemptyset(&myMask);
        sigaddset(&myMask, sig);
        sigprocmask(SIG_UNBLOCK, &myMask, NULL);
#endif

        // send signal
        raise(sig);

// install own handler back it we return
#ifdef USE_BSD_SIGNALS
        signal(sig, coSignal::doHandle);
#else
        action.sa_handler = coSignal::doHandle;
        action.sa_flags = 0;
        sigemptyset(&action.sa_mask);
        if (sigaction(sig, &action, NULL))
            cerr << "Errors re-installing default handler for "
                 << sigName[sig] << endl;
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////////
void coSignal::removeSignal(int sig, coSignalHandler &myHandler)
{
    if (!s_instance)
        s_instance = new coSignal;

    if (s_handler[sig])
    {
        handlerRec *last, *hRec;
        hRec = s_handler[sig];
        if (hRec->handler == &myHandler)
        {
            last = hRec;
            s_handler[sig] = hRec->next;
            delete last;
        }
        else
        {
            last = hRec;
            hRec = hRec->next;
            while (hRec && (hRec->handler != &myHandler))
            {
                last = hRec;
                hRec = hRec->next;
            }
            if (hRec)
            {
                last->next = hRec->next;
                delete last;
            }
            else
                cerr << "Handler '" << myHandler.sigHandlerName()
                     << " not defined for Signal" << sigName[sig] << endl;
        }

        // if this was the last - re-establish default handler
        if (s_handler[sig] == NULL)
        {
#ifdef USE_BSD_SIGNALS
            if (!s_blockMask[sig])
                signal(sig, SIG_DFL);
#else
            struct sigaction defaultAct;
            defaultAct.sa_handler = SIG_DFL;
            defaultAct.sa_flags = 0;
            sigemptyset(&defaultAct.sa_mask);
            if (sigaction(sig, &defaultAct, NULL))
                cerr << "Errors re-installing default handler for "
                     << sigName[sig] << endl;
#ifdef VERBOSE
            else
                cerr << "Re-installed default handler for " << sigName[sig] << endl;
#endif
#endif
        }
    }
    else
        cerr << "No handlers defined for Signal" << sigName[sig] << endl;
}

/// block signal for a while
void coSignal::blockSignal(int sig)
{
    if (!s_instance)
        s_instance = new coSignal;

#ifndef USE_BSD_SIGNALS
    if (!s_handler[sig])
        cerr << "Blocking signal without a defined handler" << endl;

    sigset_t myMask, oldMask;
    sigemptyset(&myMask);
    sigaddset(&myMask, sig);
    sigprocmask(SIG_BLOCK, &myMask, &oldMask);

    sigprocmask(0, NULL, &oldMask);
    if (sigismember(&oldMask, SIGINT))
        cerr << "SIGINT blocked" << endl;

#else
    if (!s_blockMask[sig]) // it might already be -1, so don't destroy
    {
        s_blockMask[sig] = 1;
        signal(sig, coSignal::doHandle);
    }
#endif
}

void coSignal::unblockSignal(int sig)
{
    if (!s_instance)
        s_instance = new coSignal;

#ifndef USE_BSD_SIGNALS
    sigset_t myMask, oldMask;
    if (!s_handler[sig])
        cerr << "Blocking signal without a defined handler" << endl;

    sigemptyset(&myMask);
    sigaddset(&myMask, sig);
    sigprocmask(SIG_UNBLOCK, &myMask, &oldMask);
#else
    if (s_blockMask[sig] == -1) // exec handler if blocked and pending
    {
        s_blockMask[sig] = 0;
        if (s_handler[sig])
            doHandle(sig);
        else
        {
            signal(sig, SIG_DFL); // no handler - blocking
#ifndef _WIN32
            ::kill(getpid(), sig);
#endif
        }
    }
    else
    {
        s_blockMask[sig] = 0;
        if (!s_handler[sig])
            signal(sig, SIG_DFL);
    }
#endif
}

/// block signal
void coSignal::blockAllSignals()
{
    if (!s_instance)
        s_instance = new coSignal;

    int signo;

#ifndef USE_BSD_SIGNALS
    sigset_t myMask;
    sigemptyset(&myMask);
    for (signo = 0; signo < NSIG; signo++)
        if (s_handler[signo])
            sigaddset(&s_sysMask, signo);

    sigprocmask(SIG_BLOCK, &myMask, NULL);

#else
    for (signo = 0; signo < NSIG; signo++)
        if (s_handler[signo])
            blockSignal(signo);
#endif
}

/// unblock signal
void coSignal::unblockAllSignals()
{
    if (!s_instance)
        s_instance = new coSignal;

    int signo;

#ifndef USE_BSD_SIGNALS
    sigset_t myMask;
    sigemptyset(&myMask);
    for (signo = 0; signo < NSIG; signo++)
        if (s_handler[signo])
            sigaddset(&myMask, signo);
    sigprocmask(SIG_UNBLOCK, &myMask, NULL);
#else
    for (signo = 0; signo < NSIG; signo++)
        if (s_handler[signo])
            unblockSignal(signo);
#endif
}

void coSignal::printBlocks()
{
    if (!s_instance)
        s_instance = new coSignal;

    int signo;
    cerr << "Blocked signals:";

#ifndef USE_BSD_SIGNALS
    sigset_t myMask;
    sigprocmask(0, NULL, &myMask);
    for (signo = 1; signo < NSIG; signo++)
        if (sigismember(&myMask, signo))
            cerr << " " << sigName[signo];
#else
    for (signo = 1; signo < NSIG; signo++)
        if (s_blockMask[signo])
            cerr << " " << sigName[signo];
#endif

    cerr << endl;
}

bool coSignal::isPending(int sig)
{
    if (!s_instance)
        s_instance = new coSignal;

#ifndef USE_BSD_SIGNALS
    sigset_t myMask;
    sigpending(&myMask);
    return (sigismember(&myMask, sig));
#else
    return (s_blockMask[sig] == -1);
#endif
}
