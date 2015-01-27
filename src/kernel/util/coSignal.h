/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_SIGNAL_H_
#define _CO_SIGNAL_H_

#ifdef _WIN32
#define USE_BSD_SIGNALS
#endif

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
 ** Author: A. Werner                                                      **
 **                                                                        **
 ** Date:  25.11.99  V2.0                                                  **
 ** Last:                                                                  **
\*************************************************************************/

#include <stdio.h>
#include "coTypes.h"

/*
#ifdef __sgi

// extern "C" void (*signal(int,void (*)(int)))(int);

// #define SIG_IGN (void (*)())0       // ignore
#define SIGHUP  1       // hangup
#define SIGINT  2       // interrupt (rubout)
#define SIGQUIT 3       // quit (ASCII FS)
#define SIGILL  4       // illegal instruction (not reset when caught)
#define SIGTRAP 5       // trace trap (not reset when caught)
#define SIGIOT  6       // IOT instruction
#define SIGABRT 6       // used by abort, replace SIGIOT in the  future
#define SIGEMT  7       // EMT instruction
#define SIGFPE  8       // floating point exception
#define SIGKILL 9       // kill (cannot be caught or ignored)
#define SIGBUS  10      // bus error
#define SIGSEGV 11      // segmentation violation
#define SIGSYS  12      // bad argument to system call
#define SIGPIPE 13      // write on a pipe with no one to read it
#define SIGALRM 14      // alarm clock
#define SIGTERM 15      // software termination signal from kill
#define SIGUSR1 16      // user defined signal 1
#define SIGUSR2 17      // user defined signal 2
//#define SIGCLD  18      // death of a child
#define SIGCHLD 18      // 4.3BSD's/POSIX name
#define SIGPWR  19      // power-fail restart
#define SIGWINCH 20     // window size changes
#define SIGURG  21      // urgent condition on IO channel
#define SIGPOLL 22      // pollable event occurred
#define SIGIO   22      // input/output possible signal
#define SIGSTOP 23      // sendable stop signal not from tty
#define SIGTSTP 24      // stop signal from tty
#define SIGCONT 25      // continue a stopped process
#define SIGTTIN 26      // to readers pgrp upon background tty read
#define SIGTTOU 27      // like TTIN for output if (tp->t_local&LTOSTOP)
#define SIGVTALRM 28    // virtual time alarm
#define SIGPROF 29      // profiling alarm
#define SIGXCPU 30      // Cpu time limit exceeded
#define SIGXFSZ 31      // Filesize limit exceeded
#define SIG32   32      // Reserved for kernel usage
#define SIGRTMIN        49      // Posix 1003.1b signals
#define SIGRTMAX        64      // Posix 1003.1b signals

#endif
*/

#if (defined(__sgi) && !defined(_BOOL)) || defined(__sun)

typedef int bool;
static const int false = 0;
static const int true = !false;
#endif

#include <signal.h>

namespace covise
{

/**
 *  Base class for any class receiving signals and empty dummy handler
 */
class UTILEXPORT coSignalHandler
{
public:
    virtual void sigHandler(int sigNo)
    {
        (void)sigNo;
    }

    virtual const char *sigHandlerName()
    {
        return "No name specified";
    }

    virtual ~coSignalHandler()
    {
    }
};

/**
 * Singleton-Class to manage signals
 */
class UTILEXPORT coSignal
{

public:
    /// Number of known signals and max. length of names
    enum
    {
        SIG_NAME_LENGTH = 15
    };

    /// Signal names
    static const char *sigName[NSIG];

protected:
    /// singleton: use coSignalHandler::getHandler() to get an instance
    coSignal();

    /// singleton: use coSignalHandler::kill() to kill all signal handling
    ~coSignal(void);

private:
    // the static and only existing signal handler class : init once
    static coSignal *s_instance;

    // this static function calls the handler class functions
    static void doHandle(int sig);

    // a system-independent 'signal' function
    void my_signal(int sigNo, void (*func)(int));

    struct handlerRec
    {
        coSignalHandler *handler;
        struct handlerRec *next;
    };
    static handlerRec *s_handler[NSIG];

#ifdef USE_BSD_SIGNALS
    // 'blocking' under BSD : =1 blocked =0 unblocked  =-1 pending
    static int s_blockMask[NSIG];
#else
    // this is what the system blocked before we came in...
    static sigset_t s_sysMask;
#endif

    // whether we call the system's handler afterwards
    static bool s_callSys[NSIG];

public:
    /// get a signal handler: if none exists, create it
    static coSignal *getHandler()
    {
        if (!s_instance)
            s_instance = new coSignal;
        return s_instance;
    }

    /// stop all user-defined signal handling, kill internal states
    static void kill()
    {
        delete s_instance;
        s_instance = NULL;
    }

    enum ReplaceOption
    {
        REPLACE_DFL,
        KEEP_DFL
    };

    /** add a signal handler after the current ones
       *  atEnd : which end of the handler sequence to add to
       *  replaceSysHandler : keep (false) or replace system handler
       */
    static void addSignal(int signo, coSignalHandler &myHandler,
                          enum ReplaceOption sysOpt = REPLACE_DFL,
                          bool atEnd = true);

    /// remove a signal's handler
    static void removeSignal(int signo, coSignalHandler &myHandler);

    /// block signal for a while
    static void blockSignal(int signo);

    /// unblock signal
    static void unblockSignal(int signo);

    /// block all signal handling of our signals
    static void blockAllSignals();

    /// unblock all signal handling of our signals
    static void unblockAllSignals();

    //is a signal pending
    static bool isPending(int signo);

    static void printBlocks();
};
}
#endif
