/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EC_SIGNAL_H_
#define _EC_SIGNAL_H_

/*
   $Log$
*/

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

#include <util/coTypes.h>

#ifndef _WIN32
#include <signal.h>

#ifdef __sgi

/*
extern "C" void (*signal(int,void (*)(int)))(int);
*/

// #define SIG_IGN (void (*)())0       /* ignore */
#define SIGHUP 1 /* hangup */
#define SIGINT 2 /* interrupt (rubout) */
#define SIGQUIT 3 /* quit (ASCII FS) */
#define SIGILL 4 /* illegal instruction (not reset when caught)*/
#define SIGTRAP 5 /* trace trap (not reset when caught) */
#define SIGIOT 6 /* IOT instruction */
#define SIGABRT 6 /* used by abort, replace SIGIOT in the  future */
#define SIGEMT 7 /* EMT instruction */
#define SIGFPE 8 /* floating point exception */
#define SIGKILL 9 /* kill (cannot be caught or ignored) */
#define SIGBUS 10 /* bus error */
#define SIGSEGV 11 /* segmentation violation */
#define SIGSYS 12 /* bad argument to system call */
#define SIGPIPE 13 /* write on a pipe with no one to read it */
#define SIGALRM 14 /* alarm clock */
#define SIGTERM 15 /* software termination signal from kill */
#define SIGUSR1 16 /* user defined signal 1 */
#define SIGUSR2 17 /* user defined signal 2 */
#define SIGCLD 18 /* death of a child */
#define SIGCHLD 18 /* 4.3BSD's/POSIX name */
#define SIGPWR 19 /* power-fail restart */
#define SIGWINCH 20 /* window size changes */
#define SIGURG 21 /* urgent condition on IO channel */
#define SIGPOLL 22 /* pollable event occurred */
#define SIGIO 22 /* input/output possible signal */
#define SIGSTOP 23 /* sendable stop signal not from tty */
#define SIGTSTP 24 /* stop signal from tty */
#define SIGCONT 25 /* continue a stopped process */
#define SIGTTIN 26 /* to readers pgrp upon background tty read */
#define SIGTTOU 27 /* like TTIN for output if (tp->t_local&LTOSTOP) */
#define SIGVTALRM 28 /* virtual time alarm */
#define SIGPROF 29 /* profiling alarm */
#define SIGXCPU 30 /* Cpu time limit exceeded */
#define SIGXFSZ 31 /* Filesize limit exceeded */
#define SIG32 32 /* Reserved for kernel usage */
#define SIGRTMIN 49 /* Posix 1003.1b signals */
#define SIGRTMAX 64 /* Posix 1003.1b signals */

#include <signal.h>

#else

#include <signal.h>
#endif
#endif

#ifdef __linux__
#define SignalHandler CO_SignalHandler
#endif

namespace covise
{

class COVISEEXPORT SignalHandler
{

public:
    typedef void SigFunctVoid(int, void *);

private:
    void *oldData;
    static void *userdata[65];
    static void *doHandle(int sig);
    static void *handler[65];
    int sigNr;
#ifndef _WIN32
    sigset_t oldmask, newmask, pendmask;
#endif
    int use_signals;
    char *tmp_env;

public:
    SignalHandler();
    void addSignal(int signo, void *handlerFunc, void *udata);
    void removeSignal(int signo);
    //    void blockSignal(int signo);
    //    void unblockSignal(int signo);
    void blockSignals(void);
    void unblockSignals(void);
    int isPending(int signo);
    ~SignalHandler(void);
};

void (*my_signal(int sig, void (*func)(int)))(int);
}
#endif
