/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EC_SIGNAL_H_
#define _EC_SIGNAL_H_


#include <util/coTypes.h>

#ifndef _WIN32
#include <signal.h>
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
