/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <do/coDistributedObject.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoOctTree.h>
#include <do/coDoOctTreeP.h>
#include "coModule.h"
#include "coPort.h"
#include "coUifSwitchCase.h"
#include "coUifSwitch.h"

#ifndef _WIN32
#include <sys/ioctl.h>
#include <sys/socket.h>
#endif

#include <util/unixcompat.h>

#include <sys/types.h>

#if defined(__sun)
#include <sys/filio.h>
#endif

using namespace covise;

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
coModule::coModule(const coModule &)
{
    assert(false);
}

/// Assignment operator: NOT  IMPLEMENTED
coModule &coModule::operator=(const coModule &)
{
    assert(false);
    return *this;
}

/// ----- Never forget the Destructor !! -------

coModule::~coModule()
{
    if (d_init_title != d_title)
    {
        delete[] d_init_title;
    }
    delete[] d_title;
}

coModule::coModule(int argc, char *argv[], const char *desc, bool propagate)
{
    (void)argc;
    (void)argv;

    d_numElem = 0;
    d_numActCase = 0;
    d_numActSwitch = 0;
    d_numSocket = 0;
    d_autoParamInit = 0;
    d_execFlag = 0;
    d_execGracePeriod = 1.0;
    _propagateObjectName = propagate;
    // declare the name of our module if given here
    if (desc)
    {
        Covise::set_module_description(desc);
    }
    else
    {
        Covise::set_module_description("No description given");
    }

    // init title of module
    d_title = NULL;
    d_init_title = NULL;

    // Read 'grace period' from config file
    d_execGracePeriod = coCoviseConfig::getFloat("System.HostInfo.ExecGracePeriod", 1.0f);
    if (d_execGracePeriod < 0)
    {
        sendWarning("Corrected ExecGracePeriod<0 from config file to 1 sec");
        d_execGracePeriod = 1.0f;
    }

    // if CO_DEBUGGER is set: start debugger

    char buffer[128];
    const char *debug = getenv("CO_DEBUGGER");
    if (debug)
    {
        sprintf(buffer, debug, getpid());
        if (!strchr(buffer, '&')) // if the command isn't sent to background
        {
            int pos = (int)strlen(buffer);
            buffer[pos] = '&';
            buffer[pos + 1] = '\0';
        }
        int retval;
        retval = system(buffer);
        if (retval == -1)
        {
            std::cerr << "coModule::coModule: execution of " << buffer << " failed" << std::endl;
        }
        sleep(10);
    }
}

/// allow users to use old-style for all other ports
//void coModule::add_port(enum appl_port_type ptype, char *name,char *type, char *desc)
//{
//   Covise::add_port(ptype,name,type,desc);
//}

/// add a port : overload Covise::add_port
int coModule::add_port(coPort *param)
{
    if (findElem(param->getName()))
    {
        cerr << "Duplicate Part/Switch Identifier '"
             << param->getName()
             << "': second appearence not registered"
             << endl;
        return -1;
    }

    if (d_numElem >= Covise::MAX_PORTS)
    {
        cerr << ((char)7) << "number of Ports limited to "
             << MAX_PORTS << " in coModule" << endl;
        return 0;
    }

    elemList[d_numElem] = param;
    d_numElem++;

    // if we are in a switch/case, we'll have to enter it there
    paraCaseAdd(param);

    return 0;
}

bool coModule::GoodOctTrees(const coDistributedObject *grid, const coDistributedObject *otree)
{
    if (grid == NULL || otree == NULL)
    {
        return false;
    }
    const coDoSet *set = dynamic_cast<const coDoSet *>(grid);
    if (set)
    {
        const coDoSet *oset = dynamic_cast<const coDoSet *>(otree);
        if (!oset)
        {
            return false;
        }
        int no_elems;
        const coDistributedObject *const *setList = set->getAllElements(&no_elems);
        int no_oelems;
        const coDistributedObject *const *osetList = oset->getAllElements(&no_oelems);
        if (no_oelems != no_elems)
        {
            return false;
        }
        int elem;
        for (elem = 0; elem < no_elems; ++elem)
        {
            if (!GoodOctTrees(setList[elem], osetList[elem]))
            {
                return false;
            }
        }
#ifdef _CLEAN_UP_
        for (elem = 0; elem < no_elems; ++elem)
        {
            delete setList[elem];
            delete osetList[elem];
        }
        delete[] setList;
        delete[] osetList;
#endif
        return true;
    }
    else if (dynamic_cast<const coDoUnstructuredGrid *>(grid))
    {
        if (dynamic_cast<const coDoOctTree *>(otree))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (dynamic_cast<const coDoPolygons *>(grid))
    {
        if (dynamic_cast<const coDoOctTreeP *>(otree))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return true;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//    --------------------------- Switch / Case  ---------------------------
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coChoiceParam *coModule::paraSwitch(const char *name, const char *desc)
{
    // cerr << "paraStartSwitch(\"" << name << "\".\"" << desc << "\");" << endl;

    coUifSwitch *newSwitch;

    // create the switch
    if (d_numActSwitch)
        newSwitch = new coUifSwitch(name, desc, 0);
    else
        newSwitch = new coUifSwitch(name, desc, 1); // toplevel

    // put on stack of open switches
    d_actSwitch[d_numActSwitch] = newSwitch;
    d_numActSwitch++;

    // put into list of UIF elements
    elemList[d_numElem] = newSwitch;
    d_numElem++;

    // if we are in a switch/case, we'll have to enter it there, too
    if (d_numActCase)
        d_actCase[d_numActCase - 1]->add(newSwitch);
    return newSwitch->getMasterChoice();
}

int coModule::paraEndSwitch()
{
    if (d_numActSwitch == 0)
    {
        Covise::sendWarning("called paraEndSwitch without pending paraSwitch");
        return -1;
    }

    d_numActSwitch--;
    //cerr << "paraEndSwitch() -> closing "  << d_actSwitch[d_numActSwitch]->getName() << endl;
    d_actSwitch[d_numActSwitch]->finish();
    return 0;
}

/// start a parameter switch's case
int coModule::paraCase(const char *name)
{
    if (d_numActSwitch == 0)
    {
        Covise::sendWarning("called paraStartCase without pending paraStartSwitch");
        return -1;
    }
    d_actCase[d_numActCase] = d_actSwitch[d_numActSwitch - 1]->addCase(name);
    d_numActCase++;
    return 0;
}

/// add an existing parameter to a parameter switch's case
int coModule::paraCaseAdd(coPort *param)
{
    if (d_numActCase && param->switchable())
    {
        d_actCase[d_numActCase - 1]->add(param);
        return 0;
    }
    return -1;
}

/// end a pending parameter switch's case
int coModule::paraEndCase()
{
    if (d_numActCase == 0)
    {
        Covise::sendWarning("called paraEndCase without pending paraCase");
        return -1;
    }
    d_numActCase--;
    return 0;
}

void coModule::autoInitParam(int value)
{
    d_autoParamInit = value;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//    ----------------------- Covise Main-Loop et al -------------------------
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// initialize Covise
void coModule::init(int argc, char *argv[])
{
    int i;

    // this creates all input parameter ports
    // aw:2000-11-07 Sequence Param, Outports, Inports (necess for setDepend()
    // change Sequence to Param, Inports, Outports (necess for setDepend() on output port)

    for (i = 0; i < d_numElem; i++)
        if (elemList[i]->kind() == coUifElem::SWITCH
            || elemList[i]->kind() == coUifElem::PARAM)
            elemList[i]->initialize();

    for (i = 0; i < d_numElem; i++)
        if (elemList[i]->kind() == coUifElem::INPORT)
            elemList[i]->initialize();

    for (i = 0; i < d_numElem; i++)
        if (elemList[i]->kind() == coUifElem::OUTPORT)
            elemList[i]->initialize();

    Covise::init(argc, argv);
    Covise::set_quit_callback(coModule::quitCallback, this);
    Covise::set_start_callback(coModule::computeCallback, this);
    Covise::set_param_callback(coModule::paramCallback, this);
    Covise::set_port_callback(coModule::portCallback, this);
    Covise::set_feedback_callback(coModule::feedbackCallback, this);
    Covise::set_add_object_callback(coModule::addObjCallback, this);

    if (d_autoParamInit)
    {
        for (i = 0; i < d_numElem; i++)
        {
            elemList[i]->show();
            elemList[i]->hide();
        }
    }

    // always show all top-level switches
    for (i = 0; i < d_numElem; i++)
    {
        if (elemList[i]->kind() == coUifElem::SWITCH)
        {
            coUifSwitch *sw = (coUifSwitch *)elemList[i];
            if (sw->isTopLevel())
                sw->show();
        }
    }

    //send description
    initDescription();
    // call the user's postInst() if he has one
    postInst();
}

/// initialize Covise and loop
void coModule::start(int argc, char *argv[])
{
    init(argc, argv);
    // and go into (own or overloaded) main-loop -> this will not return
    mainLoop();
}

void coModule::handleMessages(float time)
{
    // things required for select()
    fd_set readfds;
    int nfdsr;
    int nfds = 0;
    int i;
    struct timeval timeout;
    int locErrno;

    // get socket descriptor for connection module-controller
    int covise_fd = Covise::get_socket_id();

    {
        // initialize descriptor set
        FD_ZERO(&readfds);

        // include the covise socket descriptor into the set
        FD_SET(covise_fd, &readfds);

        // ndfs is always last socket-ID we find
        nfds = covise_fd;

        // if anything else is registered: put into select structure
        for (i = 0; i < d_numSocket; i++)
        {
            FD_SET(d_socket[i], &readfds);
            if (nfds < d_socket[i])
                nfds = d_socket[i];
        }

        // call the idle function
        float idleTime = idle();

        // The user requested self-execution
        if (d_execFlag)
        {
            // build feedback string
            char buf[256];
            sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(),
                    Covise::get_instance(),
                    Covise::get_host());
            Covise::set_feedback_info(buf);

            // send execute message
            Covise::send_feedback_message("EXEC", "");

            // ok, we've done it
            d_execFlag = 0;
        }

        float waitTime = time;
        if ((waitTime < 0) || ((idleTime != -1) && (idleTime < time)))
        {
            waitTime = idleTime;
        }

        if (waitTime > 0) // wait time seconds
        {
            timeout.tv_sec = (int)waitTime;
            timeout.tv_usec = (int)((waitTime - timeout.tv_sec) * 1000000);
            nfdsr = select(nfds + 1, &readfds, NULL, NULL, &timeout);
            locErrno = errno;
        }
        else if (waitTime == 0) // just check, don't wait
        {
            timeout.tv_sec = 0;
            timeout.tv_usec = 1; // the manpages say it has to me a non zero value...
            nfdsr = select(nfds + 1, &readfds, NULL, NULL, &timeout);
            locErrno = errno;
        }
        else // block long time otherwise
        {
            timeout.tv_sec = 10;
            timeout.tv_usec = 0;
            // examine descriptor set
            nfdsr = select(nfds + 1, &readfds, NULL, NULL, &timeout);
            locErrno = errno;
        }
// check, whether controller socket terminated or parent died
#ifndef _WIN32
        int err;
        socklen_t errlen = sizeof(err);
        getsockopt(covise_fd, SOL_SOCKET, SO_ERROR, &err, &errlen);
        if (err || getppid() == 1)
            exit(0);
#endif
        if (nfdsr < 0)
        {
            if (locErrno != EINTR)
            {
                sendError("coModule::handleMessages: ERROR in select: %s", strerror(locErrno));
            }
        }
        else
        {
            while (nfdsr != 0) // so lange Messages kommen
            {
                // check for Covise event, do not use FD_ISSET for neg. socket descr.
                // awe 2001/01/15: also check whether there is qeueued data
                if (covise_fd >= 0 && FD_ISSET(covise_fd, &readfds))
                {
                    // handle as many CTLR messages as received
                    while (Covise::check_and_handle_event(0.0001f))
                        ;
                }

                // if registered socket received data AND this data wasn't handled before
                // e.g. in the compute() callback called from check_and_handle_event.
                for (i = 0; i < d_numSocket; i++)
                    if (FD_ISSET(d_socket[i], &readfds))
                    {
#ifndef _WIN32
                        size_t bytes;
                        int res = ioctl(d_socket[i], FIONREAD, &bytes);
                        if (res == 0 && bytes > 0)
#endif
                            sockData(d_socket[i]);
                    }

                // The user requested self-execution
                if (d_execFlag)
                {
                    // just to cure the runtime problems: use select for precise sleep() call
                    timeout.tv_sec = (int)d_execGracePeriod;
                    timeout.tv_usec = (int)((d_execGracePeriod - timeout.tv_sec) * 1000000);
                    select(0, NULL, NULL, NULL, &timeout);

                    // build feedback string
                    char buf[256];
                    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(),
                            Covise::get_instance(),
                            Covise::get_host());
                    Covise::set_feedback_info(buf);

                    // send execute message
                    Covise::send_feedback_message("EXEC", "");

                    // ok, we've done it
                    d_execFlag = 0;
                }

                // es koennten noch messages warten, also pollen...
                // initialize descriptor set
                FD_ZERO(&readfds);

                // include the covise socket descriptor into the set
                FD_SET(covise_fd, &readfds);

                // ndfs is always last socket-ID we find
                nfds = covise_fd;

                // if anything else is registered: put into select structure
                for (i = 0; i < d_numSocket; i++)
                {
                    FD_SET(d_socket[i], &readfds);
                    if (nfds < d_socket[i])
                        nfds = d_socket[i];
                }

                timeout.tv_sec = 0;
                timeout.tv_usec = 1; // the manpages say it has to me a non zero value...
                nfdsr = select(nfds + 1, &readfds, NULL, NULL, &timeout);

                if (nfdsr < 0)
                {
                    Covise::sendError("ERROR: select");
                    break;
                }
            }
        }
    }
}

void coModule::mainLoop()
{
#ifdef VERBOSE
    Covise::sendInfo("... entering main loop");
#endif

    while (1)
    {
        handleMessages(-1);
    }
}

/// overload these functions for your compute / quit / param Callbacks
int
coModule::addObject(const char *objectNameToAdd, const char *objectNameToDelete)
{
    (void)objectNameToAdd;
    (void)objectNameToDelete;
    Covise::sendError(
        "ADD_OBJECT called and addObject(..) not overloaded");
    return STOP_PIPELINE;
}

/// overload these functions for your compute / quit / param Callbacks
int coModule::compute(const char *port)
{
    (void)port;
    Covise::sendError("EXEC calls and compute() not overloaded");
    return STOP_PIPELINE;
}

/// overload these functions for your compute / quit / param Callbacks
void coModule::sockData(int sockNo)
{
    (void)sockNo;
    Covise::sendError(
        "cockets registered and sockData(int sockNo) not overloaded");
}

// This is the user function: overload it... we don't need it yet
void coModule::feedback(int, const char *) {}

// This is the user function: overload it... we don't need it yet
void coModule::quit(void) {}

//  This is the user function: overload it... but we call our own first
void coModule::param(const char * /*name*/, bool /*inMapLoading*/) {}

//  This is the user function
void coModule::postInst() {}

//  this funktion is called before the select call
//  and should return the timeout in seconds for the next select
//  -1 will do a blocking select
float coModule::idle()
{
    return -1;
}

// ... and this is our own parameter Callback
void coModule::localParam(bool inMapLoading, void *callbackData)
{
    (void)callbackData;
    const char *paramname = Covise::get_reply_param_name();

    // title of module has changed
    if (!strcmp(paramname, "SetModuleTitle"))
    {
        const char *title;
        Covise::get_reply_string(&title);

        delete[] d_title;
        d_title = strcpy(new char[strlen(title) + 1],
                         title);
        param(paramname, inMapLoading);

        return;
    }

    // if this is a registered element (it should be!) -> call its handle virtual
    coUifElem *elem = findElem(paramname);

    if (elem)
    {
        coUifSwitch *sw = (coUifSwitch *)(elem);
        coUifPara *pa = (coUifPara *)(elem);

        if (elem->kind() == coUifElem::SWITCH)
            sw->paramChange();
        else if (elem->kind() == coUifElem::PARAM)
            pa->paramChange();
    }
    else if (!strstr(paramname, "___filter"))
    {
        // ignore non-registered filebrowser filters
        sendWarning("Received message for non-registered parameter '%s'", paramname);
    }

    param(paramname, inMapLoading); // in the end we call the user's routine
}

// ... and this is our port Callback
void coModule::localPort(void *callbackData)
{
    (void)callbackData;
    const char *paramname = Covise::get_reply_port_name();
    //cerr << "coModule Parameter CB called for '" << paramname << "'" << endl;

    // if this is a registered element (it should be!) -> call its handle virtual

    coUifPort *elem = (coUifPort *)findElem(paramname);
    if (elem)
    {
        elem->paramChange();
        param(paramname, false);
    }
    else
        sendWarning("Received message for non-registered port '%s'", paramname);
}

// ...  own 'compute' Callback -> retrieves all non-immediate parameters
void coModule::localCompute(void *)
{
    int i;

    // TOLERANT:  silently skip compute() call when flag is set : done in coInputPort
    for (i = 0; i < d_numElem; i++)
        if (elemList[i]->preCompute())
            return;

    // call user's compute function, stop pipeline if not successful
    if (compute(NULL) == STOP_PIPELINE)
        stopPipeline();
    //Add OBJECTNAME-Attribute to
    //the data
    for (i = 0; i < d_numElem; i++)
    {
        if (elemList[i]->kind() == coUifElem::OUTPORT)
        {
            if (NULL != ((coOutputPort *)elemList[i])->getCurrentObject())
            {
                coDistributedObject *obj = ((coOutputPort *)elemList[i])->getCurrentObject();
                if (!obj->checkObject())
                {
                    sendError("Object Integrity check failed: see console");
                    stopPipeline();
                }
                if (_propagateObjectName)
                {
                    obj->addAttribute("OBJECTNAME", this->getTitle());
                }
            }
        }
    }
    for (i = 0; i < d_numElem; i++)
        elemList[i]->postCompute();
}

void
coModule::localAddObject(void *callbackData)
{
    (void)callbackData;
    if (addObject(Covise::getObjNameToAdd(), Covise::getObjNameToDelete()) == STOP_PIPELINE)
    {
        stopPipeline();
    }
}

/*************************************************************
 *  These are used by Application interface
 *************************************************************/
// static stub callback functions calling the real class
// member functions

void coModule::quitCallback(void *userData, void *callbackData)

{
    (void)callbackData;
    coModule *thisApp = (coModule *)userData;
    thisApp->quit();
}

void coModule::computeCallback(void *userData, void *callbackData)
{
    bool executeDebug = getenv("COVISE_EXECUTE_DEBUG") != NULL;
    if (executeDebug)
        fprintf(stderr, ">>> pre %s_%s coModule::computeCallback\n", Covise::get_module(), Covise::get_instance());
    coModule *thisApp = (coModule *)userData;
    thisApp->localCompute(callbackData);
    if (executeDebug)
        fprintf(stderr, "<<< post %s_%s coModule::computeCallback\n", Covise::get_module(), Covise::get_instance());
}

void coModule::paramCallback(bool inMapLoading, void *userData, void *callbackData)
{
    coModule *thisApp = (coModule *)userData;
    thisApp->localParam(inMapLoading, callbackData);
}

void coModule::portCallback(void *userData, void *callbackData)
{
    coModule *thisApp = (coModule *)userData;
    thisApp->localPort(callbackData);
}

void coModule::feedbackCallback(void *userData, int len, const char *data)
{
    coModule *thisApp = (coModule *)userData;
    thisApp->feedback(len, data);
}

void coModule::addObjCallback(void *userData, void *callbackData)
{
    coModule *thisApp = (coModule *)userData;
    thisApp->localAddObject(callbackData);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// find Element by name or by description
coUifElem *
coModule::findElem(const char *name)
{
    int i = 0;
    while ((i < d_numElem) && (elemList[i] != NULL))
    {
        if (!strcmp(elemList[i]->getName(), name))
            return elemList[i];
        else
            i++;
    }
    // search in description list
    i = 0;
    while ((i < d_numElem) && (elemList[i] != NULL))
    {
        coPort *p;
        if ((p = dynamic_cast<coPort *>(elemList[i])) != NULL)
        {
            if (!strcmp(p->getDesc(), name))
                return elemList[i];
        }
        i++;
    }

    if (i < d_numElem)
        cerr << "coModule::findElem(..) NULL Element found in list for name " << name << endl;
    return NULL;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// Add a socket handler
void coModule::addSocket(int socket)
{
    int covise_fd = Covise::get_socket_id();
    if (socket == covise_fd)
    {
        Covise::sendError("Tried to register own handler over Covise socket");
        return;
    }
    int i;
    for (i = 0; i < d_numSocket; i++)
        if (d_socket[i] == socket)
        {
            Covise::sendError("Tried to register same socket twice");
            return;
        }
    d_socket[d_numSocket] = socket;
    d_numSocket++;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// Remove a socket handler
void coModule::removeSocket(int socket)
{
    int covise_fd = Covise::get_socket_id();
    if (socket == covise_fd)
    {
        Covise::sendError("Cannot remove Covise socket");
        return;
    }

    int i;
    for (i = 0; i < d_numSocket; i++)
        if (d_socket[i] == socket)
        {
            // move all sockets behind one step forward
            for (i++; i < d_numSocket; i++)
                d_socket[i - 1] = d_socket[i];
            d_numSocket--;
            return;
        }
    Covise::sendError("Tried to remove non-registered socket");
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/// send own module an 'execute' message
void coModule::selfExec()
{
    // we just set a flag here and than fire after leaving the callbacks
    d_execFlag = 1;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/// set my selfExec grace period
void coModule::setExecGracePeriod(float gracePeriod)
{
    d_execGracePeriod = gracePeriod;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

coBooleanParam *coModule::addBooleanParam(const char *name, const char *desc)
{
    coBooleanParam *port = new coBooleanParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coFileBrowserParam *coModule::addFileBrowserParam(const char *name, const char *desc)
{
    coFileBrowserParam *port = new coFileBrowserParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coChoiceParam *coModule::addChoiceParam(const char *name, const char *desc)
{
    coChoiceParam *port = new coChoiceParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coColormapChoiceParam *coModule::addColormapChoiceParam(const char *name, const char *desc)
{
    coColormapChoiceParam *port = new coColormapChoiceParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coMaterialParam *coModule::addMaterialParam(const char *name, const char *desc)
{
    coMaterialParam *port = new coMaterialParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coColormapParam *coModule::addColormapParam(const char *name, const char *desc)
{
    coColormapParam *port = new coColormapParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coColorParam *coModule::addColorParam(const char *name, const char *desc)
{
    coColorParam *port = new coColorParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coFloatParam *coModule::addFloatParam(const char *name, const char *desc)
{
    coFloatParam *port = new coFloatParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coFloatSliderParam *coModule::addFloatSliderParam(const char *name, const char *desc)
{
    coFloatSliderParam *port = new coFloatSliderParam(name, desc);
    add_port(port);
    return port;
}

coFloatVectorParam *coModule::addFloatVectorParam(const char *name, const char *desc, int len)
{
    coFloatVectorParam *port = new coFloatVectorParam(name, desc, len);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coIntScalarParam *coModule::addInt32Param(const char *name, const char *desc)
{
    coIntScalarParam *port = new coIntScalarParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coIntSliderParam *coModule::addIntSliderParam(const char *name, const char *desc)
{
    coIntSliderParam *port = new coIntSliderParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coIntVectorParam *coModule::addInt32VectorParam(const char *name, const char *desc, int len)
{
    coIntVectorParam *port = new coIntVectorParam(name, desc, len);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coStringParam *coModule::addStringParam(const char *name, const char *desc)
{
    coStringParam *port = new coStringParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coTimerParam *coModule::addTimerParam(const char *name, const char *desc)
{
    coTimerParam *port = new coTimerParam(name, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coInputPort *coModule::addInputPort(const char *name, const char *types, const char *desc)
{
    coInputPort *port = new coInputPort(name, types, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

coOutputPort *coModule::addOutputPort(const char *name, const char *types, const char *desc)
{
    coOutputPort *port = new coOutputPort(name, types, desc);
    if (add_port(port))
        return NULL;
    else
        return port;
}

/// stop the pipeline: do not execute Modules behind this one
void coModule::stopPipeline()
{
    Covise::send_stop_pipeline();
}

void
coModule::initDescription()
{
    const char *desc = Covise::get_module_description();

    if (desc)
    {
        setInfo(Covise::get_module_description());
    }
    else
    {
        setInfo(Covise::get_module());
    }

    d_init_title = d_title = new char[strlen(Covise::get_module()) + strlen(Covise::get_instance()) + 2];
    sprintf(d_title, "%s_%s", Covise::get_module(), Covise::get_instance());
}

void coModule::setInfo(const char *value) const
{
    if (!value)
        return;

    Covise::set_module_description(value);

    char *buffer = new char[strlen(value) + 1];

    // copy everything except \n or \t
    char *bufPtr = buffer;
    while (*value)
    {
        if (*value != '\n' && *value != '\t')
        {
            *bufPtr = *value;
            bufPtr++;
        }
        value++;
    }
    *bufPtr = '\0'; // terminate string
    Covise::send_ui_message("MODULE_DESC", buffer);
    delete[] buffer;
}

void coModule::setTitle(const char *value)
{
    if (!value)
        return;

    d_title = new char[strlen(value) + 1];

    // copy everything except \n or \t
    char *bufPtr = d_title;
    while (*value)
    {
        if (*value != '\n' && *value != '\t')
        {
            *bufPtr = *value;
            bufPtr++;
        }
        value++;
    }
    *bufPtr = '\0'; // terminate string
    Covise::send_ui_message("MODULE_TITLE", d_title);
}
