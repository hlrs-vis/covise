/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_FEATURE_MODULE_H_
#define _CO_FEATURE_MODULE_H_

// 15.09.99

#include <do/coDoPolygons.h>
#include <appl/ApplInterface.h>
#include "coBooleanParam.h"
#include "coFileBrowserParam.h"
#include "coChoiceParam.h"
#include "coMaterialParam.h"
#include "coColormapChoiceParam.h"
#include "coColormapParam.h"
#include "coColorParam.h"
#include "coFloatParam.h"
#include "coFloatSliderParam.h"
#include "coFloatVectorParam.h"
#include "coIntScalarParam.h"
#include "coIntSliderParam.h"
#include "coIntVectorParam.h"
#include "coStringParam.h"
#include "coTimerParam.h"
#include "coInputPort.h"
#include "coOutputPort.h"

// forward declarations for pointers only

namespace covise
{

class coUifElem;
class coPort;
class coUifSwitch;
class coUifSwitchCase;

/**
 * Base class for Covise applications
 *
 */
class APIEXPORT coModule : public Covise
{

private:
    /// this is the maximum number levels
    enum
    {
        MAX_LEVELS = 64
    };

    /// this is the maximum number of registered sockets
    enum
    {
        MAX_SOCKET = 256
    };

    /// Copy-Constructor: NOT  IMPLEMENTED
    coModule(const coModule &);

    /// Assignment operator: NOT  IMPLEMENTED
    coModule &operator=(const coModule &);

    // if we are in an open case : pointer to active case
    coUifSwitchCase *d_actCase[MAX_LEVELS];
    int d_numActCase;

    // if we are in an open switch : pointer to active case
    coUifSwitch *d_actSwitch[MAX_LEVELS];
    int d_numActSwitch;

    // list of registered sockets, number in list and handler Obj
    int d_socket[MAX_SOCKET];
    int d_numSocket;

    // static stubs for callback
    static void quitCallback(void *userData, void *callbackData);
    static void computeCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    static void feedbackCallback(void *userData, int len, const char *data);
    static void portCallback(void *userData, void *callbackData);
    static void addObjCallback(void *userData, void *callbackData);

    // our internal callback for changed parameters
    void localParam(bool inMapLoading, void *callbackData);
    // our internal callback connection changing ports
    void localPort(void *callbackData);

    // whether we automatically fire up all parameters first to secure the order
    int d_autoParamInit;

    // When this flag has been set, fire an EXEC message after returning
    // from this event
    int d_execFlag;

    // 'Grace period' to wait after self-exec to prevent overrinning next module
    float d_execGracePeriod;

    // Title of the module
    char *d_title;

    // initial Title of the module
    char *d_init_title;

    //send description to controller and UIF's
    void initDescription();

protected:
    // register a parameter port at the module: return 0 if ok, -1 on error
    int add_port(coPort *param);

    //should the module's name
    //be propagated as attribute?
    bool _propagateObjectName;

    // find element by name : derived classes might access directly
    coUifElem *findElem(const char *name);

    // list of all UIF Elements and number of Elements in list
    coUifElem *elemList[Covise::MAX_PORTS];
    int d_numElem;

    // our internal 'pre-compute' sets all non-immediate parameters
    virtual void localCompute(void *callbackData);

    // internal callback called if ADD_OBJECT messages arrive
    virtual void localAddObject(void *callbackData);

public:
    /// return values for call-back functions
    enum
    {
        FAIL = -1,
        SUCCESS = 0,
        STOP_PIPELINE = -1,
        CONTINUE_PIPELINE = 0
    };

    // ------------------ Constructor / destructor --------------------------

    /// YAC-compatible constructor
    coModule(int argc, char *argv[], const char *desc = NULL, bool propagate = false);

    /// Destructor : virtual in case we derive objects
    virtual ~coModule();

    // --------------------- Utility functions ------------------------------

    bool GoodOctTrees(const coDistributedObject *grid, const coDistributedObject *otree);

    /// execute your own module
    void selfExec();

    /// set the module's grace period
    void setExecGracePeriod(float gracePeriod);

    // --------------------- Parameter switching ----------------------------

    /// start a parameter switch: return a pointer to its master choice
    coChoiceParam *paraSwitch(const char *name, const char *desc);

    /// end a pending parameter switch
    int paraEndSwitch();

    /// start a parameter switch's case
    int paraCase(const char *name);

    /// end a pending parameter switch's case
    int paraEndCase();

    /// request automatical fireUp
    void autoInitParam(int value = 0);

    // -------------------- add a socket to listen to -----------------------

    /// Add socket to the main loop and a handler object to handle the events
    void addSocket(int socket);

    /// Add socket to the main loop and a handler object to handle the events
    void removeSocket(int socket);

    // --------------------- Covise Main-Loop et al -------------------------

    /// initialize Covise do NOT enter mainLoop
    virtual void init(int argc, char *argv[]);

    /// initialize Covise and start main event loop -> never returns
    virtual void start(int argc, char *argv[]);

    /// called for every EXEC callback: Overload it with your own compute routine
    //   return either SUCCESS or FAIL
    virtual int compute(const char *port);

    /// called for every ADD_OBJECT callback: Overload it with your own process routine
    //   return either SUCCESS or FAIL
    virtual int addObject(const char *objectNameToAdd, const char *objectNameToDelete);

    /// Overload this if you want to notice parameter changes
    virtual void param(const char *paramName, bool inMapLoading);

    /// Overload this if you register any ports
    virtual void sockData(int sockNo);

    /// Overload this function if you need a feedback routine
    virtual void feedback(int len, const char *data);

    /// Overload this function if you need a cleanup routine
    virtual void quit(void);

    /// Overload this if you want to do anything between init and main loop
    virtual void postInst();

    /// Overload this if you want to do anything while the module
    /// is waiting for messages
    virtual float idle();

    /// handle messages, does block for time seconds (forever if time == -1)
    virtual void handleMessages(float time);

    /// To overload the mainLoop: only do this if you REALLY know what you do
    virtual void mainLoop();

    /// stop the pipeline: do not execute Modules behind this one
    void stopPipeline();

    // --------------------- Covise Main-Loop et al -------------------------

    /// add Boolean Port : return NULL on error
    coBooleanParam *addBooleanParam(const char *name, const char *desc);

    /// add Browser Port : return NULL on error
    coFileBrowserParam *addFileBrowserParam(const char *name, const char *desc);

    /// add Choice Port : return NULL on error
    coChoiceParam *addChoiceParam(const char *name, const char *desc);

    /// add Color Choice Port : return NULL on error
    coColormapChoiceParam *addColormapChoiceParam(const char *name, const char *desc);

    /// add Material Port : return NULL on error
    coMaterialParam *addMaterialParam(const char *name, const char *desc);

    /// add Colormap Port : return NULL on error
    coColormapParam *addColormapParam(const char *name, const char *desc);

    /// add Color Port : return NULL on error
    coColorParam *addColorParam(const char *name, const char *desc);

    /// add Float Scalar Port : return NULL on error
    coFloatParam *addFloatParam(const char *name, const char *desc);

    /// add Float Slider Port : return NULL on error
    coFloatSliderParam *addFloatSliderParam(const char *name, const char *desc);

    /// add Float Vector Port : return NULL on error
    coFloatVectorParam *addFloatVectorParam(const char *name, const char *desc, int length = 3);

    /// add Integer Scalar Port : return NULL on error
    coIntScalarParam *addInt32Param(const char *name, const char *desc);

    /// add Integer Slider Port : return NULL on error
    coIntSliderParam *addIntSliderParam(const char *name, const char *desc);

    /// add Integer Vector Port : return NULL on error
    coIntVectorParam *addInt32VectorParam(const char *name, const char *desc, int length = 3);

    /// add String Port : return NULL on error
    coStringParam *addStringParam(const char *name, const char *desc);

    /// add Timer Port : return NULL on error
    coTimerParam *addTimerParam(const char *name, const char *desc);

    /// add Input Data Port : return NULL on error
    coInputPort *addInputPort(const char *name, const char *types, const char *desc);

    /// add Output Data Port : return NULL on error
    coOutputPort *addOutputPort(const char *name, const char *types, const char *desc);

    // ------------------------------ Mapeditor message
    /// Set the info Popup text
    void setInfo(const char *value) const;

    /// Set the Module's title
    void setTitle(const char *value);
    bool titleChanged()
    {
        return (strcmp(d_title, d_init_title) != 0);
    };
    const char *getTitle()
    {
        return d_title;
    };
};

class APIEXPORT coFunctionModule : public coModule
{
public:
    /// YAC-compatible constructor
    coFunctionModule(int argc, char *argv[], const char *desc = NULL, bool propagate = false)
        : coModule(argc, argv, desc, propagate)
    {
    }

    /// Destructor : virtual in case we derive objects
    virtual ~coFunctionModule()
    {
    }
};

#define MODULE_MAIN(Category, Module)           \
    int main(int argc, char *argv[])            \
    {                                           \
        coModule *app = new Module(argc, argv); \
        app->start(argc, argv);                 \
    }

#define COMODULE
}
#endif
