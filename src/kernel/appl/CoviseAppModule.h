/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// basic ApplicationModule - class for COVISE
// by Lars Frenzel
//
// history:   	01/16/1998  	started working
// 		11/18/1998	added multiprocessing support for __sgi
//

#if !defined(__COVISE_APP_MODULE_H)
#define __COVISE_APP_MODULE_H

#include <covise/covise.h>
#include "ApplInterface.h"

#if defined(__sgi)
#include <ulocks.h>
#endif

namespace covise
{

class APPLEXPORT CoviseAppModule
{
private:
    // callbacks
    static void computeCallback(void *, void *);
    void nonstaticComputeCallback();

    // doing all the work
    void preHandleObjects(const coDistributedObject **obj_in);

    // names of ports (NULL for none / unused)
    char **inport_names;
    char **outport_names;
    int num_inports;
    int num_outports;

    // should we copy set-attributes or not ?
    int copy_attributes_flag;
    int or_copy_addAttributes_flag; // sl:

    // is Multiprocessing supported by the module ?
    bool multiprocessing_flag;
    int current_level;
    bool pre_multiprocessing_flag;

    // should we compute timesteps/multiblock
    int compute_timesteps;
    int compute_multiblock;

protected:
    // get a coDistributedObject by giving the port-name
    const coDistributedObject *getUnknown(char *name, int &errflag);

public:
    // basic stuff
    CoviseAppModule();
    virtual ~CoviseAppModule();

    // handling attributes
    void setCopyAttributes(int val)
    {
        copy_attributes_flag = val;
        return;
    };
    void setCopySetAttributesFlag(int val) //sl:
    {
        or_copy_addAttributes_flag = val;
        return;
    }
    void copyAttributes(const coDistributedObject *, coDistributedObject *);

    // setting the port-names
    void setPortNames(const char **, const char **);

    // setting up the callback-functions
    void setCallbacks();

    // set, if the module is multiprocessing-aware
    void setMultiprocessing(bool val)
    {
        multiprocessing_flag = val;
    };
    void setPreMultiprocessing(bool val)
    {
        pre_multiprocessing_flag = val;
    };

    // set what type of data to compute
    void setComputeTimesteps(int val)
    {
        compute_timesteps = val;
        return;
    };
    void setComputeMultiblock(int val)
    {
        compute_multiblock = val;
        return;
    };

    // we may require information about the current level
    int getLevel()
    {
        return (current_level);
    };

    // required for multiprocessing
    void lockNewObjects();
    void unlockNewObjects();
#if defined(__sgi)
    barrier_t *thisGroupsBarrier;
    ulock_t *thisGroupsMutex;
#endif

    // virtual stuff - may be provided by programmer
    virtual void preCompute(const coDistributedObject **){};
    virtual coDistributedObject **compute(const coDistributedObject **, char **)
    {
        return NULL;
    };
    virtual void postCompute(const coDistributedObject **, coDistributedObject **){};

    // doing all the work (has to be public in order for multiprocessing to work)
    void handleObjects(const coDistributedObject **obj_in, char **obj_out_names,
                       coDistributedObject ***obj_set_out = NULL, int set_id = 0, int num_proc_wait = 1);
};

#if defined(__sgi)

struct CoviseAppModuleInfo
{
    // SMP stuff
    barrier_t *barrier;
    int first, max, step;
    int num;

    // the module
    CoviseAppModule *mod;

    // and the parameters
    coDistributedObject ***obj_in;
    char ***obj_out_names;
    coDistributedObject ***obj_set_out;
};

void scMultiProcCoviseAppModule(void *p, size_t qwery = 0);
#endif
}
#endif // __COVISE_APP_MODULE_H
