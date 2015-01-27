/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CoviseAppModule.h"

#include <config/CoviseConfig.h>
#include <do/coDistributedObject.h>
#include <do/coDoSet.h>

#undef TIMING

#ifdef TIMING
#include <sys/time.h>
#include <time.h>

#ifdef __sgi
#define BEST_TIMER CLOCK_SGI_CYCLE
#else
#define BEST_TIMER CLOCK_REALTIME
#endif
#endif

using namespace covise;

#if defined(__sgi)

void scMultiProcCoviseAppModule(void *userdata, size_t)
{
    CoviseAppModuleInfo *appInf;
    int k;

    // get parameters
    appInf = (CoviseAppModuleInfo *)userdata;

    // call the module : round-robin distribution
    for (k = appInf->first; k < appInf->max; k += appInf->step)
    {

        appInf->mod->handleObjects(appInf->obj_in[k],
                                   appInf->obj_out_names[k],
                                   appInf->obj_set_out, k);
    }

    // and don't forget the barrier
    barrier(appInf->barrier, appInf->num);
}
#endif

CoviseAppModule::CoviseAppModule()
{
    // port names not yet defined
    inport_names = NULL;
    outport_names = NULL;
    num_inports = 0;
    num_outports = 0;

    // allways copy set-attributes as default
    copy_attributes_flag = 1;
    or_copy_addAttributes_flag = 0; // sl: this value
    //     ensures that other
    //     modules that do not know about
    //     or_copy_addAttributes_flag will
    //     not be affected

    // no multiprocessing by default
    multiprocessing_flag = 0;
    pre_multiprocessing_flag = 0;
    current_level = 0;
#if defined(__sgi)
    thisGroupsBarrier = NULL;
    thisGroupsMutex = NULL;
#endif

    // by default we do everything for the programmer
    compute_timesteps = 1;
    compute_multiblock = 1;

    // done
    return;
}

CoviseAppModule::~CoviseAppModule()
{
    int i;

    // clean up
    if (inport_names)
    {
        for (i = 0; inport_names[i]; i++)
            delete[] inport_names[i];
        delete[] inport_names;
    }

    if (outport_names)
    {
        for (i = 0; outport_names[i]; i++)
            delete[] outport_names[i];
        delete[] outport_names[i];
    }

    // done
    return;
}

void CoviseAppModule::copyAttributes(const coDistributedObject *src, coDistributedObject *tgt)
{
    int n;
    const char **name, **setting;

    if (src && tgt)
    {
        n = src->getAllAttributes(&name, &setting);
        if (n > 0)
            tgt->addAttributes(n, name, setting);
    }

    return;
}

const coDistributedObject *CoviseAppModule::getUnknown(char *portname, int &errflag)
{
    const coDistributedObject *r = NULL;
    char *objname = NULL;

    if (portname == NULL)
        return (NULL);

    // get object name
    objname = Covise::get_object_name(portname);

    // not even found the name
    if (!objname)
    {
        return NULL;
    }

    // is there a object with this name ?  If not: no error, if TOLERANT
    else
    {
        r = coDistributedObject::createFromShm(objname);
    }

    if (!r)
    {
#ifndef TOLERANT
        Covise::sendError("Port name incorrect: could not retrieve Object name");
#endif
        errflag = 1;
    }

    // done
    return r;
}

void CoviseAppModule::setPortNames(const char **input, const char **output)
{
    int i;

    if (input)
    {
        for (i = 0; input[i]; i++)
            ;
        inport_names = new char *[i + 1];

        for (i = 0; input[i]; i++)
        {
            inport_names[i] = new char[strlen(input[i]) + 1];
            strcpy(inport_names[i], input[i]);
        }

        inport_names[i] = NULL;
        num_inports = i;
    }
    else
        inport_names = NULL;

    if (output)
    {
        for (i = 0; output[i]; i++)
            ;
        outport_names = new char *[i + 1];

        for (i = 0; output[i]; i++)
        {
            outport_names[i] = new char[strlen(output[i]) + 1];
            strcpy(outport_names[i], output[i]);
        }

        outport_names[i] = NULL;
        num_outports = i;
    }
    else
        outport_names = NULL;

    // done
    return;
}

void CoviseAppModule::setCallbacks()
{
    // compute-callback
    Covise::set_start_callback(CoviseAppModule::computeCallback, this);

    // done
    return;
}

void CoviseAppModule::computeCallback(void *userData, void *)
{
    CoviseAppModule *thisApp = (CoviseAppModule *)userData;
    thisApp->nonstaticComputeCallback();
}

void CoviseAppModule::nonstaticComputeCallback()
{
#ifdef TIMING
    struct timespec starttime;
    clock_gettime(BEST_TIMER, &starttime);
#endif

    const coDistributedObject **obj_in = NULL;
    char **obj_out_names = NULL;
    int i;

    // allocate memory
    if (inport_names)
    {
        obj_in = new const coDistributedObject *[num_inports + 1];
        obj_in[num_inports] = NULL;

        // get input
        int errflag = 0;
        for (i = 0; i < num_inports; i++)
            obj_in[i] = getUnknown(inport_names[i], errflag);

        if (errflag) // messages submitted by getUnknown, supress for TOLERANT
            return;
        // done
    }

    if (outport_names)
    {
        obj_out_names = new char *[num_outports + 1];
        obj_out_names[num_outports] = NULL;

        // get names
        for (i = 0; i < num_outports; i++)
            obj_out_names[i] = Covise::get_object_name(outport_names[i]);

        // done
    }

    // start working
    current_level = 0;
    preHandleObjects(obj_in);
    current_level = 0;
    handleObjects(obj_in, obj_out_names, NULL);

    // clean up
    if (obj_in)
        delete[] obj_in;

    if (obj_out_names)
        delete[] obj_out_names;

#ifdef TIMING
    struct timespec endtime;
    clock_gettime(BEST_TIMER, &endtime);
    double time_needed = (endtime.tv_sec - starttime.tv_sec)
                         + 1e-9 * (endtime.tv_nsec - starttime.tv_nsec);

    cerr << time_needed << " sec." << endl;
#endif

    // done
    return;
}

void CoviseAppModule::handleObjects(const coDistributedObject **obj_in, char **obj_out_names,
                                    coDistributedObject ***obj_set_out, int set_id, int /*num_proc_wait*/)
{
    // temp
    coDistributedObject **usr_return = NULL;
    coDistributedObject ***obj_new_set = NULL;

    int t;
    int compute_flag;
    int num_proc = 0;

    // counters
    int i, j;

    // input
    const char *dataType;
    int num_set_elem;

    const coDistributedObject *const **obj_set_in = NULL;

    // output
    char ***obj_set_out_names = NULL;
    coDoSet **obj_current_set_out = NULL;

    // check for error
    if (obj_in == NULL || obj_in[0] == NULL)
        return;

    // remember to update level
    current_level++;

    // handle current object
    dataType = (obj_in[0])->getType();

    // see if we have to do the work or the user does it
    if (strcmp(dataType, "SETELE") == 0)
    {
        // we have a set, do we have to compute it ?
        if ((obj_in[0])->getAttribute("TIMESTEP"))
            // we have transient data
            compute_flag = compute_timesteps;
        else
            // multiblock
            compute_flag = compute_multiblock;
    }
    else
        // no set
        compute_flag = 0;

    if (!compute_flag)
    {
        // no set
        // we may have to use another method if multiprocessing is enabled
        if (multiprocessing_flag)
        {
            //for( i=0; i<num_outports; i++ )
            //   fprintf(stderr, "%d: %s\n", i, obj_out_names[i]);

            // call the user function
            usr_return = compute(obj_in, obj_out_names);

            // do post-processing
            postCompute(obj_in, usr_return);

            // 07.02.2000
            lockNewObjects();

            // add the objects to the output
            for (i = 0; i < num_outports; i++)
            {
                if (obj_set_out)
                {
                    if (obj_set_out[i])
                    {
                        obj_set_out[i][set_id] = usr_return[i];
                        if (copy_attributes_flag)
                            if (i < num_inports)
                            {
                                copyAttributes(obj_in[i], obj_set_out[i][set_id]);
                            }
                    }
                    else
                        delete usr_return[i];
                }
                else
                    delete usr_return[i];
            }

            delete[] usr_return;

            // 07.02.2000
            unlockNewObjects();

            // ok
        }
        else
        {
            // no set, so just call the user-function
            usr_return = compute(obj_in, obj_out_names);

            // do post-processing
            postCompute(obj_in, usr_return);

            // and add the objects to the output
            for (i = 0; i < num_outports; i++)
            {
                if (obj_set_out)
                {
                    // see if we have a set above this one
                    if (obj_set_out[i])
                    {
                        // we have one, so add the object to the set
                        for (j = 0; obj_set_out[i][j]; j++)
                            ;
                        obj_set_out[i][j] = usr_return[i];
                        if (copy_attributes_flag)
                            if (i < num_inports)
                            {
                                copyAttributes(obj_in[i], obj_set_out[i][j]);
                            }
                        obj_set_out[i][j + 1] = NULL;
                    }
                    else
                        // no parent set, so we don't need the object any more
                        delete usr_return[i];
                }
                else
                {
                    // not a set, so we don't need the object any longer
                    if (obj_in[i] && usr_return[i])
                        copyAttributes(obj_in[i], usr_return[i]);
                    delete usr_return[i];
                }
            }
            delete[] usr_return;
        }
        // done
    }
    else
    {
        // we have a set, so the trouble starts right here
        // allocate memory
        obj_set_in = new const coDistributedObject *const *[num_inports];

        //obj_set_out_names = new char*[num_outports];
        obj_current_set_out = new coDoSet *[num_outports];
        obj_new_set = new coDistributedObject **[num_outports];

        // get the set(s)
        t = -1;
        for (i = 0; i < num_inports; i++)
        {
            // check if this port is used
            // we:01/07/04 ... and is a set object
            const coDoSet *set = dynamic_cast<const coDoSet *>(obj_in[i]);
            if (set)
            {
                obj_set_in[i] = set->getAllElements(&num_set_elem);
                if (t == -1)
                    t = num_set_elem;
                if (num_set_elem != t)
                {
                    // serious error -> data inconsistency
                    Covise::sendError("CoviseAppModule: stale object structure");
                    return;
                }
            }
            else
                obj_set_in[i] = NULL;
        }

        // for recursion we require the objects
        // to be [element][port] instead of [port][element] as we have them right now
        const coDistributedObject ***obj_sorted = new const coDistributedObject **[num_set_elem + 1];
        for (i = 0; i < num_set_elem; i++)
        {
            obj_sorted[i] = new const coDistributedObject *[num_inports];
            for (j = 0; j < num_inports; j++)
            {
                if (obj_set_in[j])
                    obj_sorted[i][j] = obj_set_in[j][i];
                else
                    obj_sorted[i][j] = NULL;
            }
        }
        obj_sorted[num_set_elem] = new const coDistributedObject *[num_inports];
        for (i = 0; i < num_inports; i++)
            obj_sorted[num_set_elem][i] = NULL;

        // create empty sets for output-objects
        for (i = 0; i < num_outports; i++)
        {
            obj_new_set[i] = new coDistributedObject *[num_set_elem + 1];
            for (j = 0; j < num_set_elem + 1; j++)
                obj_new_set[i][j] = NULL;
        }

        // compute object names
        obj_set_out_names = new char **[num_set_elem];
        for (i = 0; i < num_set_elem; i++)
        {
            obj_set_out_names[i] = new char *[num_outports];
            for (j = 0; j < num_outports; j++)
            {
                if (obj_out_names[j])
                {
                    obj_set_out_names[i][j] = new char[strlen(obj_out_names[j]) + 10];
                    sprintf(obj_set_out_names[i][j], "%s_%d", obj_out_names[j], i);
                }
                else
                    obj_set_out_names[i][j] = NULL;
            }
        }

        // see if we are forced to not use multiprocessing
        if (compute_timesteps && compute_multiblock && current_level == 1 && multiprocessing_flag)
        {
            multiprocessing_flag = coCoviseConfig::isOn("System.HostInfo.AllowSMP", true);
        }
        if (multiprocessing_flag && !num_proc)
        {
            num_proc = coCoviseConfig::getInt("System.HostInfo.NumProcessors", 2);
            if (num_proc <= 0)
                num_proc = 2;
#ifdef TIMING
            cerr << num_proc << " Processors: ";
#endif
        }

// check if this system supports multiprocessing at all
#if defined(__sgi)
// yes
#else
        // no
        multiprocessing_flag = 0;
#endif

        // now compute the output (recursive)
        if ((multiprocessing_flag) && (num_proc > 1))
        {
// here we go

#if defined(__sgi)

            char *arenaName = NULL;
            usptr_t *myArena;
            barrier_t *myBarrier;

            myArena = NULL;
            myBarrier = NULL;

            // first we need an arena
            arenaName = tempnam("", "LOCK");
            usconfig(CONF_INITUSERS, num_proc + 1);
            usconfig(CONF_ARENATYPE, US_SHAREDONLY);
            myArena = usinit(arenaName);

            // and a barrier
            myBarrier = new_barrier(myArena);
            init_barrier(myBarrier);
            thisGroupsBarrier = myBarrier;

            // and a mutex
            // ?!?!?!?!
            thisGroupsMutex = (ulock_t *)usnewlock(myArena);
            if (thisGroupsMutex == NULL)
            {
                fprintf(stderr, "ARGH !!!\n");
            }

            // load the parameters
            CoviseAppModuleInfo **info;
            info = new CoviseAppModuleInfo *[num_proc];
            j = num_set_elem / num_proc;
            for (i = 0; i < num_proc; i++)
            {
                info[i] = new CoviseAppModuleInfo();

                info[i]->barrier = myBarrier;
                info[i]->num = num_proc;

                /***********************************
            //  old: distribute 1,2,3 + 4,5,6 + 7,8,9 + ...
            if( i )
               info[i]->f = info[i-1]->t;
            else
               info[i]->f = 0;
            if( i==num_proc-1 )
               info[i]->t = num_set_elem;
            else
               info[i]->t = (i+1)*j;
            ************************************/

                // Now:  distribute 1,5,9,13 + 2,6,10,14 + 3,7,11,15 + 4,8,12,16

                info[i]->first = i;
                info[i]->max = num_set_elem;
                info[i]->step = num_proc;

                info[i]->mod = this;
                info[i]->obj_in = obj_sorted;
                info[i]->obj_out_names = obj_set_out_names;
                info[i]->obj_set_out = obj_new_set;

                // and launch the process
                if (i != num_proc - 1)
                    sprocsp(scMultiProcCoviseAppModule, PR_SALL, (void *)info[i], NULL, 1000000);
                //if( i!=num_proc-1 )
                //   scMultiProcCoviseAppModule( info[i] );
                //fprintf(stderr, "%d\n", i);
            }

            // do something, too
            scMultiProcCoviseAppModule(info[num_proc - 1]);

            // clean up the parameters
            for (i = 0; i < num_proc; i++)
                delete info[i];
            delete[] info;

            // clean up the mutex
            usfreelock(thisGroupsMutex, myArena);

            // clean up the barrier and arena
            free_barrier(myBarrier);
            if (myArena)
                unlink(arenaName);
#endif

            //for( i=0; i<num_set_elem; i++ )
            //   handleObjects( obj_sorted[i], obj_set_out_names[i], obj_new_set, i );
        }
        else
        {
            // no multiprocessing
            for (i = 0; i < num_set_elem; i++)
                handleObjects(obj_sorted[i], obj_set_out_names[i], obj_new_set, i);
        }

        // create output set
        for (i = 0; i < num_outports; i++)
        {
            if (obj_out_names[i])
            {
                // check if we have to create a set at all
                if (obj_new_set[i][0])
                {
                    obj_current_set_out[i] = new coDoSet(coObjInfo(obj_out_names[i]), obj_new_set[i]);

                    // copy set-attributes if flag is set
                    // sl: or_copy_addAttributes_flag added
                    if (copy_attributes_flag || or_copy_addAttributes_flag)
                        copyAttributes(obj_in[0], obj_current_set_out[i]);
                }
                else
                    obj_current_set_out[i] = NULL;
            }
            else
                obj_current_set_out[i] = NULL;
        }

        // post-processing
        postCompute(obj_in, (coDistributedObject **)obj_current_set_out);

        // if we come from a parent set, then append current set to it
        for (i = 0; i < num_outports; i++)
        {
            if (obj_set_out)
            {
                // is there one for this set
                if (obj_set_out[i])
                {
                    // add the object
                    for (j = 0; obj_set_out[i][j]; j++)
                        ;
                    obj_set_out[i][j] = obj_current_set_out[i];
                    obj_set_out[i][j + 1] = NULL;
                }
                else if (obj_current_set_out[i])
                    delete obj_current_set_out[i];
            }
            else if (obj_current_set_out[i])
                delete obj_current_set_out[i];
        }
        delete[] obj_current_set_out;

        // clean up
        for (i = 0; i < num_set_elem; i++)
        {
            for (j = 0; j < num_outports; j++)
            {
                if (obj_set_out_names[i][j])
                    delete[] obj_set_out_names[i][j];
            }
            delete[] obj_set_out_names[i];
        }
        delete[] obj_set_out_names;

        // clean up
        for (i = 0; i < num_inports; i++)
            if (obj_in[i])
                delete ((coDoSet *)obj_in[i]);

        for (i = 0; i < num_outports; i++)
        {
            if (obj_new_set[i])
            {
                for (j = 0; obj_new_set[i][j]; j++)
                    delete obj_new_set[i][j];
                delete[] obj_new_set[i];
            }
        }

        delete[] obj_set_in;
        delete[] obj_new_set;

        for (i = 0; i < num_set_elem + 1; i++)
            delete[] obj_sorted[i];
        delete[] obj_sorted;

        // done
    }

    // we're finished with this level
    current_level--;

    // done
    return;
}

//obj_in )
void CoviseAppModule::preHandleObjects(const coDistributedObject **)
{
    /*

   int compute_flag;
   int t;
   int i, j;
   int num_set_elem;
   coDistributedObject ***obj_set_in = NULL;
   char *dataType = NULL;

   // check for error
   if( obj_in==NULL || obj_in[0]==NULL )
   return;

   // remember to update level
   current_level++;

   // handle current object
   dataType = (obj_in[0])->getType();

   // see if we have to do the work or the user does it
   if( strcmp(dataType, "SETELE") == 0 )
   {
   // we have a set, do we have to compute it ?
   if( (obj_in[0])->getAttribute("TIMESTEP") )
   // we have transient data
   compute_flag = compute_timesteps;
   else
   // multiblock
   compute_flag = compute_multiblock;
   }
   else
   // no set
   compute_flag = 0;

   if( !compute_flag )
   {
   // no set, so just call the user-function
   preCompute( obj_in );
   }
   else
   {
   // allocate memory
   obj_set_in = new coDistributedObject**[num_inports];

   // get the set(s)
   t = -1;
   for( i=0; i<num_inports; i++ )
   {
   // check if this port is used
   if( obj_in[i] )
   {
   obj_set_in[i] = ((coDoSet *)obj_in[i])->getAllElements( &num_set_elem );
   if( t == -1 )
   t = num_set_elem;
   if( num_set_elem != t )
   {
   // serious error -> data inconsistency
   Covise::sendError( "CoviseAppModule: stale object structure" );
   return;
   }
   }
   else
   obj_set_in[i] = NULL;
   }

   // for recursion we require the objects
   // to be [element][port] instead of [port][element] as we have them right now
   coDistributedObject ***obj_sorted = new coDistributedObject**[num_set_elem+1];
   for( i=0; i<num_set_elem; i++ )
   {
   obj_sorted[i] = new coDistributedObject*[num_inports];
   for( j=0; j<num_inports; j++ )
   {
   if( obj_set_in[j] )
   obj_sorted[i][j] = obj_set_in[j][i];
   else
   obj_sorted[i][j] = NULL;
   }
   }
   obj_sorted[num_set_elem] = new coDistributedObject*[num_inports];
   for( i=0; i<num_inports; i++ )
   obj_sorted[num_set_elem][i] = NULL;

   // now compute the output (recursive)
   for( i=0; i<num_set_elem; i++ )
   {
   // recursion takes place here, we may use
   // multiprocessing if enabled
   if( current_level==1 && pre_multiprocessing_flag )
   {
   fprintf(stderr, "CoviseAppModule: multiprocessing not implemented\n");
   fprintf(stderr, "                  (auto-fallback to singleprocessing)\n");
   preHandleObjects( obj_sorted[i] );
   }
   else
   preHandleObjects( obj_sorted[i] );
   }

   // clean up
   for( i=0; i<num_inports; i++ )
   if( obj_in[i] )
   {
   obj_in[i]->incRefCount();
   delete ((coDoSet *)obj_in[i]);
   }

   delete[] obj_set_in;

   for( i=0; i<num_set_elem+1; i++ )
   delete[] obj_sorted[i];
   delete[] obj_sorted;
   }

   // weïre finished with this level
   current_level--;

   */

    // done
    return;
}

void CoviseAppModule::lockNewObjects()
{
#if defined(__sgi)
    if (thisGroupsMutex)
    {
        if (ussetlock(thisGroupsMutex) != 1)
        {
            fprintf(stderr, "couldn't lock\n");
        }
        //else fprintf(stderr, "LOCK\n");
    }
#endif
}

void CoviseAppModule::unlockNewObjects()
{
#if defined(__sgi)
    if (thisGroupsMutex)
    {
        if (usunsetlock(thisGroupsMutex) != 0)
        {
            fprintf(stderr, "couldn't unlock\n");
        }
        //else fprintf(stderr, "UNLOCK\n");
    }
#endif
}
