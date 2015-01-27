/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// basic ApplicationModule - class for COVISE
// by Lars Frenzel
//
// history:   	01/16/1998  	started working
//

#if !defined(__COVISE_APP_MODULE_H)
#define __COVISE_APP_MODULE_H

#include <util/coviseCompat.h>
#include <appl/ApplInterface.h>
using namespace covise;

class CoviseAppModule
{
private:
    // callbacks
    static void computeCallback(void *, void *);
    void nonstaticComputeCallback();

    // doing all the work
    void handleObjects(coDistributedObject **obj_in, char **obj_out_names,
                       coDistributedObject ***obj_set_out = NULL);
    void preHandleObjects(coDistributedObject **obj_in);

    // names of ports (NULL for none / unused)
    char **inport_names;
    char **outport_names;
    int num_inports;
    int num_outports;

    // should we copy set-attributes or not ?
    int copy_attributes_flag;

    // is Multiprocessing supported by the module ?
    int multiprocessing_flag;
    int current_level;
    int pre_multiprocessing_flag;

    // should we compute timesteps/multiblock
    int compute_timesteps;
    int compute_multiblock;

protected:
    // get a coDistributedObject by giving the port-name
    coDistributedObject *getUnknown(char *);

public:
    // basic stuff
    CoviseAppModule();
    ~CoviseAppModule();

    // handling attributes
    void setCopyAttributes(int val)
    {
        copy_attributes_flag = val;
        return;
    };
    void copyAttributes(coDistributedObject *, coDistributedObject *);

    // setting the port-names
    void setPortNames(char **, char **);

    // setting up the callback-functions
    void setCallbacks();

    // set, if the module is multiprocessing-aware
    void setMultiprocessing(int val)
    {
        multiprocessing_flag = val;
        return;
    };
    void setPreMultiprocessing(int val)
    {
        pre_multiprocessing_flag = val;
        return;
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

    // virtual stuff - may be provided by programmer
    virtual void preCompute(coDistributedObject **){};
    virtual coDistributedObject **compute(coDistributedObject **, char **)
    {
        return NULL;
    };
    virtual void postCompute(coDistributedObject **, coDistributedObject **){};
};

CoviseAppModule::CoviseAppModule()
{
    // port names not yet defined
    inport_names = NULL;
    outport_names = NULL;
    num_inports = 0;
    num_outports = 0;

    // allways copy set-attributes as default
    copy_attributes_flag = 1;

    // no multiprocessing by default
    multiprocessing_flag = 0;
    pre_multiprocessing_flag = 0;
    current_level = 0;

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

void CoviseAppModule::copyAttributes(coDistributedObject *src, coDistributedObject *tgt)
{
    int n;
    char **name, **setting;

    n = src->get_all_attributes(&name, &setting);
    if (n > 0)
        tgt->addAttributes(n, name, setting);

    return;
}

coDistributedObject *CoviseAppModule::getUnknown(char *portname)
{
    coDistributedObject *tmp = NULL, *r = NULL;
    char *objname = NULL;

    if (portname == NULL)
        return (NULL);

    // get object name
    objname = Covise::get_object_name(portname);

    // is there a object with this name ?
    if (objname)
    {
        // get object
        tmp = new coDistributedObject(objname);
        r = tmp->createUnknown();

        // clean up
        delete tmp;
    }
    else
        return (NULL);

    // done
    return (r);
}

void CoviseAppModule::setPortNames(char **input, char **output)
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

    coDistributedObject **obj_in = NULL;
    char **obj_out_names = NULL;
    int i;

    // allocate memory
    if (inport_names)
    {
        obj_in = new coDistributedObject *[num_inports + 1];
        obj_in[num_inports] = NULL;

        // get input
        for (i = 0; i < num_inports; i++)
            obj_in[i] = getUnknown(inport_names[i]);

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

    // done
    return;
}

void CoviseAppModule::handleObjects(coDistributedObject **obj_in, char **obj_out_names, coDistributedObject ***obj_set_out)
{
    // temp
    coDistributedObject **usr_return = NULL;
    coDistributedObject ***obj_new_set = NULL;

    int t;
    int compute_flag;

    // counters
    int i, j;

    // input
    char *dataType;
    int num_set_elem;

    coDistributedObject ***obj_set_in = NULL;

    // output
    char **obj_set_out_names = NULL;
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
                    obj_set_out[i][j + 1] = NULL;
                }
                else
                    // no parent set, so we don't need the object any more
                    delete usr_return[i];
            }
            else
                // not a set, so we don't need the object any longer
                delete usr_return[i];
        }
        delete[] usr_return;

        // done
    }
    else
    {
        // we have a set, so the trouble starts right here

        // allocate memory
        obj_set_in = new coDistributedObject **[num_inports];

        obj_set_out_names = new char *[num_outports];
        obj_current_set_out = new coDoSet *[num_outports];
        obj_new_set = new coDistributedObject **[num_outports];

        // get the set(s)
        t = -1;
        for (i = 0; i < num_inports; i++)
        {
            // check if this port is used
            if (obj_in[i])
            {
                obj_set_in[i] = ((coDoSet *)obj_in[i])->getAllElements(&num_set_elem);
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
        coDistributedObject ***obj_sorted = new coDistributedObject **[num_set_elem + 1];
        for (i = 0; i < num_set_elem; i++)
        {
            obj_sorted[i] = new coDistributedObject *[num_inports];
            for (j = 0; j < num_inports; j++)
            {
                if (obj_set_in[j])
                    obj_sorted[i][j] = obj_set_in[j][i];
                else
                    obj_sorted[i][j] = NULL;
            }
        }
        obj_sorted[num_set_elem] = new coDistributedObject *[num_inports];
        for (i = 0; i < num_inports; i++)
            obj_sorted[num_set_elem][i] = NULL;

        // create empty sets for output-objects
        for (i = 0; i < num_outports; i++)
        {
            obj_new_set[i] = new coDistributedObject *[num_set_elem + 1];
            for (j = 0; j < num_set_elem + 1; j++)
                obj_new_set[i][j] = NULL;
        }

        // now compute the output (recursive)
        for (i = 0; i < num_set_elem; i++)
        {
            // compute output-object-names
            for (j = 0; j < num_outports; j++)
            {
                // check if this port is used
                if (obj_out_names[j])
                {
                    obj_set_out_names[j] = new char[strlen(obj_out_names[j]) + 10];
                    sprintf(obj_set_out_names[j], "%s_%d", obj_out_names[j], i);
                }
                else
                    obj_set_out_names[j] = NULL;
            }

            // recursion takes place here, we may use
            // multiprocessing if enabled
            if (current_level == 1 && multiprocessing_flag)
            {
                fprintf(stderr, "CoviseAppModule: multiprocessing not implemented\n");
                fprintf(stderr, "                  (auto-fallback to singleprocessing)\n");
                handleObjects(obj_sorted[i], obj_set_out_names, obj_new_set);
            }
            else
                handleObjects(obj_sorted[i], obj_set_out_names, obj_new_set);

            // clean up output-object-names
            for (j = 0; j < num_outports; j++)
                if (obj_set_out_names[j])
                    delete[] obj_set_out_names[j];
        }

        // create output set
        for (i = 0; i < num_outports; i++)
        {
            if (obj_out_names[i])
            {
                obj_current_set_out[i] = new coDoSet(obj_out_names[i], obj_new_set[i]);

                // copy set-attributes if flag is set
                if (copy_attributes_flag)
                    copyAttributes(obj_in[0], obj_current_set_out[i]);
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
                else
                    delete obj_current_set_out[i];
            }
            else
                delete obj_current_set_out[i];
        }
        delete[] obj_current_set_out;

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
        delete[] obj_set_out_names;
        delete[] obj_new_set;

        for (i = 0; i < num_set_elem + 1; i++)
            delete[] obj_sorted[i];
        delete[] obj_sorted;

        // done
    }

    // weïre finished with this level
    current_level--;

    // done
    return;
}

//obj_in )
void CoviseAppModule::preHandleObjects(coDistributedObject **)
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
#endif // __COVISE_APP_MODULE_H
