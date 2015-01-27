/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:                                                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Lars Frenzel                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date: aug97                                                            **
\**************************************************************************/

//
// C library stuff
//
//
#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include <string.h>

//
//
// COVISE include stuff
//
//
#include <appl/ApplInterface.h>

#include "HandleSet.h"

void Covise_Set_Handler::Compute(int num_dataports, char **inportnames, char **outportnames)
{
    //
    // parameters:
    //   num_dataports = number of data_ports (not counting the grid-in-object !)
    //   inportnames, (note: port-names and not real-names !)
    //   outportnames, see inportnames
    //

    // needed local variables
    coDistributedObject **data_in;
    char **dataoutname;
    int i;

    // first we need to keep the num_dataports
    INDATAPORTS = num_dataports;

    // allocate required memory
    data_in = new coDistributedObject *[INDATAPORTS];
    dataoutname = new char *[INDATAPORTS];

    // get input-data
    for (i = 0; i < INDATAPORTS; i++)
        if (inportnames[i] == NULL)
            data_in[i] = NULL;
        else
        {
            data_in[i] = Covise_Set_Handler::get_unknown(inportnames[i]);
            if (i == 0 && data_in[i] == NULL)
            { // first input port get's empty data, so just do nothing
                return;
            }
        }

    // get output-data-names
    for (i = 0; i < INDATAPORTS; i++)
        if (outportnames[i] == NULL)
            dataoutname[i] = NULL;
        else
            dataoutname[i] = Covise::get_object_name(outportnames[i]);

    // now start working
    Covise_Set_Handler::handle_objects(data_in, dataoutname);

    // free our allocated mem
    delete[] dataoutname;
    delete[] data_in;

    // and back to covise
}

void Covise_Set_Handler::handle_objects(coDistributedObject **data_in,
                                        char **data_out_name,
                                        coDistributedObject ***data_set_out)
{
    // local variables
    char *dataType;
    int i, j;
    int data_set_num_elem;
    coDistributedObject ***data_in_objs;
    char **bfr;
    coDistributedObject ***data_new_set;
    coDoSet **D_set;
    coDistributedObject **data;
    if (data_in[0] == NULL)
        return;
    // allocate mem
    data_in_objs = new coDistributedObject **[INDATAPORTS];
    bfr = new char *[INDATAPORTS];
    data_new_set = new coDistributedObject **[INDATAPORTS];
    D_set = new coDoSet *[INDATAPORTS];

    // now handle the current object
    dataType = (data_in[0])->getType();
    if (strcmp(dataType, "SETELE") != 0)
    { // no set - so handle this and all corresponding data-objects
        data = ComputeObject(data_in, data_out_name, INDATAPORTS);

        // add objects to set
        for (j = 0; j < INDATAPORTS; j++)
        {
            Covise_Set_Handler::copy_attributes(data_in[j], data[j]);
            if (data_set_out)
            {
                if (data_set_out[j])
                {
                    for (i = 0; data_set_out[j][i]; i++)
                        ;
                    data_set_out[j][i] = data[j];
                    data_set_out[j][i + 1] = NULL;
                }
                else
                {
                    delete data[j];
                }
            }
        }
        delete[] data;

        // that's it
    }
    else if (strcmp(dataType, "SETELE") == 0)
    { // SET !!!

        // get the sets
        for (i = 0; i < INDATAPORTS; i++)
        {
            if (data_in[i] == NULL)
            { // this input-port isn't used
                data_in_objs[i] = NULL;
            }
            else
                data_in_objs[i] = ((coDoSet *)data_in[i])->getAllElements(&data_set_num_elem);
        }

        // for recursion we need data_in_objs [element][port] instead
        // of [port][element] as we have it now
        coDistributedObject ***s_data_in_objs = new coDistributedObject **[data_set_num_elem + 1];
        for (i = 0; i < data_set_num_elem; i++)
        {
            s_data_in_objs[i] = new coDistributedObject *[INDATAPORTS];
            for (j = 0; j < INDATAPORTS; j++)
            {
                if (data_in_objs[j])
                    s_data_in_objs[i][j] = data_in_objs[j][i];
                else
                    s_data_in_objs[i][j] = NULL;
            }
        }

        // create new set (for us)
        for (i = 0; i < INDATAPORTS; i++)
        {
            if (data_in[i] == NULL)
                data_new_set[i] = NULL;
            else
            {
                data_new_set[i] = new coDistributedObject *[data_set_num_elem + 1];
                for (j = 0; j < (data_set_num_elem + 1); j++)
                    data_new_set[i][j] = NULL;
            }
        }

        // now go through the sets
        for (i = 0; i < data_set_num_elem; i++)
        {
            for (j = 0; j < INDATAPORTS; j++)
            {
                if (data_out_name[j] == NULL)
                    bfr[j] = NULL;
                else
                {
                    bfr[j] = new char[strlen(data_out_name[j]) + 10];
                    sprintf(bfr[j], "%s_%d", data_out_name[j], i);
                }
            }
            // recursive
            handle_objects(s_data_in_objs[i], bfr, data_new_set);
            // and free our mem
            for (j = 0; j < INDATAPORTS; j++)
                if (bfr[j])
                    delete[] bfr[j];
        }

        // finally create new Set (for Covise)
        for (i = 0; i < INDATAPORTS; i++)
        {
            if (data_in[i] == NULL || data_out_name[i] == NULL)
                D_set[i] = NULL;
            else
            {
                D_set[i] = new coDoSet(data_out_name[i], data_new_set[i]);
                Covise_Set_Handler::copy_attributes(data_in[i], D_set[i]);
            }
        }

        // we may come from a parent-set (then append current set to it)
        for (j = 0; j < INDATAPORTS; j++)
        {
            if (data_set_out)
            {
                if (data_set_out[j])
                {
                    for (i = 0; data_set_out[j][i]; i++)
                        ;
                    data_set_out[j][i] = D_set[j];
                    data_set_out[j][i + 1] = NULL;
                }
                else
                    delete data_set_out[j];
            }
        }

        // free all mem
        for (j = 0; j < INDATAPORTS; j++)
        {
            if (data_in[j])
                delete ((coDoSet *)data_in[j]);
            if (data_new_set[j])
            {
                for (i = 0; data_new_set[j][i]; i++)
                    delete data_new_set[j][i];
                delete[] data_new_set[j];
            }
        }
    }

    // and even more mem
    delete[] data_in_objs;
    delete[] bfr;
    delete[] data_new_set;
    delete[] D_set;

    // bye
    return;
}

coDistributedObject *Covise_Set_Handler::get_unknown(char *pname)
{
    coDistributedObject *tmp, *r;
    char *tname;

    if (pname == NULL)
        return (NULL);

    // first, get real object name
    tname = Covise::get_object_name(pname);
    // check if it's valid
    if (tname == NULL)
    { // error, no such object
        r = NULL;
    }
    else
    { // create a new object
        tmp = new coDistributedObject(tname);
        r = tmp->createUnknown();
        delete tmp;
    }

    // this method will return the object-pointer or NULL if any
    // error occured
    return (r);
}

coDistributedObject *Covise_Set_Handler::getElement(coDistributedObject *obj, int number)
{
    char *typ;
    coDistributedObject *r;

    typ = obj->getType();
    if (strcmp(typ, "SETELE") == 0)
    { // set
        coDoSet *data_set;
        data_set = (coDoSet *)obj;
        r = data_set->getElement(number);
    }
    else
    {
        if (number == 0)
            r = obj;
        else
            r = NULL;
    }

    // returns an element or NULL if no more is available
    return (r);
}

void Covise_Set_Handler::copy_attributes(coDistributedObject *src, coDistributedObject *tgt)
{
    int n;
    char **attr, **setting;
    n = src->get_all_attributes(&attr, &setting);
    if (n > 0)
        tgt->addAttributes(n, attr, setting);
}

coDistributedObject **Covise_Set_Handler::ComputeObject(coDistributedObject **, char **, int)
{ // dummy
    Covise::sendError("ERROR: you should provide a ComputeObject-function");

    return (NULL);
}
