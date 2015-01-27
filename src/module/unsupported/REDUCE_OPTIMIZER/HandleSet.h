/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LARS_SET_HANDLER_H
#define _LARS_SET_HANDLER_H

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

#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include <string.h>
#include <appl/ApplInterface.h>
using namespace covise;

class Covise_Set_Handler
{
private:
    // functions
    coDistributedObject *get_unknown(char *);
    coDistributedObject *getElement(coDistributedObject *, int);

    // this functions do the work
    void handle_objects(coDistributedObject **data_in,
                        char **data_out_name,
                        coDistributedObject ***data_set_out = NULL);
    void handle_grid_object(coDistributedObject *, coDistributedObject **,
                            char *, char **,
                            coDistributedObject **, coDistributedObject ***);
    void handle_object(coDistributedObject *, char *, coDistributedObject **);

    // variables
    int INDATAPORTS;

public:
    void Compute(int, char **, char **);
    void copy_attributes(coDistributedObject *, coDistributedObject *);
    virtual coDistributedObject **ComputeObject(coDistributedObject **, char **, int);
};
#endif // _LARS_SET_HANDLER_H
