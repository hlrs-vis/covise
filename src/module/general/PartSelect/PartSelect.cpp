/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE PartSelect application module                           **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa                                                **
 **                                                                        **
 **                                                                        **
 ** Date:  05.05.01  V1.0                                                  **
\**************************************************************************/

#include "PartSelect.h"
#include <util/coviseCompat.h>
#include <do/coDoSet.h>

PartSelect::PartSelect(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Select one Part")
{
    char buffer[10];
    int i;
    //parameters

    p_numbers = addStringParam("numbers", "numbers of selected parts");
    p_numbers->setValue("0");

    p_inPort = new coInputPort *[MAX_INPUT_PORTS];
    p_outPort = new coOutputPort *[MAX_INPUT_PORTS];

    //ports
    p_inPort[0] = addInputPort("inport_1", "StructuredGrid|Float|Vec3|UnstructuredGrid|Float|Vec3|Geometry|Polygons|Lines|Points|TriangleStrips|IntArr", "input object");

    for (i = 1; i < MAX_INPUT_PORTS; i++)
    {
        sprintf(buffer, "inport_%d", i + 1);
        p_inPort[i] = addInputPort(buffer, "StructuredGrid|Float|Vec3|UnstructuredGrid|Float|Vec3|Geometry|Polygons|Lines|Points|TriangleStrips|IntArr", "input object");
        p_inPort[i]->setRequired(0);
    }

    for (i = 0; i < MAX_INPUT_PORTS; i++)
    {
        sprintf(buffer, "outport_%d", i + 1);
        p_outPort[i] = addOutputPort(buffer, "StructuredGrid|Float|Vec3|UnstructuredGrid|Float|Vec3|Geometry|Polygons|Lines|Points|TriangleStrips|IntArr", "output object");
        if (i > 0)
            p_outPort[i]->setDependencyPort(p_inPort[i]);
    }

    setComputeTimesteps(0);
    setComputeMultiblock(1);
}

PartSelect::~PartSelect()
{
    delete[] p_inPort;
    delete[] p_outPort;
}

int PartSelect::compute(const char *)
{
    int i, n, nb, f;
    int num_elem_in;

    int num_elem_remove = 0;
    int *elem_remove = 0;

    // get parameter

    const char *bfr = p_numbers->getValue();

    char selection[1024];
    char bfr_corrected[1024];

    int i_correct = 0;
    for (i = 0; bfr[i]; i++)
    {
        if (bfr[i] == '\177')
            continue;
        bfr_corrected[i_correct] = bfr[i];
        i_correct++;
    }
    bfr_corrected[i_correct] = '\0';

    for (i = 0; bfr_corrected[i]; i++)
        selection[i] = (bfr_corrected[i] == '/') ? ' ' : bfr_corrected[i];
    selection[i] = '\0';

    // sl: compute how many elements to remove and build list
    {
        int dummy;
        std::istringstream input(selection);
        while (input >> dummy)
            num_elem_remove++;
    }

    if (num_elem_remove)
    {
        std::istringstream input(selection);
        elem_remove = new int[num_elem_remove];
        n = 0;
        while (input >> elem_remove[n++])
        {
        }
    }

    /// we may not select multiple times the same part
    int j;
    for (i = 1; i < num_elem_remove; i++)
    {
        // find matching elements
        for (j = 0; j < i; j++)
        {
            if (elem_remove[i] == elem_remove[j])
            {
                // move rest of the field one forward
                for (j = i + 1; j < num_elem_remove; j++)
                    elem_remove[j - 1] = elem_remove[j];
                --num_elem_remove; // one element less
                --i; // redo loop for i
                break;
            }
        }
    }

    //objects

    for (j = 0; j < MAX_INPUT_PORTS; j++)
    {

        const coDistributedObject *obj = p_inPort[j]->getCurrentObject();
        if (obj != NULL)
        {
            if (obj->isType("SETELE"))
            {
                const coDistributedObject **set_elements;
                const coDistributedObject *const *in_set_elements = ((coDoSet *)obj)->getAllElements(&num_elem_in);
                // sl: It is possible that some elements of the selection are out of bounds...
                int out_of_bounds = 0;
                for (i = 0; i < num_elem_remove; ++i)
                    if (elem_remove[i] < 0 || elem_remove[i] >= num_elem_in)
                        out_of_bounds++;

                set_elements = new const coDistributedObject *[num_elem_in - num_elem_remove + out_of_bounds + 1];
                set_elements[num_elem_in - num_elem_remove + out_of_bounds] = NULL;

                // create output
                n = 0;
                for (i = 0; i < num_elem_in; i++)
                {
                    // check if object has to be removed
                    f = 0;
                    for (nb = 0; nb < num_elem_remove; nb++)
                        if (elem_remove[nb] == i)
                            f = 1;

                    if (!f)
                    {
                        // keep object
                        set_elements[n] = in_set_elements[i];
                        in_set_elements[i]->incRefCount();
                        n++;
                    }
                }
                coDoSet *out_obj = new coDoSet(p_outPort[j]->getObjName(), set_elements);
                if (!out_obj)
                {
                    sendError("Failed to create object '%s' for port '%s' ", p_outPort[j]->getObjName(), p_outPort[j]->getName());
                    return FAIL;
                }
                p_outPort[j]->setCurrentObject(out_obj);
            }
            else
            {
                sendError("There is nothing to select at port '%s' because the object type is not SETELE", p_inPort[j]->getName());
                return FAIL;
            }
        }
    }
    return SUCCESS;
}

MODULE_MAIN(Filter, PartSelect)
