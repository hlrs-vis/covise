/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
// coSimpleModule --- by Lars Frenzel
//
// 09/22/1999  -  started working
// 11/18/1999  -  redesigned to be api 2.0 compatible
// 06/03/2001  -  COVER interaction

#if !defined(__CO_SIMPLE_MODULE)
#define __CO_SIMPLE_MODULE

#include <appl/ApplInterface.h>
#include "coModule.h"

namespace covise
{

class coSimpleModule;

class APIEXPORT coSimpleModule : public coModule
{
private:
    int compute_timesteps;
    int compute_multiblock;

    int copy_attributes_flag;

    // sl
    int copy_attributes_non_set_flag;

    coInputPort **originalInPorts;
    coOutputPort **originalOutPorts;
    int numInPorts, numOutPorts;

    void swapObjects(coInputPort **inPorts, coOutputPort **outPorts);

    // return CONTINUE_PIPELINE or STOP_PIPELINE on error
    int handleObjects(coInputPort **inPorts, coOutputPort **outPorts);

    // COVER interaction
    char INTattribute[300];
    int cover_interaction_flag; // 0: turned off, 1: turned on

    // information if object is part of a set
    int object_level;

    // currently within a block of multiblock data?
    int multiblock_flag;

    // currently within a timestep?
    bool timestep_flag;

    // number of current element in each currently traversed level of set hierarchy
    std::vector<int> element_counter;

    // number of elements in each currently traversed level of set hierarchy
    std::vector<int> num_elements;

protected:
    virtual void localCompute(void *callbackData);

    // sl: from which input object are the attributes of
    //     the i-th output object copied? The default
    //     assumption that it is from the i-th input object
    //     may have to be modified by many a module
    virtual void copyAttributesToOutObj(coInputPort **, coOutputPort **, int);

    // sl: Ergaenzung fuer IsoSurfaceP
    //     und vielleicht auch angebracht fuer andere Module

    // called once before running across the tree
    virtual void preHandleObjects(coInputPort **){};

    // called before a level with sets is opened
    virtual void setIterator(coInputPort **, int){};

    // called for each set level : overload it
    virtual int compute(const char *port)
    {
        (void)port;
        return CONTINUE_PIPELINE;
    }

    // called once after running across the tree
    virtual void postHandleObjects(coOutputPort **){};

    int portLeader; // which input port is inspected in handleObjects
    // when deciding whether there is anything to compute

    // get current level in object hierarchy, only call from compute()
    int getObjectLevel() const;

    // get number of current set element at level, only call from compute()
    int getElementNumber(int level = -1) const;

    // get total number of set elements at level, only call from compute()
    int getNumberOfElements(int level = -1) const;

public:
    coSimpleModule(int argc, char *argv[], const char *desc = NULL, bool propagate = false);

    // are we currently handling multiblock data?
    int isPartOfMultiblock()
    {
        return multiblock_flag;
    }

    // are we currently handling timestep data?
    bool isTimestep()
    {
        return (timestep_flag != 0);
    }

    /// whether object is part of a set or not
    int isPartOfSet()
    {
        return (object_level > 0);
    }

    /// set this if you want to add an FEEDBACK attribute string to the highest set level
    void setInteraction(const char *string)
    {
        strcpy(INTattribute, string);
        cover_interaction_flag = 1;
    }

    /// set this if you need to handle timesteps yourself (default=0)
    void setComputeTimesteps(const int v)
    {
        compute_timesteps = v;
        return;
    };

    /// set this if you need to handle multiblock yourself (default=0)
    void setComputeMultiblock(const int v)
    {
        compute_multiblock = v;
        return;
    };

    /// copy attributes
    void copyAttributes(coDistributedObject *tgt, const coDistributedObject *src) const;

    /// set if you dont want to keep track of attributes yourself (default=1)
    void setCopyAttributes(const int v)
    {
        copy_attributes_flag = v;
        return;
    };

    /// set if you dont want to keep track of attributes yourself for
    /// non-set elements (default=1)
    void setCopyNonSetAttributes(const int v)
    {
        copy_attributes_non_set_flag = v;
        return;
    };
};
}
#endif // __CO_SIMPLE_MODULE
