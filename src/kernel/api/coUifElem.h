/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_UIF_ELEM_H_
#define _CO_UIF_ELEM_H_

// 15.09.99

/**
 * Base class for all ports
 *
 */
namespace covise
{

class APIEXPORT coUifElem
{

public:
    /// enum Kind: SWITCH, PARAM, INPORT, OUTPORT
    enum Kind
    {
        SWITCH,
        PARAM,
        INPORT,
        OUTPORT
    };

    /// Destructor : virtual because we derive objects
    virtual ~coUifElem();

    /// Hide everything below
    virtual void hide();

    /// Show everything below
    virtual void show();

    /// return my type of element
    virtual Kind kind() const = 0;

    /// get the name of this object
    virtual const char *getName() const = 0;

    /// give all necessary info to Covise -> automatically called by coModule in init()
    virtual void initialize() = 0;

    /// whether this may be a part of a switch group
    virtual int switchable() const;

    // --- all these are pre-defined empty ---

    /// handle parameter changes: called by paramCB
    virtual int paramChange();

    /// do whatever is needed before compute CB : pre-set to do nothing
    virtual int preCompute();

    /// do whatever is needed after compute CB : pre-set to do nothing
    virtual int postCompute();
};
}
#endif
