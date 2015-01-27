/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_UIF_SWITCH_H_
#define _CO_UIF_SWITCH_H_

// 15.09.99

#include "coUifElem.h"
#include "coUifSwitchCase.h"

namespace covise
{

class coChoiceParam; // we need a pointer to one...

/**
 * Switch level of parameter switching
 *
 */
class APIEXPORT coUifSwitch : public coUifElem
{

private:
    enum
    {
        MAX_CASES = 64
    };

    /// Copy-Constructor: NOT  IMPLEMENTED
    coUifSwitch(const coUifSwitch &);

    /// Assignment operator: NOT  IMPLEMENTED
    coUifSwitch &operator=(const coUifSwitch &);

    /// Default constructor: NOT  IMPLEMENTED
    coUifSwitch();

    // list of cases
    coUifSwitchCase *d_swCase[MAX_CASES];

    // check whether this case is currently active
    int d_valid[MAX_CASES];

    // number of cases
    int d_numCases;

    // name of this switch == name of the Choice parameter handling it
    char *d_name;

    // label of this switch == Written on the top Choice
    char *d_desc;

    // active Case
    int d_actCase;

    // whether this is top-level or not
    int d_toplevel;

    // a pointer to my master choice
    coChoiceParam *d_masterChoice;

public:
    /// Constructor: give a name for the switch and a descriptive label
    coUifSwitch(const char *name, const char *label, int toplevel);

    /// Destructor
    virtual ~coUifSwitch();

    // add a case to this switch
    coUifSwitchCase *addCase(const char *name);

    /// finalize switch : returns number of switch entries
    int finish();

    // ------------------ Virtuals from coUifElem

    /// Hide everything below
    virtual void hide();

    /// Show everything below
    virtual void show();

    /// return my type of element
    virtual Kind kind() const;

    /// get the name of this object
    virtual const char *getName() const;

    /// handle parameter changes: called by paramCB
    virtual int paramChange();

    /// retrieve the values from an EXEC call in compute CB
    virtual int setNonImmediateValue();

    /// give all necessary info to Covise -> automatically called by coModule in init()
    virtual void initialize();

    /// whether this can be a part-object of a switch
    virtual int switchable() const;

    /// whether this switch is on top-level
    int isTopLevel() const;

    /// performs one show() and hide() on all my sub-objects
    void startUp();

    /// get a pointer to my choice
    coChoiceParam *getMasterChoice();
};
}
#endif
