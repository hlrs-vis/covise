/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coUifSwitch.h"
#include <appl/ApplInterface.h>
#include "coChoiceParam.h"
#include "coBlankConv.h"
#include "coModule.h"

/// ----- Prevent auto-generated functions by assert -------

using namespace covise;

/// Copy-Constructor: NOT IMPLEMENTED
coUifSwitch::coUifSwitch(const coUifSwitch &)
    : coUifElem()
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
coUifSwitch &coUifSwitch::operator=(const coUifSwitch &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
coUifSwitch::coUifSwitch()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

coUifSwitch::~coUifSwitch()
{
    delete[] d_name;
    delete[] d_desc;
}

coUifSwitch::coUifSwitch(const char *name, const char *label, int toplevel)
{
    d_name = strcpy(new char[strlen(name) + 1], name);
    d_desc = strcpy(new char[strlen(label) + 1], label);
    d_numCases = 0;
    d_actCase = -1;
    d_toplevel = toplevel;
    d_masterChoice = new coChoiceParam(name, label);
}

// add a case to this switch
coUifSwitchCase *coUifSwitch::addCase(const char *name)
{
    // make sure field is large enough
    assert(d_numCases < MAX_CASES);

    // create the new case
    coUifSwitchCase *newCase = new coUifSwitchCase(name, this);
    d_swCase[d_numCases] = newCase;
    d_valid[d_numCases] = 1;
    d_numCases++;
    return newCase;
}

/// finalize switch : returns number of switch entries
int coUifSwitch::finish()
{
    // the choice copies the list
    char **choice = new char *[d_numCases + 1];
    choice[0] = coBlankConv::all(d_desc); //  Master desc.

    int i;
    for (i = 0; i < d_numCases; i++)
        choice[i + 1] = coBlankConv::all(d_swCase[i]->getName());

    d_masterChoice->setValue(d_numCases + 1, choice, 0);

    for (i = 0; i <= d_numCases; i++)
        delete[] choice[i];
    delete[] choice;

    return d_numCases;
}

/// handle parameter changes: called by paramCB
int coUifSwitch::paramChange()
{
    int pos;
    Covise::get_reply_choice(&pos);
    if (d_actCase >= 0)
        d_swCase[d_actCase]->hide();

    // set new current position - check before
    d_actCase = pos - 2;
    if (d_actCase >= d_numCases)
        d_actCase = d_numCases - 1;
    //if (d_actCase<0)
    // d_actCase=0;

    if (d_actCase >= 0)
        d_swCase[d_actCase]->show();

    // if the user requests the master choice, it must contain the new value
    d_masterChoice->paramChange();

    return 0;
}

/// Hide me and everything below
void coUifSwitch::hide()
{
    Covise::hide_param(d_name);
    if (d_actCase >= 0)
        d_swCase[d_actCase]->hide();
}

/// Show me and currently active switch
void coUifSwitch::show()
{
    if (d_masterChoice->isActive())
    {
        Covise::show_param(d_name);
        if (d_actCase >= 0)
            d_swCase[d_actCase]->show();
    }
}

/// return my type of element
coUifElem::Kind coUifSwitch::kind() const
{
    return SWITCH;
}

/// get the name of this object
const char *coUifSwitch::getName() const
{
    return d_name;
}

/// retrieve the values from an EXEC call in compute CB
int coUifSwitch::setNonImmediateValue()
{
    /// we should never get here
    Covise::sendWarning("Switch master choice '%s' non-immediate !!", d_name);

    int pos;
    Covise::get_choice_param(d_name, &pos);
    if (d_actCase >= 0)
        d_swCase[d_actCase]->hide();

    d_actCase = pos - 2;
    if (d_actCase >= 0)
        d_swCase[d_actCase]->show();

    return 0;
}

/// give all necessary info to Covise -> automatically called by coModule in init()
void coUifSwitch::initialize()
{
    // the master is NOT registered at he module, so we have to initialize it here
    d_masterChoice->initialize();
    return;
}

/// whether this can be a part-object of a switch
int coUifSwitch::switchable() const
{
    return 1;
}

/// whether this switch is on top-level
int coUifSwitch::isTopLevel() const
{
    return d_toplevel;
}

/// performs one show() and hide() on all my sub-objects
void coUifSwitch::startUp()
{
    int i;

    // make all master once visable
    d_masterChoice->show();
    d_masterChoice->hide();

    // make all sub-parts once visable -> create correct sorting
    for (i = 0; i < d_numCases; i++)
    {
        d_swCase[i]->show();
        d_swCase[i]->hide();
    }
}

/// get a pointer to my choice
coChoiceParam *coUifSwitch::getMasterChoice()
{
    return d_masterChoice;
}
