/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coTimerParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coTimerParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coTimerParam::s_type = "TIMERP";
coUifPara::Typeinfo coTimerParam::s_paraType = coUifPara::numericType("TIMERP");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coTimerParam::coTimerParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_start = 0;
    d_delta = 1;
    d_state = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coTimerParam::~coTimerParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coTimerParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coTimerParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coTimerParam::paramChange()
{
    return Covise::get_reply_timer(&d_start, &d_delta, &d_state);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coTimerParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[128];
    sprintf(d_defString, "%ld %ld %ld", d_start, d_delta, d_state);

    Covise::add_port(PARIN, d_name, "Timer", d_desc);
    Covise::set_port_default(d_name, d_defString);
    //Covise::set_port_immediate(d_name,d_immediate); // cannot be immediate
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coTimerParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "Timer: Start = " << d_start
         << " Delta = " << d_delta
         << " State = " << d_state
         << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coTimerParam::setValue(long min, long max, long value)
{
    d_start = min;
    d_delta = max;
    d_state = value;
    if (d_start > d_state)
        d_start = d_state;
    if (d_delta < d_state)
        d_delta = d_state;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_slider_param(d_name, d_start, d_delta, d_state);
    else
        return 1;
}

int coTimerParam::setStart(long start)
{
    d_start = start;
    if (d_start > d_state)
        d_state = d_start;
    if (d_delta > d_start)
        d_delta = d_start;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_slider_param(d_name, d_start, d_delta, d_state);
    else
        return 1;
}

int coTimerParam::setDelta(long delta)
{
    d_delta = delta;
    if (d_state > d_delta)
        d_state = d_delta;
    if (d_start > d_delta)
        d_start = d_delta;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_timer_param(d_name, d_start, d_delta, d_state);
    else
        return 1;
}

int coTimerParam::setState(long state)
{
    d_state = state;
    if (d_start > d_state)
        d_start = d_state;
    if (d_delta < d_state)
        d_delta = d_state;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_timer_param(d_name, d_start, d_delta, d_state);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

void coTimerParam::getValue(long &start, long &delta, long &state) const
{
    start = d_start;
    delta = d_delta;
    state = d_state;
}

long coTimerParam::getStart() const { return d_start; }
long coTimerParam::getDelta() const { return d_delta; }
long coTimerParam::getState() const { return d_state; }

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// prohibit some functions

void coTimerParam::setImmediate(int) { para_error("setImmediate not supported"); }
void coTimerParam::setActive(int) { para_error("setActive not supported"); }
void coTimerParam::enable() { para_error("enable not supported"); }
void coTimerParam::disable() { para_error("disable not supported"); }
void coTimerParam::hide() { para_error("hide not supported"); }
void coTimerParam::show() { para_error("show not supported"); }

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coTimerParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coTimerParam::getValString() const
{
    static char valString[192];
    sprintf(valString, "%ld %ld %ld", d_start, d_delta, d_state);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coTimerParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%ld %ld %ld", &d_start, &d_delta, &d_state);
    if (retval != 3)
    {
        std::cerr << "coTimerParam::setValString: sscanf failed" << std::endl;
        return;
    }
}
