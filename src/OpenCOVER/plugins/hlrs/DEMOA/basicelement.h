/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// basicelement.h
// basicelement class definition
//
////////////////////////////////////////////////////////////////////////

#ifndef __BASICELEMENT_H
#define __BASICELEMENT_H

#include <functional>
#include <map>
#include <string>

#include "common_include.h"

class BasicElement
{
public:
    inline BasicElement();
    virtual inline ~BasicElement();

public:
    inline std::string name() const;
    inline std::string parentname() const;

protected:
    enum
    {
        _UNKNOWN,
        _NONE,
        _DEG,
        _RAD,
        _POS,
        _NEG,
        _BI,
        _EULER,
        _CARDAN,
        _DE,
        _DERF,
        _RK4,
        _ODEINT_RK,
        _ODEINT_BS,
        _INVERSE,
        _DYNAMIC
    };

protected:
    inline int check_option(const std::string &word) const;

protected:
    std::string myname;
    std::string myparentname;

private:
    typedef std::map<std::string, int, std::less<std::string> > options_t;
    options_t options;
};

inline BasicElement::BasicElement()
{
    options[""] = _NONE;
    options["deg"] = _DEG;
    options["rad"] = _RAD;
    options["+"] = _POS;
    options["-"] = _NEG;
    options["bi"] = _BI;
    options["euler"] = _EULER;
    options["cardan"] = _CARDAN;
    options["de"] = _DE;
    options["derf"] = _DERF;
    options["rk4"] = _RK4;
    options["odeint_rk"] = _ODEINT_RK;
    options["odeint_bs"] = _ODEINT_BS;
    options["inverse"] = _INVERSE;
    options["dynamic"] = _DYNAMIC;
}

inline BasicElement::~BasicElement()
{
    nooperation();
}

inline std::string BasicElement::name() const
{
    return myname;
}

inline std::string BasicElement::parentname() const
{
    return myparentname;
}

inline int BasicElement::check_option(const std::string &word) const
{
    const options_t::const_iterator cit = options.find(word);
    return (cit != options.end()) ? cit->second : _UNKNOWN;
}

#endif // __BASICELEMENT_H
