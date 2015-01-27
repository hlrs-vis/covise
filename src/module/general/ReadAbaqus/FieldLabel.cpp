/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "FieldLabel.h"
#include "odb_Enum.h"
#include <algorithm>
using std::find;

FieldLabel::FieldLabel()
    : _name("None")
    , _description()
    , _type(odb_Enum::UNDEFINED_DATATYPE)
{
    AccumulateComponent("None");
    AccumulateInvariant(odb_Enum::UNDEFINED_INVARIANT);
    AccumulateFieldLocation(); // accumulate a dummy
}

FieldLabel::FieldLabel(const char *name,
                       const char *description,
                       odb_Enum::odb_DataTypeEnum type)
    : _name(name)
    , _description(description)
    , _type(type)
{
}

FieldLabel::FieldLabel(const FieldLabel &rhs)
    : _name(rhs._name)
    , _description(rhs._description)
    , _type(rhs._type)
    , _components(rhs._components)
    , _invariants(rhs._invariants)
    , _locations(rhs._locations)
{
}

FieldLabel::~FieldLabel()
{
}

bool
    FieldLabel::
    operator==(const FieldLabel &rhs) const
{
    return (_name == rhs._name
            && _description == rhs._description
            && _type == rhs._type);
}

FieldLabel &
    FieldLabel::
    operator=(const FieldLabel &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    _name = rhs._name;
    _description = rhs._description;
    _type = rhs._type;
    _components = rhs._components;
    _invariants = rhs._invariants;
    _locations = rhs._locations;
    return *this;
}

void
FieldLabel::AccumulateComponent(string component)
{
    vector<string>::iterator it_comp;
    it_comp = find(_components.begin(), _components.end(), component);
    if (it_comp == _components.end())
    {
        _components.push_back(component);
    }
}

void
FieldLabel::AccumulateInvariant(odb_Enum::odb_InvariantEnum inv)
{
    vector<odb_Enum::odb_InvariantEnum>::iterator it_inv;
    it_inv = find(_invariants.begin(), _invariants.end(), inv);
    if (it_inv == _invariants.end())
    {
        _invariants.push_back(inv);
    }
}

void
FieldLabel::AccumulateFieldLocation()
{
    _locations.push_back(odb_Enum::UNDEFINED_POSITION);
    Location &thisLocation = *_locations.rbegin();
    thisLocation.AccumulateSectionPoint();
}

void
FieldLabel::AccumulateFieldLocation(const odb_FieldLocation &floc,
                                    odb_FieldOutput &field)
{
    // we have to test whether this odb_FieldLocation
    // has already been registered
    Location testLocation(floc.position());
    vector<Location>::iterator it_Loc = find(_locations.begin(), _locations.end(), testLocation);

    // and add if pertinent
    if (it_Loc == _locations.end())
    {
        _locations.push_back(testLocation);
        it_Loc = _locations.end();
        --it_Loc;
    }

    Location &thisLocation = *it_Loc;

    // now we work on a Location object (thisLocation) and test whether
    // it has all sectionPoints in floc (add them otherwise)
    int numSP = floc.sectionPoint().size();
    int i;
    for (i = 0; i < numSP; i++)
    {
        thisLocation.AccumulateSectionPoint(floc.sectionPoint(i));
    }

    if (numSP == 0)
    {
        // no section points defined for this position,
        // this is the normal situation for continuous elements
        thisLocation.AccumulateSectionPoint(); // this adds a dummy entry
    }
    else if (thisLocation.NoDummy())
    {
        // in case that the list of sectionPoints does not include a dummy,
        // check whether all FieldValues in field can be assigned to
        // one of the available sectionPoints, if not, add a dummy.
        // This may be slow...
        odb_SequenceFieldValue fvCon = field.values();
        int numVal = fvCon.size();
        int i;
        bool foundWithoutSP = false;
        for (i = 0; i < numVal; i++)
        {
            const odb_FieldValue &f = fvCon.constGet(i);
            if (f.sectionPoint().number() <= 0) // found a field value without sP
            {
                foundWithoutSP = true;
                break;
            }
        }
        if (foundWithoutSP)
        {
            thisLocation.AccumulateSectionPoint(); // this adds a dummy entry
        }
    }
}

void
FieldLabel::GetLists(const OutputChoice &choice,
                     vector<string> &components,
                     vector<string> &invariants,
                     vector<string> &locations,
                     vector<string> &sectionPoints) const
{
    // components
    components = _components;

    // invariants
    int inv;
    for (inv = 0; inv < _invariants.size(); ++inv)
    {
        invariants.push_back(InvariantEnumToString(_invariants[inv]));
    }

    // locations
    int loc;
    for (loc = 0; loc < _locations.size(); ++loc)
    {
        locations.push_back(_locations[loc].str());
    }

    // section points
    if (choice._location >= 0 && choice._location < _locations.size())
    {
        _locations[choice._location].GetLists(sectionPoints);
    }
}

string
FieldLabel::InvariantEnumToString(odb_Enum::odb_InvariantEnum inv)
{
    switch (inv)
    {
    case odb_Enum::MAGNITUDE:
        return "MAGNITUDE";
    case odb_Enum::MISES:
        return "MISES";
    case odb_Enum::TRESCA:
        return "TRESCA";
    case odb_Enum::PRESS:
        return "PRESS";
    case odb_Enum::INV3:
        return "INV3";
    case odb_Enum::MAX_PRINCIPAL:
        return "MAX_PRINCIPAL";
    case odb_Enum::MID_PRINCIPAL:
        return "MID_PRINCIPAL";
    case odb_Enum::MIN_PRINCIPAL:
        return "MIN_PRINCIPAL";
    case odb_Enum::MAX_INPLANE_PRINCIPAL:
        return "MAX_INPLANE_PRINCIPAL";
    case odb_Enum::MIN_INPLANE_PRINCIPAL:
        return "MIN_INPLANE_PRINCIPAL";
    case odb_Enum::OUTOFPLANE_PRINCIPAL:
        return "OUTOFPLANE_PRINCIPAL";
    default:
        return "UNDEFINED_INVARIANT";
    }
    return "UNDEFINED_INVARIANT";
}

INV_FUNC
FieldLabel::InvariantEnumToINV_FUNC(odb_Enum::odb_InvariantEnum inv)
{
    switch (inv)
    {
    case odb_Enum::MAGNITUDE:
        return &odb_FieldValue::magnitude;
    case odb_Enum::MISES:
        return &odb_FieldValue::mises;
    case odb_Enum::TRESCA:
        return &odb_FieldValue::tresca;
    case odb_Enum::PRESS:
        return &odb_FieldValue::press;
    case odb_Enum::INV3:
        return &odb_FieldValue::inv3;
    case odb_Enum::MAX_PRINCIPAL:
        return &odb_FieldValue::maxPrincipal;
    case odb_Enum::MID_PRINCIPAL:
        return &odb_FieldValue::midPrincipal;
    case odb_Enum::MIN_PRINCIPAL:
        return &odb_FieldValue::minPrincipal;
    case odb_Enum::MAX_INPLANE_PRINCIPAL:
        return &odb_FieldValue::maxInPlanePrincipal;
    case odb_Enum::MIN_INPLANE_PRINCIPAL:
        return &odb_FieldValue::minInPlanePrincipal;
    case odb_Enum::OUTOFPLANE_PRINCIPAL:
        return &odb_FieldValue::outOfPlanePrincipal;
    }
    return NULL;
}

string
FieldLabel::str() const
{
    string ret(_name);
    ret += ": ";
    ret += _description;
    return ret;
}

void
FieldLabel::ReproduceVisualisation(OutputChoice &choice,
                                   const string &componentOld,
                                   const string &invariantOld,
                                   const string &locationOld,
                                   const string &sectionPointOld) const
{
    choice._component = 0;
    choice._invariant = 0;
    choice._location = 0;
    choice._secPoint = 0;
    int comp;
    for (comp = 0; comp < _components.size(); ++comp)
    {
        if (componentOld == _components[comp])
        {
            break;
        }
    }

    if (comp == _components.size() && componentOld != "None")
    {
        return;
    }
    else
    {
        choice._component = comp;
    }

    int inv;
    for (inv = 0; inv < _invariants.size(); ++inv)
    {
        if (invariantOld == InvariantEnumToString(_invariants[inv]))
        {
            break;
        }
    }

    if (inv == _invariants.size() && invariantOld != "None")
    {
        return;
    }
    else
    {
        choice._invariant = inv;
    }

    int loc;
    for (loc = 0; loc < _locations.size(); ++loc)
    {
        if (locationOld == _locations[loc].str())
        {
            break;
        }
    }
    if (loc == _locations.size())
    {
        return;
    }
    _locations[loc].ReproduceVisualisation(choice, sectionPointOld);
}
