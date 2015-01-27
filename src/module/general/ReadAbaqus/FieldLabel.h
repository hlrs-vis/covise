/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS FieldLabel
//
//  A FieldLabel encapsulate all field properties relevant for the GUI:
//  its available components, invariants and locations.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _FIELD_LABEL_H_
#define _FIELD_LABEL_H_

#include "Location.h"

typedef float (odb_FieldValue::*INV_FUNC)() const;

class FieldLabel
{
public:
    FieldLabel();
    FieldLabel(const char *name,
               const char *description,
               odb_Enum::odb_DataTypeEnum);
    FieldLabel(const FieldLabel &rhs);

    void AccumulateComponent(string component);
    void AccumulateInvariant(odb_Enum::odb_InvariantEnum);
    void AccumulateFieldLocation(const odb_FieldLocation &fl,
                                 odb_FieldOutput &field);

    virtual ~FieldLabel();
    // this equal operator checks for a shallow equality
    // of _name, _description and type!!!!
    bool operator==(const FieldLabel &rhs) const;
    FieldLabel &operator=(const FieldLabel &rhs);

    void GetLists(const OutputChoice &choice,
                  vector<string> &components,
                  vector<string> &invariants,
                  vector<string> &locations,
                  vector<string> &sectionPoints) const;

    static string InvariantEnumToString(odb_Enum::odb_InvariantEnum);
    static INV_FUNC InvariantEnumToINV_FUNC(odb_Enum::odb_InvariantEnum inv);

    void ReproduceVisualisation(OutputChoice &choice,
                                const string &componentOld,
                                const string &invariantOld,
                                const string &locationOld,
                                const string &sectionPointOld) const;

    string str() const;

protected:
private:
    void AccumulateFieldLocation();
    string _name;
    string _description;
    odb_Enum::odb_DataTypeEnum _type;

    vector<string> _components;
    vector<odb_Enum::odb_InvariantEnum> _invariants;
    vector<Location> _locations;
};
#endif
