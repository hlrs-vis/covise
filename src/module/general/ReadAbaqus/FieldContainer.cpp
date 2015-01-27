/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FieldContainer.h"
#include <algorithm>
using std::find;

FieldContainer::FieldContainer()
{
    // one possible entry is no data ->
    // in case we only want to see the geometry
    _fieldLabels.push_back(FieldLabel());
}

FieldContainer::~FieldContainer()
{
}

void
FieldContainer::AccumulateField(odb_FieldOutput &field)
{
    cerr << field.name().CStr() << endl;
    FieldLabel testLabel(field.name().CStr(),
                         field.description().CStr(),
                         field.type());
    vector<FieldLabel>::iterator it_field_label = find(_fieldLabels.begin(), _fieldLabels.end(), testLabel);

    if (it_field_label == _fieldLabels.end())
    {
        _fieldLabels.push_back(testLabel);
        it_field_label = _fieldLabels.end();
        --it_field_label;
    }

    FieldLabel &thisField = *it_field_label;

    // components
    int numComp = field.componentLabels().size();
    thisField.AccumulateComponent("None");
    int comp;
    for (comp = 0; comp < numComp; comp++)
    {
        thisField.AccumulateComponent(field.componentLabels().constGet(comp).CStr());
    }

    // invariants
    int numInvar = field.validInvariants().size();
    thisField.AccumulateInvariant(odb_Enum::UNDEFINED_INVARIANT);
    int invar;
    for (invar = 0; invar < numInvar; invar++)
    {
        thisField.AccumulateInvariant(field.validInvariants().constGet(invar));
    }

    // locations
    odb_SequenceFieldLocation flCon = field.locations();
    int numFieldloc = flCon.size();
    int fieldLoc;
    for (fieldLoc = 0; fieldLoc < numFieldloc; ++fieldLoc)
    {
        thisField.AccumulateFieldLocation(flCon.constGet(fieldLoc), field);
    }
}

void
FieldContainer::GetLists(const OutputChoice &choice,
                         vector<string> &fieldLabels,
                         vector<string> &components,
                         vector<string> &invariants,
                         vector<string> &locations,
                         vector<string> &sectionPoints) const
{
    int field;
    for (field = 0; field < _fieldLabels.size(); ++field)
    {
        fieldLabels.push_back(_fieldLabels[field].str());
    }
    if (choice._fieldLabel >= 0 && choice._fieldLabel < _fieldLabels.size())
    {
        _fieldLabels[choice._fieldLabel].GetLists(choice, components, invariants,
                                                  locations, sectionPoints);
    }
}

void
FieldContainer::ReproduceVisualisation(OutputChoice &choice,
                                       const string &fieldLabelOld,
                                       const string &componentOld,
                                       const string &invariantOld,
                                       const string &locationOld,
                                       const string &sectionPointOld) const
{
    choice._fieldLabel = 0;
    choice._component = 0;
    choice._invariant = 0;
    choice._location = 0;
    choice._secPoint = 0;

    int field;
    for (field = 0; field < _fieldLabels.size(); ++field)
    {
        if (fieldLabelOld == _fieldLabels[field].str())
        {
            break;
        }
    }
    if (field == _fieldLabels.size())
    {
        return;
    }
    choice._fieldLabel = field;
    _fieldLabels[field].ReproduceVisualisation(choice,
                                               componentOld, invariantOld,
                                               locationOld, sectionPointOld);
}
