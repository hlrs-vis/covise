/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS FieldContainer
//
//  The name FieldContainer is not very felicitous. It would be
//  better something like FieldLabelContainer. It encapsulates
//  all available choices available from a database file.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef _FIELD_CONTAINER_H_
#define _FIELD_CONTAINER_H_

#include "FieldLabel.h"
#include <util/coviseCompat.h>

class FieldContainer
{
public:
    FieldContainer();
    virtual ~FieldContainer();
    void AccumulateField(odb_FieldOutput &field);
    void GetLists(const OutputChoice &choice,
                  vector<string> &fieldLabels,
                  vector<string> &components,
                  vector<string> &invariants,
                  vector<string> &locations,
                  vector<string> &sectionPoints) const;
    void ReproduceVisualisation(OutputChoice &choice,
                                const string &fieldLabelOld,
                                const string &componentOld,
                                const string &invariantOld,
                                const string &locationOld,
                                const string &sectionPointOld) const;

protected:
private:
    vector<FieldLabel> _fieldLabels;
};
#endif
