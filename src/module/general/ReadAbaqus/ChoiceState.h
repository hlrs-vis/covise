/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ChoiceState
//
//  This class interfaces between the module class and FieldContainer.
//  An OutputChoice is used in order to select info from a FieldContainer.
//  This info is handed over to the module class in order to correctly update
//  the GUI.
//  On output it attaches output objects to the pertinent ports.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _CHOICE_STATE_H_
#define _CHOICE_STATE_H_

#include <util/coviseCompat.h>
#include "FieldContainer.h"
#include "odb_Step.h"
#include <api/coChoiceParam.h>
#include "InstanceMesh.h"

namespace covise
{
class coOutputPort;
}

using namespace covise;

class ChoiceState
{
public:
    ChoiceState(int no_sets, const char *path);
    virtual ~ChoiceState();
    bool OK() const;
    bool OK(string &mssgOut) const;
    bool updateRequired() const;
    string Path() const;
    int no_steps() const;
    bool ReproduceVisualisation(const ChoiceState *old_state);
    void RebuildVisualisation(int dataset,
                              const coChoiceParam *p_fields,
                              const coChoiceParam *p_components,
                              const coChoiceParam *p_invariants,
                              const coChoiceParam *p_locations,
                              const coChoiceParam *p_sectionPoints);
    void SetVariable(int dataset,
                     coChoiceParam *p_fields,
                     coChoiceParam *p_components,
                     coChoiceParam *p_invariants,
                     coChoiceParam *p_locations,
                     coChoiceParam *p_sectionPoints);
    void SetComponentOrInvariant(int dataset,
                                 coChoiceParam *p_components,
                                 coChoiceParam *p_invariants);
    void SetLocation(int dataset,
                     coChoiceParam *p_locations,
                     coChoiceParam *p_sectionPoints);
    void SetSectionPoint(int dataset,
                         coChoiceParam *p_sectionPoints);
    void AdjustParams(int dataset, coChoiceParam *p_variable_,
                      coChoiceParam *p_components_,
                      coChoiceParam *p_invariants_,
                      coChoiceParam *p_locations_,
                      coChoiceParam *p_sectionPoints_) const;
    // compute
    void PartDictionary(coOutputPort *p_partdict) const;
    void DataSet(int dataset,
                 int minStep, int maxStep, int jumpSteps,
                 int jumpFrames,
                 bool onlyThisSection,
                 bool localCS,
                 coOutputPort *p_mesh,
                 coOutputPort *p_data,
                 coOutputPort *p_part_indices,
                 coOutputPort *p_displacements,
                 bool conjugate);
    bool harmonic() const;
    void doUpdate(const string &updPath);

protected:
private:
    void ReadDisplacements(odb_FieldOutput &disp_field,
                           vector<InstanceMesh> &local_instance_meshes);

    void ReadStatus(odb_FieldOutput &status_field,
                    vector<InstanceMesh> &local_instance_meshes);

    void ReadField(odb_FieldOutput &field,
                   bool onlyThisSection,
                   bool localCS,
                   const string &component,
                   const string &invariant,
                   const string &location,
                   const string &sectionPoint,
                   vector<InstanceMesh> &local_instance_meshes,
                   bool conjugate);
    bool ShowInstance(int instance_number) const;
    int CheckInstanceLocationSectionPoint(const odb_FieldValue &f,
                                          const string &location,
                                          const string &sectionPoint,
                                          bool onlyThisSection) const;
    bool CheckLocation(odb_FieldOutput &field,
                       bool onlyThisSection,
                       const string &location) const;

    map<string, int> _instance_dictionary;
    vector<InstanceMesh> _instance_meshes;

    bool _harmonic;

    void AccumulateFieldsInStep(odb_Step &step, int step_label);
    void AccumulateFieldsInFrame(odb_Frame &frame);

    OutputChoice *_choice_sets;

    FieldContainer _fieldContainer;
    int _no_sets;
    string _path;
    string _mssgOut;
    bool _OK;

    bool odbUpdateRequired_;
    int _no_steps;
};
#endif
