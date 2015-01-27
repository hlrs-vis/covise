/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE ReadABAQUS
//
//  ABAQUS reader
//
//  Initial version: 25.09.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _READ_ABAQUS_H_
#define _READ_ABAQUS_H_

#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;

class ChoiceState;

class ReadABAQUS : public coModule
{
public:
    ReadABAQUS(int argc, char *argv[]);
    virtual ~ReadABAQUS();

    enum ChoiceType
    {
        DO_NOT_KNOW,
        VARIABLE,
        COMPONENT,
        INVARIANT,
        LOCATION,
        SECTION_POINT
    };

protected:
private:
    enum
    {
        NUM_OUTPUTS = 1
    };
    // ports
    coOutputPort *p_mesh_[NUM_OUTPUTS];
    coOutputPort *p_data_[NUM_OUTPUTS];
    coOutputPort *p_part_indices_[NUM_OUTPUTS];
    coOutputPort *p_parts_;
    coOutputPort *p_displacements;

    // parameters
    coFileBrowserParam *p_odb_;
    coIntVectorParam *p_steps_;
    coIntScalarParam *p_frames_;
    coFloatParam *p_scale_;

    coChoiceParam *p_variable_[NUM_OUTPUTS];
    coChoiceParam *p_components_[NUM_OUTPUTS];
    coChoiceParam *p_invariants_[NUM_OUTPUTS];
    coChoiceParam *p_locations_[NUM_OUTPUTS];
    coChoiceParam *p_sectionPoints_[NUM_OUTPUTS];
    coBooleanParam *p_onlyThisSectionPoint_[NUM_OUTPUTS];
    coBooleanParam *p_LocalCS_[NUM_OUTPUTS];

    coBooleanParam *p_doUpdate_;
    coFileBrowserParam *p_updatedOdb_;
    coBooleanParam *p_ConjugateData_;

    string manualUpdateName_;

    void updateOdb();
    virtual int compute(const char *port);
    virtual void postInst();
    bool _callRebuild;
    virtual void param(const char *paramName, bool inMapLoading);

    ChoiceType ParamChanged(const char *paramName, int &dataset);

    ChoiceState *p_choices_;
};
#endif
