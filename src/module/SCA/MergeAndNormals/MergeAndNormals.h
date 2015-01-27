/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE MergeAndNormals
//
//  merges nodes along lines parallel to the coord. directions
//  and generates trivial normals for the nodes with null Z coordinate
//
//  Initial version:   21.10.97 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _MERGE_AND_NORMALS_H_
#define _MERGE_AND_NORMALS_H_
#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoData.h>

class MergeAndNormals : public coSimpleModule
{
public:
    MergeAndNormals(int argc, char *argv[]);
    ~MergeAndNormals();

private:
    // these variables are set in preHandleObjects
    bool preOK_;
    bool merge_;
    float grundZellenHoehe_;
    bool readGrundZellenHoehe_;
    float grundZellenBreite_;
    bool readGrundZellenBreite_;
    int mergeNodes_;
    bool readMergeNodes_;

    // ports
    coInputPort *p_inGeom_;
    coInputPort *p_inNormals_;
    coInputPort *p_text_;
    coOutputPort *p_outGeom_;
    coOutputPort *p_Normals_;
    int ProjectNormals(coDoPolygons *, coDoVec3 *);
    virtual int compute(const char *port);
    virtual void preHandleObjects(coInputPort **in_ports);
};
#endif
