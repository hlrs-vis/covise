/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeGroup.h

#ifndef _VRMLNODEGROUP_
#define _VRMLNODEGROUP_

#include "VrmlMFNode.h"
#include "VrmlSFString.h"
#include "VrmlSFVec3f.h"

#include "VrmlNode.h"
#include "Viewer.h"
#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeGroup : public VrmlNodeChild
{

public:
    // Define the fields of all built in group nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const override;

    VrmlNodeGroup(VrmlScene *s = 0);
    virtual ~VrmlNodeGroup();
    void flushRemoveList();

    virtual VrmlNode *cloneMe() const override;
    virtual void cloneChildren(VrmlNamespace *) override;

    virtual VrmlNodeGroup *toGroup() const override;

    virtual bool isModified() const override;
    virtual void clearFlags() override;

    virtual void addToScene(VrmlScene *s, const char *relativeUrl) override;

    virtual void copyRoutes(VrmlNamespace *ns) override;

    virtual std::ostream &printFields(std::ostream &os, int indent) override;

    virtual void render(Viewer *) override;

    virtual void accumulateTransform(VrmlNode *) override;

    void activate(double timeStamp, bool isOver, bool isActive, double *p, double *M);

    void addChildren(const VrmlMFNode &children);
    void removeChildren(const VrmlMFNode &children);
    void removeChildren();

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue) override;

    virtual void setField(const char *fieldName, const VrmlField &fieldValue) override;
    virtual const VrmlField *getField(const char *fieldName) const override;

    int size();
    VrmlNode *child(int index);

    virtual VrmlNode *getParentTransform() override;

    // LarryD Feb 11/99
    VrmlMFNode *getNodes()
    {
        return &d_children;
    }

    Viewer::Object getViewerObject()
    {
        return d_viewerObject;
    };

    bool isOnlyGeometry() const override;

protected:
    void checkAndRemoveNodes(Viewer *viewer);
    VrmlSFVec3f d_bboxCenter;
    VrmlSFVec3f d_bboxSize;
    VrmlMFNode d_children;

    VrmlSFString d_relative;
    VrmlNode *d_parentTransform;
    Viewer::Object d_viewerObject;
    VrmlMFNode d_childrenToRemove;
};
}
#endif //_VRMLNODEGROUP_
