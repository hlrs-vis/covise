/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++                                                                     ++
// ++ Description: Base class for groups of nodes in the scene graph of   ++
// ++              the COVISE viewer which will be handled interactively. ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++          VirCinity GmbH                                             ++
// ++          Nobelstrasse 15                                            ++
// ++          70569 Stuttgart                                            ++
// ++                                                                     ++
// ++ Date: 22.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef INVACTIVENODE_H
#define INVACTIVENODE_H

#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>

#include <Inventor/nodes/SoAnnotation.h>
#include <Inventor/SoPickedPoint.h>

class InvActiveNode
{
public:
    InvActiveNode();

    virtual SoSeparator *getSeparator() const
    {
        return actNode_;
    };

    // destroy an InvActiveNode
    // does proper removal of all INVENTOR nodes by calling removeAll() unref()
    // Therefore: ALL OBJECTS DERIVED FROM InvActiveNode MUST NOT do this themself!!
    virtual ~InvActiveNode();

    // show and activate the handle
    virtual void show();

    // hide and deactivate the handle
    virtual void hide();

    // call this callback either from application level selection CB
    // or use it as selection CB
    static void selectionCB(void *me, SoPath *sp);

    // call this callback either from application level deselection CB
    // or use it as deselection CB
    static void deSelectionCB(void *me, SoPath *);

    virtual int isShown()
    {
        return show_;
    };

protected:
    SoSwitch *getSwitch()
    {
        return activeSwitch_;
    };

private:
    int show_;

    SoSeparator *actNode_;
    SoSwitch *activeSwitch_;
};
#endif
