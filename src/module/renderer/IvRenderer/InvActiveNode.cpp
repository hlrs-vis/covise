/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Implementation of class InvActiveNode                  ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:  22.10.2001                                                   ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>

#include "InvActiveNode.h"

//
// Constructor
//
InvActiveNode::InvActiveNode()
    : show_(0)
{
    actNode_ = new SoSeparator;
    actNode_->ref();

    activeSwitch_ = new SoSwitch();
    activeSwitch_->whichChild.setValue(SO_SWITCH_NONE);
    actNode_->addChild(activeSwitch_);
}

//
// show the active node
//
void
InvActiveNode::show()
{
    show_ = 1;
    activeSwitch_->whichChild.setValue(SO_SWITCH_ALL);
}

//
// hide and deactivate the handle
//
void
InvActiveNode::hide()
{
    show_ = 0;
    activeSwitch_->whichChild.setValue(SO_SWITCH_NONE);
}

// call this callback either from your application level selection CB
// or use it as selection CB
void
InvActiveNode::selectionCB(void *me, SoPath *)
{
    InvActiveNode *mee = static_cast<InvActiveNode *>(me);

    if (!mee->show_)
    {
        mee->show();
    }
}

// call this callback either from your application level deselection CB
// or use it as deselection CB
void
InvActiveNode::deSelectionCB(void *me, SoPath *)
{
    InvActiveNode *mee = static_cast<InvActiveNode *>(me);

    if (mee->show_)
    {
        mee->hide();
    }
}

//
// Destructor
//
InvActiveNode::~InvActiveNode()
{
    actNode_->removeAllChildren();
    actNode_->unref();
}
