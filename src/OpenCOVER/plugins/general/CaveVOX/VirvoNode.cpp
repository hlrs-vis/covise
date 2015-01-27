/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Virvo:
#include <vvrenderer.h>
#include <vvvoldesc.h>

// Local:
#include "Virvo.H"
#include "VirvoNode.H"

VirvoNode::VirvoNode(Virvo::AlgorithmType alg)
{
    _drawable = new Virvo(alg);
    _geode = new Geode();
    _geode->addDrawable(_drawable.get());
    _geode->setNodeMask(~1);
    addChild(_geode.get());
}

VirvoNode::~VirvoNode()
{
}

Virvo *VirvoNode::getDrawable()
{
    return _drawable.get();
}

vvRenderer *VirvoNode::getRenderer()
{
    return _drawable->getRenderer();
}

vvVolDesc *VirvoNode::getVD()
{
    return _drawable->getVD();
}

bool VirvoNode::loadVolumeFile(const char *filename)
{
    return _drawable->loadVolumeFile(filename);
}
