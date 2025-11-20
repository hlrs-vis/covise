/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/core/ref_ptr.h>
#include <vsg/state/Image.h>


namespace vrui
{

class VSGVruiRendererInterface: public vruiRendererInterface
{
  public:
    using vruiRendererInterface::vruiRendererInterface;

    static VSGVruiRendererInterface *the()
    {
        return static_cast<VSGVruiRendererInterface *>(vrui::vruiRendererInterface::the());
    }

    virtual void addToTransfer(vsg::BufferInfo *bi) = 0;
    virtual vsg::ref_ptr<vsg::Image> createVsgTexture(const std::string &textureName) = 0;
};

}
