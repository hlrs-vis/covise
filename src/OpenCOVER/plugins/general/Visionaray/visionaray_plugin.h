/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_COVER_PLUGIN_H
#define VSNRAY_COVER_PLUGIN_H

#include <memory>

#include <cover/coVRPlugin.h>

namespace visionaray
{
namespace cover
{

    class Visionaray : public opencover::coVRPlugin
    {
    public:
        Visionaray();
        ~Visionaray();

        // COVER plugin interface

        bool init();
        void addNode(osg::Node *node, opencover::RenderObject *obj = NULL);
        void removeNode(osg::Node *node, bool isGroup, osg::Node *realNode);
        void preDraw(osg::RenderInfo &info);
        void expandBoundingSphere(osg::BoundingSphere &bs);
        void key(int type, int key_sym, int /* mod */);

    private:
        struct impl;
        std::unique_ptr<impl> impl_;
    };
}
} // namespace visionaray::cover

#endif // VSNRAY_COVER_PLUGIN_H
