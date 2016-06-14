/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_COVER_DRAWABLE_H
#define VSNRAY_COVER_DRAWABLE_H 1

#include <memory>

#include <osg/BoundingSphere>
#include <osg/Drawable>

namespace visionaray { namespace cover {

struct render_state;
struct debug_state;

class drawable : public osg::Drawable
{
public:

    drawable();
   ~drawable();

    void expandBoundingSphere(osg::BoundingSphere& bs);

    void update_state(
            std::shared_ptr<render_state> const& state,
            std::shared_ptr<debug_state>  const& dev_state
            );

private:

    struct impl;
    std::unique_ptr<impl> impl_;

private:

    // osg::Drawable interface

    drawable* cloneType() const;
    osg::Object* clone(const osg::CopyOp& op) const;
    drawable(drawable const& rhs, osg::CopyOp const& op = osg::CopyOp::SHALLOW_COPY);
    void drawImplementation(osg::RenderInfo& info) const;

};

}} // namespace visionaray::cover

#endif // VSNRAY_COVER_DRAWABLE_H
