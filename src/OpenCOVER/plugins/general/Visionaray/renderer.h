/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_RENDERER_H
#define VSNRAY_PLUGIN_RENDERER_H 1

#include <memory>

#include <osg/BoundingSphere>
#include <osg/Drawable>

#include <PluginUtil/MultiChannelDrawer.h>

#include <visionaray/pixel_traits.h>
#include <visionaray/render_target.h>

#define VSNRAY_COLOR_PIXEL_FORMAT	visionaray::PF_RGBA32F
#define VSNRAY_DEPTH_PIXEL_FORMAT	visionaray::PF_DEPTH32F

namespace osg
{
    class Sequence;
}

namespace visionaray
{

    struct render_state;
    struct debug_state;

    class renderer
    {
    public:
        using color_type = typename pixel_traits<VSNRAY_COLOR_PIXEL_FORMAT>::type;
        using depth_type = typename pixel_traits<VSNRAY_DEPTH_PIXEL_FORMAT>::type;

        using ref_type = render_target_ref<VSNRAY_COLOR_PIXEL_FORMAT, VSNRAY_DEPTH_PIXEL_FORMAT>;

    public:
        renderer();
        ~renderer();

        color_type* color();
        depth_type* depth();

        color_type const* color() const;
        depth_type const* depth() const;

        int width() const;
        int height() const;

        ref_type ref();

        void expandBoundingSphere(osg::BoundingSphere &bs);

        void update_state(
            std::shared_ptr<render_state> const &state,
            std::shared_ptr<debug_state> const &dev_state);

        // Acquire scene data, additionally store the provided
        // animation sequences in dedicated BVHs
        void acquire_scene_data(const std::vector<osg::Sequence *> &seqs);

        // Suppress rendering with Visionaray and resort to OpenGL,
        // but keep the Visionaray data structures intact
        void set_suppress_rendering(bool enable);

        void render_frame(osg::RenderInfo &info);

        void begin_frame();
        void end_frame();

        void init();

    private:
        struct impl;
        std::unique_ptr<impl> impl_;

        int cur_channel_;
    };

} // namespace visionaray

#endif // VSNRAY_PLUGIN_RENDERER_H
