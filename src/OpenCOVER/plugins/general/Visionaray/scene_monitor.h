/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <memory>
#include <vector>

#include "common.h"

namespace scene
{
    class Observable
    {
    public:
        virtual bool changed() = 0;
        virtual void visit() = 0;
    };

    //-----------------------------------------------------

    class Material : public Observable
    {
    public:
        Material(osg::Material *om, material_list &vm, size_t idx);

        bool changed();
        void visit();
    private:
        // Monitored osg material
        osg::Material *osg_mat_;

        // List of visionaray materials
        material_list &vsnray_mats_;

        // Index to the list of visionaray materials
        size_t index_;

        // Current material attributes
        osg::Vec4 ca_;
        osg::Vec4 cd_;
        osg::Vec4 cs_;
        osg::Vec4 ce_;
        float shininess_;

        // Specular light state
        bool specular_;
    };


    //-----------------------------------------------------

    class Monitor
    {
    public:

        // Flags that indicate what scene data changed
        static const unsigned UpdateGeometry  =  1;
        static const unsigned UpdateNormals   =  2;
        static const unsigned UpdateTexCoords =  4;
        static const unsigned UpdateColors    =  8;
        static const unsigned UpdateMaterials = 16;
        static const unsigned UpdateTextures  = 32;

    public:

        void add_observable(std::shared_ptr<Observable> obs);
        void update();

        bool need_clear_frame();

        // Return a bit field that indicates what scene data changed
        unsigned update_bits();

    private:

        std::vector<std::shared_ptr<Observable> > observables_;

        bool need_clear_frame_ = true;
        unsigned update_bits_ = 0;
    };
}
