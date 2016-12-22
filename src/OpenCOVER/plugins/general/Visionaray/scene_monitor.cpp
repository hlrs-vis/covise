/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <GL/glew.h>

#include <cover/coVRLighting.h>

#include "scene_monitor.h"

namespace scene
{

//-------------------------------------------------------------------------------------------------
// Material observable
//

Material::Material(osg::Material *om, material_list &vm, size_t idx)
    : osg_mat_(om)
    , vsnray_mats_(vm)
    , index_(idx)
    , ca_(om->getAmbient(osg::Material::Face::FRONT))
    , cd_(om->getDiffuse(osg::Material::Face::FRONT))
    , cs_(om->getSpecular(osg::Material::Face::FRONT))
    , ce_(om->getEmission(osg::Material::Face::FRONT))
    , shininess_(om->getShininess(osg::Material::Face::FRONT))
    , specular_(opencover::coVRLighting::instance()->specularlightState)
{
}

bool Material::changed()
{
    auto ca = osg_mat_->getAmbient(osg::Material::Face::FRONT);
    auto cd = osg_mat_->getDiffuse(osg::Material::Face::FRONT);
    auto cs = osg_mat_->getSpecular(osg::Material::Face::FRONT);
    auto ce = osg_mat_->getEmission(osg::Material::Face::FRONT);
    auto sh = osg_mat_->getShininess(osg::Material::Face::FRONT);
    bool sp = opencover::coVRLighting::instance()->specularlightState;

    return ca_ != ca || cd_ != cd || cs_ != cs || ce_ != ce || shininess_ != sh || specular_ != sp;
}

void Material::visit()
{
    auto ca = osg_mat_->getAmbient(osg::Material::Face::FRONT);
    auto cd = osg_mat_->getDiffuse(osg::Material::Face::FRONT);
    auto cs = osg_mat_->getSpecular(osg::Material::Face::FRONT);
    auto ce = osg_mat_->getEmission(osg::Material::Face::FRONT);
    auto sh = osg_mat_->getShininess(osg::Material::Face::FRONT);
    bool sp = opencover::coVRLighting::instance()->specularlightState;

    vsnray_mats_[index_] = osg_cast(osg_mat_, sp);
    ca_ = ca;
    cd_ = cd;
    cs_ = cs;
    ce_ = ce;
    shininess_ = sh;
    specular_ = sp;
}


//-------------------------------------------------------------------------------------------------
// Monitor
//

void Monitor::add_observable(std::shared_ptr<Observable> obs)
{
    observables_.push_back(obs);
}

void Monitor::update()
{
    need_clear_frame_ = false;
    update_bits_ = 0;

    for (auto &o: observables_)
    {
        if (o->changed())
        {
            // TODO: merge into bits!?
            need_clear_frame_ = true;

            // TODO: nicer implementation
            if (std::dynamic_pointer_cast<Material>(o) != nullptr)
                update_bits_ |= UpdateMaterials;

            o->visit();
        }
    }
}

bool Monitor::need_clear_frame()
{
    return need_clear_frame_;
}

unsigned Monitor::update_bits()
{
    return update_bits_;
}

}
