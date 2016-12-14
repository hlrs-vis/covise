/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
{
}

void Material::visit()
{
    auto ca = osg_mat_->getAmbient(osg::Material::Face::FRONT);
    auto cd = osg_mat_->getDiffuse(osg::Material::Face::FRONT);
    auto cs = osg_mat_->getSpecular(osg::Material::Face::FRONT);
    auto ce = osg_mat_->getEmission(osg::Material::Face::FRONT);
    auto sh = osg_mat_->getShininess(osg::Material::Face::FRONT);

    if (ca_ != ca || cd_ != cd || cs_ != cs || ce_ != ce || shininess_ != sh)
    {
        vsnray_mats_[index_] = osg_cast(osg_mat_);
        ca_ = ca;
        cd_ = cd;
        cs_ = cs;
        ce_ = ce;
        shininess_ = sh;
    }
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
    for (auto &o: observables_)
        o->visit();
}

}
