#include "TmtEntity.h"
#include <cover/coVRPluginSupport.h>

TmtEntity::TmtEntity(const std::string &id)
    : id(id)
{
}

const std::string &
TmtEntity::getId() const
{
    return id;
}

TmtEntityTransformMixin::TmtEntityTransformMixin()
    : interactor(osg::Matrix::identity(), 1000.0, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
{
    showInteractor(false);
    setOffset(osg::Matrix::identity());

    transform = new osg::MatrixTransform;
    opencover::cover->getObjectsRoot()->addChild(transform);
}

void TmtEntityTransformMixin::setOffset(osg::Matrix offset_)
{
    offset = offset_;
    offset_i.invert(offset);
}

void TmtEntityTransformMixin::setTransform(osg::Matrix transform)
{
    this->transform->setMatrix(transform);
    interactor.updateTransform(offset * transform);
}

osg::Matrix TmtEntityTransformMixin::getTransform() const
{
    return transform->getMatrix();
}

osg::ref_ptr<osg::MatrixTransform> TmtEntityTransformMixin::getTransformNode()
{
    return transform;
}

bool TmtEntityTransformMixin::checkTransformChanged()
{
    interactor.preFrame();

    if (interactor.isRunning())
    {
        osg::Matrix m = offset_i * interactor.getMatrix();
        transform->setMatrix(m);
        return true;
    }
    return false;
}

void TmtEntityTransformMixin::showInteractor(bool show)
{
    if (show)
    {
        interactor.show();
        interactor.enableIntersection();
    }
    else
    {
        interactor.hide();
        interactor.disableIntersection();
    }
}
