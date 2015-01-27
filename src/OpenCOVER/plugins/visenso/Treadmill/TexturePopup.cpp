/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TexturePopup.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
//#include <cover/OpenCOVER.h>
using namespace covise;
using namespace opencover;

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Geometry>
#include <osg/Image>
#include <osgDB/ReadFile>

TexturePopup::TexturePopup(double x, double y, double width, double height)
{
    osg::Geode *geode = new osg::Geode();
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _camera = new osg::Camera;
    _camera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, 1024.0, 0.0, 768.0)); // virtual screen is 0-400/0-400
    _camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    _camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    _camera->setViewMatrix(osg::Matrix::translate(osg::Vec3(0, 0, 0)));
    _camera->setViewMatrix(osg::Matrix::identity());
    _camera->setClearMask(GL_DEPTH_BUFFER_BIT);
    _camera->setRenderOrder(osg::Camera::POST_RENDER);
    _camera->addChild(geode);

    // geometry
    osg::Geometry *geom = new osg::Geometry;
    osg::Vec3Array *vertices = new osg::Vec3Array;

    vertices->push_back(osg::Vec3(x, y, 0.0));
    vertices->push_back(osg::Vec3(x, y - height, 0.0));
    vertices->push_back(osg::Vec3(x + width, y - height, 0.0));
    vertices->push_back(osg::Vec3(x + width, y, 0.0));
    geom->setVertexArray(vertices);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec2Array *texcoords = new osg::Vec2Array(4);
    (*texcoords)[0].set(0.0f, 1.0f);
    (*texcoords)[1].set(0.0f, 0.0f);
    (*texcoords)[2].set(1.0f, 0.0f);
    (*texcoords)[3].set(1.0f, 1.0f);
    geom->setTexCoordArray(0, texcoords);

    _state = new osg::StateSet();
    geode->setStateSet(_state);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    geode->addDrawable(geom);

    _isShowing = false;
}

TexturePopup::~TexturePopup()
{
}

void TexturePopup::show()
{
    if (_isShowing)
        return;

    cover->getScene()->addChild(_camera.get());

    _isShowing = true;
}

void TexturePopup::hide()
{
    if (!_isShowing)
        return;

    cover->getScene()->removeChild(_camera.get());

    _isShowing = false;
}

void TexturePopup::setImageFile(std::string fileName)
{
    // create new texture from unknown image
    if (_popupTextures.find(fileName) == _popupTextures.end())
    {
        osg::Image *image = osgDB::readImageFile(fileName);
        osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
        texture->setImage(image);

        _popupTextures[fileName] = texture;
    }

    _state->setTextureAttributeAndModes(0, _popupTextures[fileName], osg::StateAttribute::ON);
}
