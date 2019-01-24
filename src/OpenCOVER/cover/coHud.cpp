/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coHud.h"
#include "VRViewer.h"
#include "coVRPluginSupport.h"
#include "coVRFileManager.h"
#include <iostream>
#include <config/CoviseConfig.h>
#include <osgDB/ReaderWriter>
#include <osgDB/ReadFile>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <config/coConfig.h>

using namespace osg;
using namespace covise;
using namespace opencover;

coHud *coHud::instance_ = NULL;

class UpdateCamera: public osg::Camera
{
 public:
    UpdateCamera()
    {
        setNumChildrenRequiringUpdateTraversal(getNumChildrenRequiringUpdateTraversal()+1);
    }
};

coHud::coHud()
{
    assert(!instance_);

    visible = false;
    doHide = false;

    geode = new osg::Geode();

    std::string defaultFont = coVRFileManager::instance()->getFontFile(NULL);

    // turn lighting off for the text and disable depth test to ensure its always ontop.
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int width = coCoviseConfig::getInt("width", "COVER.SplashScreen", 640);
    int height = coCoviseConfig::getInt("height", "COVER.SplashScreen", 480);
    float pr = coCoviseConfig::getFloat("r", "COVER.SplashScreen.PanelColor", 1.0);
    float pg = coCoviseConfig::getFloat("g", "COVER.SplashScreen.PanelColor", 1.0);
    float pb = coCoviseConfig::getFloat("b", "COVER.SplashScreen.PanelColor", 1.0);
    float pa = coCoviseConfig::getFloat("a", "COVER.SplashScreen.PanelColor", 0.8);
    float tr = coCoviseConfig::getFloat("r", "COVER.SplashScreen.TextColor", 1.0);
    float tg = coCoviseConfig::getFloat("g", "COVER.SplashScreen.TextColor", 1.0);
    float tb = coCoviseConfig::getFloat("b", "COVER.SplashScreen.TextColor", 1.0);
    float ta = coCoviseConfig::getFloat("a", "COVER.SplashScreen.TextColor", 0.8);

    //osg::Vec3 position(320.0f,200.0f,0.2f);
    osg::Vec3 position(width / 2., height / 2 - 40, 0.2f);
    osg::Vec3 delta(0.0f, -40.0f, 0.0f);

    line1 = new osgText::Text;
    geode->addDrawable(line1.get());

    line1->setFont(defaultFont);
    line1->setColor(osg::Vec4(tr, tg, tb, ta));
    line1->setPosition(position);
    line1->setText("starting", osgText::String::ENCODING_UTF8);
    line1->setCharacterSize(20);
    line1->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);
    position += delta;

    line2 = new osgText::Text;
    if (!coCoviseConfig::isOn("CyberClassroom", false)) // dont display too much information in CyberClassroom
    {
        geode->addDrawable(line2.get());
    }
    line2->setFont(defaultFont);
    line2->setColor(osg::Vec4(tr, tg, tb, ta));
    line2->setPosition(position);
    line2->setText("startup", osgText::String::ENCODING_UTF8);
    line2->setCharacterSize(20);
    line2->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);

    position += delta;

    line3 = new osgText::Text;
    if (!coCoviseConfig::isOn("CyberClassroom", false)) // dont display too much information in CyberClassroom
    {
        geode->addDrawable(line3.get());
    }
    line3->setFont(defaultFont);
    line3->setColor(osg::Vec4(tr, tg, tb, ta));
    line3->setPosition(position);
    line3->setText("Tracking", osgText::String::ENCODING_UTF8);
    line3->setCharacterSize(20);
    line3->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);

    position += delta;

    {
        osg::BoundingBox bb;
        for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
        {
            bb.expandBy(geode->getDrawable(i)->getBound());
        }
        //cerr << "bb min" << bb.xMin() << endl;
        //cerr << "bb max" << bb.xMax() << endl;
        osg::Geometry *geom = new osg::Geometry;

        osg::Vec3Array *vertices = new osg::Vec3Array;

        vertices->push_back(osg::Vec3(0, height, 0.001));
        vertices->push_back(osg::Vec3(0, 0, 0.001));
        vertices->push_back(osg::Vec3(width, 0, 0.001));
        vertices->push_back(osg::Vec3(width, height, 0.001));
        geom->setVertexArray(vertices);

        osg::Vec2Array *texcoords = new osg::Vec2Array;
        texcoords->push_back(osg::Vec2(0, 1));
        texcoords->push_back(osg::Vec2(0, 0));
        texcoords->push_back(osg::Vec2(1, 0));
        texcoords->push_back(osg::Vec2(1, 1));
        geom->setTexCoordArray(0, texcoords);

        osg::Vec3Array *normals = new osg::Vec3Array;
        normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
        geom->setNormalArray(normals);
        geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

        osg::Vec4Array *colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(pr, pg, pb, pa));
        geom->setColorArray(colors);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);

        geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::StateSet *stateset = geom->getOrCreateStateSet();

        string logoFile = coCoviseConfig::getEntry("value", "COVER.SplashScreen", "share/covise/icons/OpenCOVERLogo.tif");
        const char *logoName = coVRFileManager::instance()->getName(logoFile.c_str());
        osg::Image *image = NULL;
        if (logoName && (image = osgDB::readImageFile(logoName)) != NULL)
        {
            osg::Texture2D *texture = new osg::Texture2D;
            texture->setImage(image);

            /* osg::TexEnv* texenv = new osg::TexEnv;
         texenv->setMode(osg::TexEnv::BLEND);
         texenv->setColor(osg::Vec4(0.3f,0.3f,0.3f,0.3f));*/

            stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
            /*stateset->setTextureAttributeAndModes(0,texgen,osg::StateAttribute::ON);
         stateset->setTextureAttribute(0,texenv);*/
        }
        else
        {
            osg::notify(osg::NOTICE) << "unable to load 'share/covise/icons/OpenCOVERLogo.tif'" << std::endl;
        }

        stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
        //stateset->setAttribute(new osg::PolygonOffset(1.0f,1.0f),osg::StateAttribute::ON);
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setNestRenderBins(false);

        geode->addDrawable(geom);
    }

    camera = new UpdateCamera;
    camera->setName("HUD");

    int projx, projy;
    if (width > 1024)
    {
        projx = width;
        projy = height;
    }
    else
    {
        projx = 1024;
        projy = 768;
    }

    camera->setProjectionMatrix(osg::Matrix::ortho2D(0, projx, 0, projy));
    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    // set the view matrix
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setViewMatrix(osg::Matrix::translate(osg::Vec3(0, 0, 100)));
    camera->setViewMatrix(osg::Matrix::identity());

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::POST_RENDER);

    osg::MatrixTransform *m = new osg::MatrixTransform;
    m->addChild(geode.get());
    m->setMatrix(osg::Matrix::translate((projx - width) / 2.0, (projy - height) / 2.0, 0));
    camera->addChild(m);

    //cover->getScene()->addChild(camera.get());
}

coHud *coHud::instance()
{
    if (!instance_)
        instance_ = new coHud;
    return instance_;
}

void coHud::redraw()
{
    VRViewer::instance()->redrawHUD(0.0);
}
void coHud::setText1(const std::string &text)
{
    line1->setText(text, osgText::String::ENCODING_UTF8);
}
void coHud::setText2(const std::string &text)
{
    line2->setText(text, osgText::String::ENCODING_UTF8);
}
void coHud::setText3(const std::string &text)
{
    line3->setText(text, osgText::String::ENCODING_UTF8);
}

void coHud::update()
{
    if (doHide)
    {
        if (cover->frameTime() - hudTime >= logoTime)
        {
            hudTime = cover->frameTime();
            hide();
            doHide = false;
        }
    }
}

void coHud::hideLater(float time)
{
    if (visible)
    {
        doHide = true;
        hudTime = cover->frameTime();
        if (time < 0.)
            logoTime = coCoviseConfig::getFloat("COVER.LogoTime", 1.0);
        else
            logoTime = time;
    }
}

void coHud::show()
{
    if (!visible)
    {
        visible = true;
        doHide = false;
        cover->getScene()->addChild(camera.get());
    }
}

void coHud::hide()
{
    if (visible)
    {
        visible = false;
        cover->getScene()->removeChild(camera.get());
    }
}

coHud::~coHud()
{
    hide();
    instance_ = NULL;
}
