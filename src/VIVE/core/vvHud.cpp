/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "vvHud.h"
#include "vvViewer.h"
#include "vvPluginSupport.h"
#include "vvFileManager.h"
#include <iostream>
#include <config/CoviseConfig.h>
#include <vsg/nodes/MatrixTransform.h>
#include <config/coConfig.h>

using namespace vsg;
using namespace covise;
using namespace vive;

vvHud *vvHud::instance_ = NULL;

class UpdateCamera: public vsg::Camera
{
 public:
    UpdateCamera()
    {
        //setNumChildrenRequiringUpdateTraversal(children.sizeRequiringUpdateTraversal()+1);
    }
};

vvHud::vvHud()
{
    assert(!instance_);

    visible = false;
    doHide = false;
    /*
    geode = new osg::Geode();

    std::string defaultFont = vvFileManager::instance()->getFontFile(NULL);

    // turn lighting off for the text and disable depth test to ensure its always ontop.
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int width = coCoviseConfig::getInt("width", "VIVE.SplashScreen", 640);
    int height = coCoviseConfig::getInt("height", "VIVE.SplashScreen", 480);
    float pr = coCoviseConfig::getFloat("r", "VIVE.SplashScreen.PanelColor", 1.0);
    float pg = coCoviseConfig::getFloat("g", "VIVE.SplashScreen.PanelColor", 1.0);
    float pb = coCoviseConfig::getFloat("b", "VIVE.SplashScreen.PanelColor", 1.0);
    float pa = coCoviseConfig::getFloat("a", "VIVE.SplashScreen.PanelColor", 0.8);
    float tr = coCoviseConfig::getFloat("r", "VIVE.SplashScreen.TextColor", 1.0);
    float tg = coCoviseConfig::getFloat("g", "VIVE.SplashScreen.TextColor", 1.0);
    float tb = coCoviseConfig::getFloat("b", "VIVE.SplashScreen.TextColor", 1.0);
    float ta = coCoviseConfig::getFloat("a", "VIVE.SplashScreen.TextColor", 0.8);

    //vsg::vec3 position(320.0f,200.0f,0.2f);
    vsg::vec3 position(width / 2., height / 2 - 40, 0.2f);
    vsg::vec3 delta(0.0f, -40.0f, 0.0f);

    line1 = new osgText::Text;
    geode->addDrawable(line1);

    line1->setFont(defaultFont);
    line1->setColor(vsg::vec4(tr, tg, tb, ta));
    line1->setPosition(position);
    line1->setText("starting", osgText::String::ENCODING_UTF8);
    line1->setCharacterSize(20);
    line1->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);
    position += delta;

    line2 = new osgText::Text;
    if (!coCoviseConfig::isOn("CyberClassroom", false)) // dont display too much information in CyberClassroom
    {
        geode->addDrawable(line2);
    }
    line2->setFont(defaultFont);
    line2->setColor(vsg::vec4(tr, tg, tb, ta));
    line2->setPosition(position);
    line2->setText("startup", osgText::String::ENCODING_UTF8);
    line2->setCharacterSize(20);
    line2->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);

    position += delta;

    line3 = new osgText::Text;
    if (!coCoviseConfig::isOn("CyberClassroom", false)) // dont display too much information in CyberClassroom
    {
        geode->addDrawable(line3);
    }
    line3->setFont(defaultFont);
    line3->setColor(vsg::vec4(tr, tg, tb, ta));
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
        vsg::Node *geom = new vsg::Node;

        vsg::vec3Array *vertices = new vsg::vec3Array;

        vertices->push_back(vsg::vec3(0, height, 0.001));
        vertices->push_back(vsg::vec3(0, 0, 0.001));
        vertices->push_back(vsg::vec3(width, 0, 0.001));
        vertices->push_back(vsg::vec3(width, height, 0.001));
        geom->setVertexArray(vertices);

        osg::Vec2Array *texcoords = new osg::Vec2Array;
        texcoords->push_back(osg::Vec2(0, 1));
        texcoords->push_back(osg::Vec2(0, 0));
        texcoords->push_back(osg::Vec2(1, 0));
        texcoords->push_back(osg::Vec2(1, 1));
        geom->setTexCoordArray(0, texcoords);

        vsg::vec3Array *normals = new vsg::vec3Array;
        normals->push_back(vsg::vec3(0.0f, 0.0f, 1.0f));
        geom->setNormalArray(normals);
        geom->setNormalBinding(vsg::Node::BIND_OVERALL);

        vsg::vec4Array *colors = new vsg::vec4Array;
        colors->push_back(vsg::vec4(pr, pg, pb, pa));
        geom->setColorArray(colors);
        geom->setColorBinding(vsg::Node::BIND_OVERALL);

        geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::StateSet *stateset = geom->getOrCreateStateSet();

        string logoFile = coCoviseConfig::getEntry("value", "VIVE.SplashScreen", "share/covise/icons/OpenCOVERLogo.tif");
        const char *logoName = vvFileManager::instance()->getName(logoFile.c_str());
        osg::Image *image = NULL;
        if (logoName && (image = osgDB::readImageFile(logoName)) != NULL)
        {
            osg::Texture2D *texture = new osg::Texture2D;
            texture->setImage(image);


            stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
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

    camera->setProjectionMatrix(vsg::dmat4::ortho2D(0, projx, 0, projy));
    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    // set the view matrix
    camera->setReferenceFrame(vsg::MatrixTransform::ABSOLUTE_RF);
    camera->setViewMatrix(vsg::translate(vsg::vec3(0, 0, 100)));
    camera->setViewMatrix(vsg::dmat4::identity());

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::POST_RENDER);

    vsg::MatrixTransform *m = vsg::MatrixTransform::create();
    m->addChild(geode);
    m->matrix = (vsg::translate((projx - width) / 2.0, (projy - height) / 2.0, 0));
    camera->addChild(m);
    */
    //vv->getScene()->addChild(camera);
}

vvHud *vvHud::instance()
{
    if (!instance_)
        instance_ = new vvHud;
    return instance_;
}

void vvHud::redraw()
{
    vvViewer::instance()->redrawHUD(0.0);
}
void vvHud::setText1(const std::string &text)
{
    //line1->setText(text, osgText::String::ENCODING_UTF8);
}
void vvHud::setText2(const std::string &text)
{
    //line2->setText(text, osgText::String::ENCODING_UTF8);
}
void vvHud::setText3(const std::string &text)
{
    //line3->setText(text, osgText::String::ENCODING_UTF8);
}

bool vvHud::update()
{
    if (doHide)
    {
        if (vv->frameTime() - hudTime >= logoTime)
        {
            hudTime = vv->frameTime();
            hide();
            doHide = false;

            return true;
        }
    }

    return false;
}

bool vvHud::isVisible() const
{
    return visible || doHide;
}

void vvHud::hideLater(float time)
{
    if (visible)
    {
        doHide = true;
        hudTime = vv->frameTime();
        if (time < 0.)
            logoTime = coCoviseConfig::getFloat("VIVE.LogoTime", 1.0);
        else
            logoTime = time;
    }
}

void vvHud::show()
{
    if (!visible)
    {
        visible = true;
        doHide = false;
        //vv->getScene()->addChild(camera);
    }
}

void vvHud::hide()
{
    if (visible)
    {
        visible = false;
        //vv->getScene()->removeChild(camera);
    }
}

vvHud::~vvHud()
{
    hide();
    instance_ = NULL;
}
