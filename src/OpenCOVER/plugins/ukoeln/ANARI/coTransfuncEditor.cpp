/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/TexEnv>
#include <osg/Texture2D>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <cover/coIntersection.h>
#include "coTransfuncEditor.h"

using namespace vrui;
using namespace opencover;

class Canvas : public coAction,
               public coUIElement
{
public:
    Canvas() : dcs(new OSGVruiTransformNode(new osg::MatrixTransform()))
    {
        unsigned numColors=3;
        color.data.resize(numColors*3);
        color.data[0*3] = 0.f; color.data[0*3+1] = 0.f; color.data[0*3+2] = 1.f;
        color.data[1*3] = 0.f; color.data[1*3+1] = 1.f; color.data[1*3+2] = 0.f;
        color.data[2*3] = 1.f; color.data[2*3+1] = 0.f; color.data[2*3+2] = 0.f;
        color.updated = true;

        opacity.data.resize(width);
        for (int x=0; x<width; ++x) {
            opacity.data[x] = x/(width-1.f);
            opacity.updated = true;
        }

        dcs->getNodePtr()->asGroup()->addChild(createGeode());
        coIntersection::getIntersectorForAction("coAction")->add(dcs, this);
        interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "TFE Canvas", coInteraction::Menu);
    }

   ~Canvas()
    {
        delete interactionA;

        coIntersection::getIntersectorForAction("coAction")->remove(this);
        dcs->removeAllChildren();
        dcs->removeAllParents();
        delete dcs;
    }

    void update()
    {
        if (unregister && interactionA->wasStopped()) {
            if (interactionA->isRegistered()) {
                coInteractionManager::the()->unregisterInteraction(interactionA);
            }
            xPrev = -1.f;
            unregister = false;
        }

        if (color.updated) {
            if (color.updateFunc) color.updateFunc(color.data.data(),
                                                   color.data.size(),
                                                   color.userData);
        }

        if (opacity.updated) {
            if (opacity.updateFunc) opacity.updateFunc(opacity.data.data(),
                                                       opacity.data.size(),
                                                       opacity.userData);
            opacity.updated = false;
        }
    }

    int hit(vruiHit *hit)
    {
        if (!interactionA->isRegistered()) {
            coInteractionManager::the()->registerInteraction(interactionA);
            interactionA->setHitByMouse(hit->isMouseHit());
        }

        if (interactionA->wasStopped()) {
            xPrev = -1.f;
        }

        if (interactionA->isRunning()) {
            int x = hit->getLocalIntersectionPoint()[0];
            int y = hit->getLocalIntersectionPoint()[1];

            if (xPrev < 0.f || xPrev >= width) xPrev = hit->getLocalIntersectionPoint()[0];

            for (int xi=std::min((int)xPrev,x); xi<=std::max((int)xPrev,x); ++xi) {
                int index = std::max(0,std::min(int(width)-1,xi));
                float val = y > 8 ? y/(height-1.f) : 0.f;
                opacity.data[index] = val;
                opacity.updated = true;
            }
            xPrev = hit->getLocalIntersectionPoint()[0];
            updateAlphaTextureData();
            auto image = alphaTexture->getImage();
            if (!image) {
                image = new osg::Image;
                alphaTexture->setImage(image);
            }
            image->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                            alphaTextureData.data(), osg::Image::NO_DELETE, 4);
            geom->dirtyDisplayList();
        }

        return ACTION_DONE;
        //return ACTION_CALL_ON_MISS;
    }

    void miss()
    {
        unregister = true;
    }

    float getWidth() const
    { return width; }

    float getHeight() const
    { return height; }

    float getXpos() const
    { return xPos; }

    float getYpos() const
    { return yPos; }

    void setPos(float x, float y, float)
    {
        xPos = x;
        yPos = y;
        dcs->setTranslation(x, y + getHeight(), 0.f);
    }

    vruiTransformNode *getDCS()
    {
        return dcs;
    }

    osg::ref_ptr<osg::Geode> createGeode()
    {
        vertices = new osg::Vec3Array(4);
        (*vertices)[0].set(0.f, 0.f, 1e-1f);
        (*vertices)[1].set(width, 0.f, 1e-1f);
        (*vertices)[2].set(width, height, 1e-1f);
        (*vertices)[3].set(0.f, height, 1e-1f);

        texcoords = new osg::Vec2Array(4);
        (*texcoords)[0].set(0.f, 0.f);
        (*texcoords)[1].set(1.f, 0.f);
        (*texcoords)[2].set(1.f, 1.f);
        (*texcoords)[3].set(0.f, 1.f);

        // color tex
        colorTexture = new osg::Texture2D;
        colorTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
        colorTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
        colorTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP);
        colorTexture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP);

        updateColorTextureData();

        osg::ref_ptr<osg::Image> colorImage = new osg::Image;
        colorImage->setImage(color.data.size()/3, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                             colorTextureData.data(), osg::Image::NO_DELETE, 4);
        colorTexture->setImage(colorImage.get());

        // alpha tex
        alphaTexture = new osg::Texture2D;
        alphaTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
        alphaTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
        alphaTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP);
        alphaTexture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP);

        updateAlphaTextureData();

        osg::ref_ptr<osg::Image> alphaImage = new osg::Image;
        alphaImage->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                             alphaTextureData.data(), osg::Image::NO_DELETE, 4);
        alphaTexture->setImage(alphaImage.get());

        stateset = new osg::StateSet;
        stateset->setGlobalDefaults();
        stateset->setTextureAttributeAndModes(0, colorTexture.get(), osg::StateAttribute::ON);
        stateset->setTextureAttributeAndModes(1, alphaTexture.get(), osg::StateAttribute::ON);

        osg::ref_ptr<osg::TexEnv> texenv = new osg::TexEnv;
        texenv->setMode(osg::TexEnv::BLEND);
        stateset->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), osg::StateAttribute::ON);
        stateset->setTextureAttributeAndModes(1, OSGVruiPresets::getTexEnvModulate(), osg::StateAttribute::ON);

        geom = new osg::Geometry;
        geom->setVertexArray(vertices.get());
        geom->setTexCoordArray(0,texcoords.get());
        geom->setTexCoordArray(1,texcoords.get());
        geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));
    
        geode = new osg::Geode;
        geode->setStateSet(stateset.get());
        geode->addDrawable(geom);

        return geode.get();
    }

    void updateColorTextureData()
    {
        unsigned numColors = color.data.size()/3;
        colorTextureData.resize(numColors * height * 4);
        for (unsigned x=0; x<numColors; ++x) {
            for (int y=0; y<height; ++y) {
                colorTextureData[(y*numColors+x)*4] = color.data[x*3]*255;
                colorTextureData[(y*numColors+x)*4+1] = color.data[x*3+1]*255;
                colorTextureData[(y*numColors+x)*4+2] = color.data[x*3+2]*255;
                colorTextureData[(y*numColors+x)*4+3] = 255;
            }
        }
    }

    void updateAlphaTextureData()
    {
        alphaTextureData.resize(width * height * 4);
        for (int x=0; x<width; ++x) {
            for (int y=0; y<height; ++y) {
                float valy = y/float(height-1);
                if (valy < opacity.data[x]) {
                    alphaTextureData[(y*(int)width+x)*4] = 255;
                    alphaTextureData[(y*(int)width+x)*4+1] = 255;
                    alphaTextureData[(y*(int)width+x)*4+2] = 255;
                    alphaTextureData[(y*(int)width+x)*4+3] = 128;
                } else {
                    alphaTextureData[(y*(int)width+x)*4] = 0;
                    alphaTextureData[(y*(int)width+x)*4+1] = 0;
                    alphaTextureData[(y*(int)width+x)*4+2] = 0;
                    alphaTextureData[(y*(int)width+x)*4+3] = 0;
                }
            }
        }
    }

    void setColorUpdateFunc(coColorUpdateFunc func, void *userData)
    {
        color.updateFunc = func;
        color.userData = userData;
    }

    void setOpacityUpdateFunc(coOpacityUpdateFunc func, void *userData)
    {
        opacity.updateFunc = func;
        opacity.userData = userData;
    }
private:
    float width=512.f, height=256.f;
    float xPos=0.f, yPos=0.f, xPrev=-1.f;
    vrui::OSGVruiTransformNode *dcs{nullptr};
    vrui::coCombinedButtonInteraction *interactionA{nullptr};
    bool unregister{false}; // mouse left, unregister interaction(s)

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::StateSet> stateset;
    osg::ref_ptr<osg::Vec3Array> vertices;
    osg::ref_ptr<osg::Vec2Array> texcoords;

    osg::ref_ptr<osg::Texture2D> colorTexture;
    std::vector<unsigned char> colorTextureData;

    osg::ref_ptr<osg::Texture2D> alphaTexture;
    std::vector<unsigned char> alphaTextureData;

    struct {
        std::vector<float> data;
        coColorUpdateFunc updateFunc{nullptr};
        void *userData{nullptr};
        bool updated{false};
    } color;

    struct {
        std::vector<float> data;
        coOpacityUpdateFunc updateFunc{nullptr};
        void *userData{nullptr};
        bool updated{false};
    } opacity;
};

coTransfuncEditor::coTransfuncEditor()
{
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    handle = new coPopupHandle("TFE");
    canvas = new Canvas;
    panel->addElement(canvas);
    panel->resize();

    frame = new coFrame("UI/Frame");
    frame->addElement(panel);
    handle->addElement(frame);

    show();
}

coTransfuncEditor::~coTransfuncEditor()
{
    delete handle;
    delete panel;
    delete frame;
}

void coTransfuncEditor::show()
{
    handle->setVisible(true);
    panel->show(canvas);
}

void coTransfuncEditor::hide()
{
    //panel->hide(canvas);
    handle->setVisible(false);
}

void coTransfuncEditor::update()
{
    canvas->update();
    handle->update();
}

void coTransfuncEditor::setColorUpdateFunc(coColorUpdateFunc func, void *userData)
{
    canvas->setColorUpdateFunc(func, userData);
}

void coTransfuncEditor::setOpacityUpdateFunc(coOpacityUpdateFunc func, void *userData)
{
    canvas->setOpacityUpdateFunc(func, userData);
}
