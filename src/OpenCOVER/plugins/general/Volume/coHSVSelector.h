/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_HSV_SELECTOR_H
#define CO_HSV_SELECTOR_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coUIElement.h>
#include <cover/coCollabInterface.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/Texture2D>

class coHSVSelector;
class coPreviewCube;
class coFunctionEditor;

namespace vrui
{
class coCombinedButtonInteraction;
}

class coHSVSelector : public vrui::coAction, public vrui::coUIElement, public vrui::vruiCollabInterface
{
public:
    coHSVSelector(coPreviewCube *prevCube, coFunctionEditor *functionEditor);
    coHSVSelector(coFunctionEditor *functionEditor);
    virtual ~coHSVSelector();

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vrui::vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void setPos(float x, float y, float z = 0);
    void setCross(float x, float y);
    void setBrightness(float v);
    void setColorRGB(float, float, float);
    void setColorHSB(float, float, float);
    vrui::vruiTransformNode *getDCS();
    virtual float getWidth() const
    {
        return 2.0 * (A + B) + C;
    }
    virtual float getHeight() const
    {
        return 2.0 * (A + B) + C;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    virtual void update();

    virtual void createGeometry();
    virtual void resizeGeometry();

    osg::ref_ptr<osg::Group> createGeodes();

protected:
    vrui::OSGVruiTransformNode *myDCS;
    osg::ref_ptr<osg::MatrixTransform> crossDCS;
    float myX, myY;
    float crossX, crossY;
    float brightness;
    unsigned char *textureData;
    coFunctionEditor *myFunctionEditor;
    bool unregister;

    osg::ref_ptr<osg::Texture2D> tex;
    osg::ref_ptr<osg::Texture2D> crossTex;
    void createLists();
    void createCross();
    float A;
    float B;
    float C;
    float D;
    float OFFSET;
    float lastRoll;
    osg::ref_ptr<osg::Vec4Array> color;
    osg::ref_ptr<osg::Vec3Array> coord;
    osg::ref_ptr<osg::Vec3Array> coordt;
    osg::ref_ptr<osg::Vec3Array> coordCross;
    osg::ref_ptr<osg::Vec3Array> normal;
    osg::ref_ptr<osg::Vec3Array> normalt;
    osg::ref_ptr<osg::Vec2Array> texcoord;

    osg::ref_ptr<osg::DrawElementsUShort> coordIndex;
    osg::ref_ptr<osg::UShortArray> normalIndex;

    osg::ref_ptr<osg::Material> textureMat;
    osg::ref_ptr<osg::StateSet> normalGeostate;
    osg::ref_ptr<osg::Geode> crosshair;
    coPreviewCube *myCube;
    virtual void remoteLock(const char *message);
    virtual void remoteOngoing(const char *message);
    virtual void releaseRemoteLock(const char *message);
    virtual void sendLockMessageLocal();
    vrui::coCombinedButtonInteraction *interactionA; ///< interaction for first button
    vrui::coCombinedButtonInteraction *interactionB; ///< interaction for second button
};
#endif
