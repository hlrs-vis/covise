/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_PIN_EDITOR_H
#define CO_PIN_EDITOR_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coUIElement.h>
#include "coPin.h"
#include <OpenVRUI/sginterface/vruiCollabInterface.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Texture2D>

#include <list>

#define TEXTURE_RES_BACKGROUND 256

class coPinEditor;
class vvTransFunc;
class coPreviewCube;
class PointerTooltip;
class coDefaultFunctionEditor;

class vvTFPyramid;

namespace vrui
{
class coLabel;
class coCombinedButtonInteraction;
class coColoredBackground;
class vruiHit;
class OSGVruiTransformNode;
}

class coPinEditor : public vrui::coAction, public vrui::coUIElement, public vrui::vruiCollabInterface
{
public:
    enum EditMode
    {
        SELECTION = 0, ///< selectionMode
        EDIT_COLOR = 1, ///< Color Edit Mode
        EDIT_RAMP = 2, ///< Alpha ramp Edit Mode
        EDIT_BLANK = 3, ///< Alpha blank Edit Mode
        EDIT_HAT = 4, ///< Alpha hat Edit Mode
        ADD_PIN = 5, ///< addNew Pin
        MIX_CHANNELS01 = 6, ///< blend weights for channel 0/1
    };
    enum PinType
    {
        COLOR = 0, ///< color pin (RGB)
        ALPHA_HAT = 1, ///< alpha hat
        POS_ALPHA_RAMP = 2, ///< positive alpha ramp
        NEG_ALPHA_RAMP = 3, ///< negative alpha ramp
        ALPHA_BLANK = 4, ///< blank Region
    };

    EditMode mode;
    void setMode(EditMode newMode);
    coPin *currentPin;
    coPinEditor(vvTransFunc *transFunc, coDefaultFunctionEditor *functionEditor);
    virtual ~coPinEditor();
    void setTransFuncPtr(vvTransFunc *transFunc);

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vrui::vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    virtual void createGeometry();
    virtual void resizeGeometry();

    void setPos(float x, float y, float z = 0);
    void setColor(float h, float s, float v, int context = -1);
    void setBrightness(float v, int context = -1);
    void setTopWidth(float s, int context = -1);
    void setBotWidth(float w, int context = -1);
    void setMax(float m, int context = -1);
    void selectPin(float x, float y);

    vrui::vruiTransformNode *getDCS();

    virtual float getWidth() const
    {
        return 2.0 * (A + B) + W;
    }
    virtual float getHeight() const
    {
        return 2.0 * (A + B) + H + SELH + A + B + COLORH + A + B;
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
    void updatePinList(float minv = 0.0f, float maxv = 1.0f); // update my pinList to reflect the transferFunction

    osg::ref_ptr<osg::Group> createBackgroundGroup();

    void showSelectionBar();
    void hideSelectionBar();
    void updateColorBar();
    void undoAddPin();
    void addPin(int type, int local = 1);
    void deleteCurrentPin();
    void deletePin(int ID);
    void deleteAllPins();
    void init();
    void sortPins();
    void updateBackground(unsigned char *backgroundTextureData);
    void setBackgroundType(int mode);

protected:
    std::list<coPin *> pinList;

    vrui::OSGVruiTransformNode *myDCS;

    osg::ref_ptr<osg::MatrixTransform> pinDCS;

    float myX, myY;
    unsigned char *textureData;
    bool unregister;
    osg::ref_ptr<osg::Texture2D> tex;
    osg::ref_ptr<osg::Texture2D> histoTex;
    vvTransFunc *myTransFunc;
    coDefaultFunctionEditor *myFunctionEditor;
    vrui::coColoredBackground *labelBackground;

    osg::ref_ptr<osg::StateSet> HistoBackgroundGeostate;
    osg::ref_ptr<osg::StateSet> NormalBackgroundGeostate;

    osg::ref_ptr<osg::Geode> backgroundGeode;
    osg::ref_ptr<osg::Group> backgroundGroup;
    osg::ref_ptr<osg::Geometry> backgroundGeometry;

    int backgroundMode;
    vrui::coLabel *currentScalarLabel;
    void createLists();
    bool isNearestSelected(float x, float y);
    float A;
    float B;
    float W;
    float H;
    float SELH;
    float COLORH;
    float OFFSET;
    float lastRoll;
    float w1, w2, w3, w4;
    int selectedRegion;
    float pickThreshold;
    float moveThreshold;
    bool doMove;
    double pickTime;
    float pickCoordX, pickCoordY;

    osg::ref_ptr<osg::Vec4Array> color;
    osg::ref_ptr<osg::Vec3Array> coord;
    osg::ref_ptr<osg::Vec3Array> coordSel;
    osg::ref_ptr<osg::Vec3Array> coordColor;
    osg::ref_ptr<osg::Vec3Array> coordt;
    osg::ref_ptr<osg::Vec3Array> coordColort;
    osg::ref_ptr<osg::Vec3Array> normal;
    osg::ref_ptr<osg::Vec3Array> normalSel;
    osg::ref_ptr<osg::Vec3Array> normalt;
    osg::ref_ptr<osg::Vec2Array> texcoord;
    osg::ref_ptr<osg::Vec2Array> texcoordColor;

    osg::ref_ptr<osg::DrawElementsUShort> vertices;
    osg::ref_ptr<osg::DrawElementsUShort> verticesSel;

    osg::ref_ptr<osg::Material> textureMat;
    osg::ref_ptr<osg::StateSet> normalGeostate;

    osg::ref_ptr<osg::Vec4Array> selectionBarColor;
    osg::ref_ptr<osg::Vec3Array> selectionBarCoord;
    osg::ref_ptr<osg::Vec3Array> selectionBarNormal;

    osg::ref_ptr<osg::DrawElementsUShort> selectionBarVertices;

    void adjustSelectionBar();
    void createSelectionBarLists();

    osg::ref_ptr<osg::Geometry> selectionBarGeometry;
    osg::ref_ptr<osg::Geode> selectionBarGeode;

    osg::ref_ptr<osg::Geode> createSelectionBarGeode();

    virtual void remoteLock(const char *message);
    virtual void remoteOngoing(const char *message);
    virtual void releaseRemoteLock(const char *message);

    void highlightCurrent();

    vrui::coCombinedButtonInteraction *interactionA; ///< interaction for first button
    vrui::coCombinedButtonInteraction *interactionB; ///< interaction for second button
    vrui::coCombinedButtonInteraction *interactionC; ///< interaction for third button

private:
    inline coPin *findPin(int context);
};
#endif
