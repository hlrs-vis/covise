/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_PRESETS
#define OSG_VRUI_PRESETS

#include <OpenVRUI/coUIElement.h>

#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/CullFace>
#include <osg/TexEnv>
#include <osg/PolygonMode>
#include <osg/Node>

#include <vector>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiPresets
{

private:
    OSGVruiPresets();
    virtual ~OSGVruiPresets();

    static OSGVruiPresets *instance;

public:
    static osg::StateSet *getStateSet(coUIElement::Material material);
    static osg::StateSet *getStateSetCulled(coUIElement::Material material);
    static osg::StateSet *makeStateSet(coUIElement::Material material);
    static osg::StateSet *makeStateSetCulled(coUIElement::Material material);
    static osg::Material *getMaterial(coUIElement::Material material);
    static osg::TexEnv *getTexEnvModulate();
    static osg::PolygonMode *getPolyModeFill();
    static osg::CullFace *getCullFaceBack();
    static osg::BlendFunc *getBlendOneMinusSrcAlpha();
    static void makeTransparent(osg::StateSet *state, bool continuous = false);
    static std::string getFontFile();

private:
    std::vector<osg::ref_ptr<osg::StateSet> > stateSets;
    std::vector<osg::ref_ptr<osg::StateSet> > stateSetsCulled;
    std::vector<osg::ref_ptr<osg::Material> > materials;
    osg::ref_ptr<osg::TexEnv> texEnvModulate;
    osg::ref_ptr<osg::PolygonMode> polyModeFill;
    osg::ref_ptr<osg::CullFace> cullFaceBack;
    osg::ref_ptr<osg::BlendFunc> oneMinusSourceAlphaBlendFunc;
    std::string fontFile;

    void setColorFromConfig(const char *configEntry, int materialIndex, osg::Vec4 def);
};
}
#endif
