/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_PRESETS
#define SG_VRUI_PRESETS

#include <OpenVRUI/coUIElement.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGMaterialChunk.h>
#include <OpenSG/OSGBlendChunk.h>
#include <OpenSG/OSGPolygonChunk.h>
#include <OpenSG/OSGNode.h>
#include <OpenSG/OSGRefPtr.h>

#include <vector>

class SGVRUIEXPORT SGVruiPresets
{

private:
    SGVruiPresets();
    virtual ~SGVruiPresets();

    static SGVruiPresets *instance;

public:
    static osg::ChunkMaterialPtr getStateSet(coUIElement::Material material);
    static osg::ChunkMaterialPtr getStateSetCulled(coUIElement::Material material);
    static osg::ChunkMaterialPtr makeStateSet(coUIElement::Material material);
    static osg::ChunkMaterialPtr makeStateSetCulled(coUIElement::Material material);
    static osg::MaterialChunkPtr getMaterial(coUIElement::Material material);
    //static osg::TexEnv      * getTexEnvModulate ();
    static osg::PolygonChunkPtr getPolyChunkFill();
    static osg::PolygonChunkPtr getPolyChunkFillCulled();
    //static osg::CullFace    * getCullFaceBack   ();
    static osg::BlendChunkPtr getBlendOneMinusSrcAlpha();

private:
    std::vector<osg::RefPtr<osg::ChunkMaterialPtr> > stateSets;
    std::vector<osg::RefPtr<osg::ChunkMaterialPtr> > stateSetsCulled;
    std::vector<osg::RefPtr<osg::MaterialChunkPtr> > materials;
    //osg::ref_ptr<osg::TexEnv> texEnvModulate;
    osg::PolygonChunkPtr polyModeFill;
    osg::PolygonChunkPtr polyModeFillCull;
    osg::BlendChunkPtr oneMinusSourceAlphaBlendFunc;
};
#endif
