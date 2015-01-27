/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiPresets.h>

#include <OpenVRUI/util/vruiLog.h>

OSG_USING_NAMESPACE

SGVruiPresets *SGVruiPresets::instance = 0;

SGVruiPresets::SGVruiPresets()
{

    polyModeFill = PolygonChunk::create();
    beginEditCP(polyModeFill);
    polyModeFill->setFrontMode(GL_FILL);
    polyModeFill->setBackMode(GL_FILL);
    polyModeFill->setCullFace(GL_NONE);
    endEditCP(polyModeFill);

    polyModeFillCull = PolygonChunk::create();
    beginEditCP(polyModeFillCull);
    polyModeFillCull->setFrontMode(GL_FILL);
    polyModeFillCull->setBackMode(GL_FILL);
    polyModeFillCull->setCullFace(GL_BACK);
    endEditCP(polyModeFillCull);

    oneMinusSourceAlphaBlendFunc = BlendChunk::create();
    beginEditCP(oneMinusSourceAlphaBlendFunc);
    oneMinusSourceAlphaBlendFunc->setSrcFactor(GL_SRC_ALPHA);
    oneMinusSourceAlphaBlendFunc->setDestFactor(GL_ONE_MINUS_SRC_ALPHA);
    endEditCP(oneMinusSourceAlphaBlendFunc);

    for (int i = 0; i < coUIElement::NUM_MATERIALS; ++i)
    {

        RefPtr<MaterialChunkPtr> material(MaterialChunk::create());
        beginEditCP(material);

        material->setColorMaterial(GL_AMBIENT_AND_DIFFUSE);
        material->setSpecular(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
        material->setEmission(Color4f(0.0f, 0.0f, 0.0f, 1.0f));
        material->setShininess(16.0f);
        material->setLit(true);

        materials.push_back(material);
    }

    materials[coUIElement::RED]->setAmbient(Color4f(0.2f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::RED]->setDiffuse(Color4f(1.0f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::GREEN]->setAmbient(Color4f(0.0f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::GREEN]->setDiffuse(Color4f(0.0f, 1.0f, 0.0f, 1.0f));
    materials[coUIElement::BLUE]->setAmbient(Color4f(0.0f, 0.0f, 0.2f, 1.0f));
    materials[coUIElement::BLUE]->setDiffuse(Color4f(0.0f, 0.0f, 1.0f, 1.0f));
    materials[coUIElement::YELLOW]->setAmbient(Color4f(0.2f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::YELLOW]->setDiffuse(Color4f(1.0f, 1.0f, 0.0f, 1.0f));
    materials[coUIElement::GREY]->setAmbient(Color4f(0.3f, 0.3f, 0.3f, 1.0f));
    materials[coUIElement::GREY]->setDiffuse(Color4f(0.3f, 0.3f, 0.3f, 1.0f));
    materials[coUIElement::WHITE]->setAmbient(Color4f(0.3f, 0.3f, 0.3f, 1.0f));
    materials[coUIElement::WHITE]->setDiffuse(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::BLACK]->setAmbient(Color4f(0.0f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::BLACK]->setDiffuse(Color4f(0.0f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::DARK_YELLOW]->setAmbient(Color4f(0.2f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::DARK_YELLOW]->setDiffuse(Color4f(0.2f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setAmbient(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setEmission(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setDiffuse(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setLit(false);

    for (int i = 0; i < coUIElement::NUM_MATERIALS; ++i)
    {

        RefPtr<ChunkMaterialPtr> stateSet(ChunkMaterial::create());
        RefPtr<ChunkMaterialPtr> stateSetCulled(ChunkMaterial::create());

        beginEditCP(stateSet);
        beginEditCP(stateSetCulled);

        stateSet->addChunk(materials[i]);
        stateSet->addChunk(polyModeFill);

        stateSetCulled->addChunk(materials[i]);
        stateSetCulled->addChunk(polyModeFillCull);

        stateSets.push_back(stateSet);
        stateSetsCulled.push_back(stateSetCulled);

        endEditCP(stateSet);
        endEditCP(stateSetCulled);
    }
}

SGVruiPresets::~SGVruiPresets()
{
}

ChunkMaterialPtr SGVruiPresets::getStateSet(coUIElement::Material material)
{

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("SGVruiPresets::getStateSet err: material " << material << " not found");
        return NullFC;
    }

    if (instance == 0)
        instance = new SGVruiPresets();

    return instance->stateSets[material];
}

ChunkMaterialPtr SGVruiPresets::getStateSetCulled(coUIElement::Material material)
{

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("SGVruiPresets::getStateSetCulled err: material " << material << " not found");
        return NullFC;
    }

    if (instance == 0)
        instance = new SGVruiPresets();

    return instance->stateSetsCulled[material];
}

ChunkMaterialPtr SGVruiPresets::makeStateSet(coUIElement::Material material)
{

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("SGVruiPresets::makeStateSet err: material " << material << " not found");
        return NullFC;
    }

    if (instance == 0)
        instance = new SGVruiPresets();

    return ChunkMaterialPtr::dcast(deepClone(instance->stateSets[material]));
}

ChunkMaterialPtr SGVruiPresets::makeStateSetCulled(coUIElement::Material material)
{

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("SGVruiPresets::makeStateSetCulled err: material " << material << " not found");
        return NullFC;
    }

    if (instance == 0)
        instance = new SGVruiPresets();

    return ChunkMaterialPtr::dcast(deepClone(instance->stateSetsCulled[material]));
}

MaterialChunkPtr SGVruiPresets::getMaterial(coUIElement::Material material)
{

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("SGVruiPresets::getMaterial err: material " << material << " not found");
        return NullFC;
    }

    if (instance == 0)
        instance = new SGVruiPresets();

    return instance->materials[material];
}

// TexEnv * OSGVruiPresets::getTexEnvModulate()
// {
//    if (instance == 0) instance = new OSGVruiPresets();
//    return instance->texEnvModulate.get();
// }

PolygonChunkPtr SGVruiPresets::getPolyChunkFill()
{
    if (instance == 0)
        instance = new SGVruiPresets();
    return instance->polyModeFill;
}

PolygonChunkPtr SGVruiPresets::getPolyChunkFillCulled()
{
    if (instance == 0)
        instance = new SGVruiPresets();
    return instance->polyModeFillCull;
}

BlendChunkPtr SGVruiPresets::getBlendOneMinusSrcAlpha()
{
    if (instance == 0)
        instance = new SGVruiPresets();
    return instance->oneMinusSourceAlphaBlendFunc;
}
