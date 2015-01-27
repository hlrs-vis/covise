/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiDefaultButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <osg/Geode>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osgText/Font>
#include <osgText/String>
#include <osg/StateSet>

#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

using namespace std;
using namespace osg;
using namespace osgText;

namespace vrui
{

OSGVruiDefaultButtonGeometry::OSGVruiDefaultButtonGeometry(coDefaultButtonGeometry *geometry)
    : vruiButtonProvider(geometry)
    , normalNode(0)
    , pressedNode(0)
    , highlightNode(0)
    , pressedHighlightNode(0)
    , disabledNode(0)
    , myDCS(0)
{
    textString = geometry->getTextureName();
}

OSGVruiDefaultButtonGeometry::~OSGVruiDefaultButtonGeometry()
{
}

void OSGVruiDefaultButtonGeometry::createGeometry()
{

    if (normalNode == 0)
    {

        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = createNode(false, true);
        pressedHighlightNode = createNode(true, true);
        disabledNode = createNode(false, false, true);

        ref_ptr<MatrixTransform> transformNode = new MatrixTransform();
        switchNode = new Switch();

        switchNode->addChild(normalNode.get());
        switchNode->addChild(pressedNode.get());
        switchNode->addChild(highlightNode.get());
        switchNode->addChild(pressedHighlightNode.get());
        switchNode->addChild(disabledNode.get());

        transformNode->addChild(switchNode.get());

        myDCS = new OSGVruiTransformNode(transformNode.get());
    }
}

ref_ptr<StateSet> OSGVruiDefaultButtonGeometry::createGeoState(bool highlighted)
{

    ref_ptr<Material> material;
    ref_ptr<StateSet> geostate;

    // Create material:
    material = new Material();

    material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    if (highlighted)
    {
        material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.3f, 0.5f, 0.0f, 1.0f));
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.0f, 1.0f));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.0f, 1.0f));
    }
    else
    {
        material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    }

    material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(Material::FRONT_AND_BACK, 80.0f);

    // Create GeoState:
    geostate = new StateSet();
    geostate->setGlobalDefaults();

    geostate->setAttributeAndModes(material.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    //geostate->setAttr(PFSTATE_BACKMTL,  material);
    geostate->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    //geostate->setMode(PFSTATE_TRANSPARENCY, PFTR_OFF);

    return geostate;
}

ref_ptr<Text> OSGVruiDefaultButtonGeometry::createText(const string &textString,
                                                       Text::AlignmentType align,
                                                       float size)
{

    ref_ptr<Text> text = new Text();

    text->setDataVariance(Object::DYNAMIC);
    text->setText(textString, String::ENCODING_UTF8);
    text->setFont(OSGVruiPresets::getFontFile());
    text->setAlignment(align);
    text->setCharacterSize(size);
    text->setDrawMode(Text::TEXT);

    return text;
}

ref_ptr<Node> OSGVruiDefaultButtonGeometry::createNode(bool pressed, bool highlighted, bool disabled)
{

    ref_ptr<ShapeDrawable> geometry;
    ref_ptr<Geode> geode;
    ref_ptr<Geode> textNode;
    ref_ptr<MatrixTransform> transform;
    ref_ptr<MatrixTransform> buttonTransform;
    ref_ptr<MatrixTransform> textTransform;

    ref_ptr<Cylinder> cylinder = new Cylinder(Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f);
    ref_ptr<TessellationHints> th = new TessellationHints();

    th->setTargetNumFaces(DETAIL_LEVEL);
    th->setCreateFrontFace(true);
    th->setCreateBackFace(true);
    th->setCreateNormals(true);
    th->setCreateTop(true);
    th->setCreateBottom(true);
    th->setCreateBody(true);
    th->setTessellationMode(TessellationHints::USE_TARGET_NUM_FACES);

    geometry = new ShapeDrawable(cylinder.get(), th.get());

    geode = new Geode();
    geode->setStateSet(createGeoState(highlighted).get());

    geode->setName(textString);
    geode->addDrawable(geometry.get());

    transform = new MatrixTransform();
    char *name = new char[256 + strlen(textString.c_str())];
    sprintf(name, "OSGVruiDefaultButtonGeometry%s%s%s(%s)",
            pressed ? "-pressed" : "",
            highlighted ? "-highlighted" : "",
            disabled ? "-disabled" : "",
            textString.c_str());

    transform->setName(name);
    delete[] name;

    buttonTransform = new MatrixTransform();
    Matrix matrix;
    Matrix scaleMatrix;
    if (pressed)
    {
        matrix.makeTranslate(0.0, 0.0, 2.0);
        scaleMatrix.makeScale(10.0, 5.0, 2.0);
        matrix *= scaleMatrix;
    }
    else
    {
        matrix.makeTranslate(0.0, 0.0, 5.0);
        scaleMatrix.makeScale(10.0, 5.0, 5.0);
        matrix *= scaleMatrix;
    }
    buttonTransform->setMatrix(matrix);

    buttonTransform->addChild(geode.get());
    transform->addChild(buttonTransform.get());

    textTransform = new MatrixTransform();
    matrix.makeTranslate(0.0f, 0.0f, 3.0f);
    matrix.rotate(270.0f, Vec3(1.0f, 0.0f, 0.0f));
    textTransform->setMatrix(matrix);
    transform->addChild(textTransform.get());

    textNode = new Geode();
    textNode->addDrawable(createText(textString, Text::CENTER_CENTER, 8.0f).get());
    ref_ptr<Material> textMaterial = new Material();

    textMaterial->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    textMaterial->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    textMaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    textMaterial->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    textMaterial->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    textMaterial->setShininess(Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<StateSet> textState = textNode->getOrCreateStateSet();

    textState->setAttributeAndModes(textMaterial.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    textState->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);

    textTransform->addChild(textNode.get());

    return transform.get();
}

void OSGVruiDefaultButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}

void OSGVruiDefaultButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *OSGVruiDefaultButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

float OSGVruiDefaultButtonGeometry::getWidth() const
{
    return 10.0f;
}

float OSGVruiDefaultButtonGeometry::getHeight() const
{
    return 5.0f;
}
}
