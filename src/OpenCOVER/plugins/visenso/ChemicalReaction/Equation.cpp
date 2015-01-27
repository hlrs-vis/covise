/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Equation.h"

#include <osg/Version>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
using namespace opencover;
using namespace covise;

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

Equation::Equation()
{
    group = new osg::Group();

    // material
    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    material = new osg::Material;
    material->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    material->setEmission(osg::Material::FRONT_AND_BACK, textColor);

    textDrawable = new osgText::Text();
    textDrawable->setCharacterSize(0.8f);
    // LEFT_TOP would be better in the following line because we want to position the text below an object.
    // However, the bounding box calculation doesn't seem to work properly for LEFT_TOP.
    // It might be a bug in OpenSceneGraph.
    textDrawable->setAlignment(osgText::Text::LEFT_BOTTOM);
    textDrawable->setAxisAlignment(osgText::Text::XZ_PLANE);
    textDrawable->setPosition(osg::Vec3(-8.0f, 0.0f, -9.1f));
    textDrawable->setFont(coVRFileManager::instance()->getFontFile(NULL));

    textGeode = new osg::Geode();
    textGeode->addDrawable(textDrawable.get());
    textGeode->getOrCreateStateSet()->setAttributeAndModes(material.get());

    group->addChild(textGeode.get());

    createArrow();
}

Equation::~Equation()
{
}

void Equation::setVisible(bool visible)
{
    if (cover->getObjectsRoot()->containsNode(group.get()))
    {
        if (!visible)
            cover->getObjectsRoot()->removeChild(group.get());
    }
    else
    {
        if (visible)
            cover->getObjectsRoot()->addChild(group.get());
    }
}

// Form: "3O2 + 4Fe > 2Fe2O3"
// every number without a leading space will be lowered
// > will be replaced with an arrow
void Equation::setEquation(std::string e)
{
    if (e.compare(equation) != 0)
    {
        equation = e;

        // clear helpers
        for (std::vector<osg::ref_ptr<osg::Geode> >::iterator it = helperGeodes.begin(); it < helperGeodes.end(); ++it)
        {
            group->removeChild((*it).get());
        }
        helperGeodes.clear();

        // remove arrow
        if (group->containsNode(arrowTransform.get()))
            group->removeChild(arrowTransform.get());

        // clear text
        std::string growingText = "";

        // loop
        char lastC = ' ';
        for (int i = 0; i < e.length(); ++i)
        {
            char c = e[i];
            if ((c > 47) && (c < 58) && (lastC != ' '))
            {
                textDrawable->setText(growingText);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
                osg::BoundingBox bb = textDrawable->computeBoundingBox();
#else
                osg::BoundingBox bb = textDrawable->computeBound();
#endif

                osgText::Text *helper = new osgText::Text();
                helper->setCharacterSize(0.5f);
                helper->setAlignment(osgText::Text::LEFT_TOP);
                helper->setAxisAlignment(osgText::Text::XZ_PLANE);
                helper->setPosition(osg::Vec3(bb.xMax(), 0.0f, bb.zMin() + 0.1f));
                helper->setFont(coVRFileManager::instance()->getFontFile(NULL));
                helper->setText(std::string("") + c);

                osg::ref_ptr<osg::Geode> helperGeode = new osg::Geode();
                helperGeodes.push_back(helperGeode);
                helperGeode->addDrawable(helper);
                helperGeode->getOrCreateStateSet()->setAttributeAndModes(material.get());

                group->addChild(helperGeode.get());

                growingText += " ";
            }
            else if (c == '>')
            {
                textDrawable->setText(growingText);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
                osg::BoundingBox bb = textDrawable->computeBoundingBox();
#else
                osg::BoundingBox bb = textDrawable->computeBound();
#endif

                osg::Matrix m;
                m.makeScale(0.08f, 0.08f, 0.08f);
                m.postMultTranslate(osg::Vec3(bb.xMax() + 0.2f, 0.0f, bb.center()[2]));
                arrowTransform->setMatrix(m);
                group->addChild(arrowTransform.get());

                growingText += "    ";
            }
            else
            {
                growingText += c;
            }
            lastC = c;
        }

        // set text
        textDrawable->setText(growingText);
    }
}

void Equation::createArrow()
{
    osg::Geode *geode = new osg::Geode();
    osg::Geometry *geometry = new osg::Geometry();
    geode->addDrawable(geometry);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    vertices->push_back(osg::Vec3(0.0f, 0.0f, 0.5f)); // line left top
    vertices->push_back(osg::Vec3(0.0f, 0.0f, -0.5f)); // line left bottom
    vertices->push_back(osg::Vec3(7.0f, 0.0f, -0.5f)); // line right botom
    vertices->push_back(osg::Vec3(7.0f, 0.0f, 0.5f)); // line right top
    vertices->push_back(osg::Vec3(11.0f, 0.0f, 0.0f)); // head front
    vertices->push_back(osg::Vec3(6.5f, 0.0f, 2.5f)); // head top
    vertices->push_back(osg::Vec3(6.5f, 0.0f, -2.5f)); // head bottom
    geometry->setVertexArray(vertices);

    osg::DrawElementsUInt *faces = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
    faces->push_back(0);
    faces->push_back(1);
    faces->push_back(3); // line top left
    faces->push_back(1);
    faces->push_back(2);
    faces->push_back(3); // line bottom right
    faces->push_back(2);
    faces->push_back(4);
    faces->push_back(3); // line right to head front
    faces->push_back(3);
    faces->push_back(4);
    faces->push_back(5); // head (upper part)
    faces->push_back(2);
    faces->push_back(6);
    faces->push_back(4); // head (lower part)
    geometry->addPrimitiveSet(faces);

    geode->getOrCreateStateSet()->setAttributeAndModes(material.get());

    arrowTransform = new osg::MatrixTransform();
    arrowTransform->addChild(geode);
}
