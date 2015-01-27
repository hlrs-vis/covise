/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiColoredBackground.h>

#include <OpenVRUI/opensg/SGVruiPresets.h>
#include <OpenVRUI/opensg/SGVruiTransformNode.h>

#include <OpenVRUI/coColoredBackground.h>

#include <OpenVRUI/util/vruiLog.h>

#include <OpenSG/OSGSimpleAttachments.h>

using namespace osg;
using namespace std;

osg::GeoNormals3fPtr SGVruiColoredBackground::normal = NullFC;

/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
*/
SGVruiColoredBackground::SGVruiColoredBackground(coColoredBackground *background)
    : SGVruiUIContainer(background)
{

    this->background = background;

    myDCS = 0;
    fancyDCS = NullFC;

    material = NullFC;
    highlightMaterial = NullFC;
    disabledMaterial = NullFC;
}

/** Destructor
 */
SGVruiColoredBackground::~SGVruiColoredBackground()
{
}

void SGVruiColoredBackground::resizeGeometry()
{

    createGeometry();

    float myHeight = background->getHeight();
    float myWidth = background->getWidth();

    //VRUILOG("SGVruiColoredBackground::resizeGeometry info: width = " << myWidth << ", height = " << myHeight)

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->setValue(Pnt3f(0.0f, myHeight, 0.0f), 3);
    coord->setValue(Pnt3f(myWidth, myHeight, 0.0f), 2);
    coord->setValue(Pnt3f(myWidth, 0.0f, 0.0f), 1);
    coord->setValue(Pnt3f(0.0f, 0.0f, 0.0f), 0);
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
}

/** create geometry elements shared by all OSGVruiColoredBackgrounds
 */
void SGVruiColoredBackground::createSharedLists()
{
    if (normal == NullFC)
    {
        normal = GeoNormals3f::create();
        beginEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        endEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);
    }
}

/** create the geometry node
 */
void SGVruiColoredBackground::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("SGVruiColoredBackground::createGeometry info: creating geometry")

    material = SGVruiPresets::getStateSet(background->getBackgroundMaterial());
    highlightMaterial = SGVruiPresets::getStateSet(background->getHighlightMaterial());
    disabledMaterial = SGVruiPresets::getStateSet(background->getDisableMaterial());

    // This DCS is necessary, otherwise the Backgroung will not stay the first
    // element in sorting order, which leads to artefacts.

    NodePtr transform = makeCoredNode<ComponentTransform>();

    myDCS = new SGVruiTransformNode(transform);
    fancyDCS = makeCoredNode<ComponentTransform>();

    beginEditCP(transform, Node::ChildrenFieldMask);
    transform->insertChild(0, fancyDCS);
    endEditCP(transform, Node::ChildrenFieldMask);

    createSharedLists();

    coord = GeoPositions3f::create();

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->addValue(Pnt3f(0.0f, 0.0f, 0.0f));
    coord->addValue(Pnt3f(60.0f, 0.0f, 0.0f));
    coord->addValue(Pnt3f(60.0f, 60.0f, 0.0f));
    coord->addValue(Pnt3f(0.0f, 60.0f, 0.0f));
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);

    geometryNode = makeCoredNode<Geometry>();
    geometry = GeometryPtr::dcast(geometryNode->getCore());

    GeoPLengthsPtr length = GeoPLengthsUI32::create();
    beginEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);
    length->addValue(4);
    endEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);

    GeoPTypesPtr types = GeoPTypesUI8::create();
    beginEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);
    types->addValue(GL_QUADS);
    endEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);

    beginEditCP(geometry);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setPositions(coord);
    geometry->setNormals(normal);
    geometry->setMaterial(material);
    endEditCP(geometry);

    osg::setName(geometryNode, "SGVruiColoredBackground::geode");

    beginEditCP(fancyDCS, Node::ChildrenFieldMask);
    fancyDCS->addChild(geometryNode);
    endEditCP(fancyDCS, Node::ChildrenFieldMask);

    resizeGeometry();
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
*/
void SGVruiColoredBackground::setEnabled(bool en)
{

    beginEditCP(geometry, Geometry::MaterialFieldMask);
    if (en)
    {
        if (background->isHighlighted())
        {
            geometry->setMaterial(highlightMaterial);
        }
        else
        {
            geometry->setMaterial(material);
        }
    }
    else
    {
        geometry->setMaterial(disabledMaterial);
    }
    endEditCP(geometry, Geometry::MaterialFieldMask);
}

void SGVruiColoredBackground::setHighlighted(bool hl)
{

    beginEditCP(geometry, Geometry::MaterialFieldMask);
    if (background->isEnabled())
    {
        if (hl)
        {
            geometry->setMaterial(highlightMaterial);
        }
        else
        {
            geometry->setMaterial(material);
        }
    }
    else
    {
        geometry->setMaterial(disabledMaterial);
    }
    endEditCP(geometry, Geometry::MaterialFieldMask);
}
