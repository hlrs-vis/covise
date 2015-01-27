/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiColoredBackground.h>

#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <OpenVRUI/coColoredBackground.h>

#include <osg/Texture>
#include <osg/Material>

#include <OpenVRUI/util/vruiLog.h>

using namespace osg;
using namespace std;

namespace vrui
{

ref_ptr<Vec3Array> OSGVruiColoredBackground::normal = 0;

/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
*/
OSGVruiColoredBackground::OSGVruiColoredBackground(coColoredBackground *background)
    : OSGVruiUIContainer(background)
{

    this->background = background;

    myDCS = 0;
    fancyDCS = 0;

    state = 0;
    highlightState = 0;
    disabledState = 0;
}

/** Destructor
 */
OSGVruiColoredBackground::~OSGVruiColoredBackground()
{
}

void OSGVruiColoredBackground::resizeGeometry()
{

    createGeometry();

    float myHeight = background->getHeight();
    float myWidth = background->getWidth();

    (*coord)[3].set(0.0f, myHeight, 0.0f);
    (*coord)[2].set(myWidth, myHeight, 0.0f);
    (*coord)[1].set(myWidth, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);

    geometryNode->dirtyBound();
    geometry->dirtyBound();
    geometry->dirtyDisplayList();
}

/** create geometry elements shared by all OSGVruiColoredBackgrounds
 */
void OSGVruiColoredBackground::createSharedLists()
{
    if (normal == 0)
    {
        normal = new Vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }
}

/** create the geometry node
 */
void OSGVruiColoredBackground::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("OSGVruiColoredBackground::createGeometry info: creating geometry")

    state = OSGVruiPresets::getStateSet(background->getBackgroundMaterial());
    highlightState = OSGVruiPresets::getStateSet(background->getHighlightMaterial());
    disabledState = OSGVruiPresets::getStateSet(background->getDisableMaterial());

    // This DCS is necessary, otherwise the Backgroung will not stay the first
    // element in sorting order, which leads to artefacts.

    ref_ptr<MatrixTransform> transform = new MatrixTransform();

    myDCS = new OSGVruiTransformNode(transform.get());
    fancyDCS = new MatrixTransform();

    transform->insertChild(0, fancyDCS.get());

    createSharedLists();

    coord = new Vec3Array(4);

    (*coord)[3].set(0.0f, 60.0f, 0.0f);
    (*coord)[2].set(60.0f, 60.0f, 0.0f);
    (*coord)[1].set(60.0f, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);

    geometry = new Geometry();

    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);

    geometryNode = new Geode();
    geometryNode->setStateSet(state.get());
    geometryNode->addDrawable(geometry.get());
    geometryNode->setName("OSGVruiColoredBackground::geode");

    fancyDCS->addChild(geometryNode.get());

    resizeGeometry();
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
*/
void OSGVruiColoredBackground::setEnabled(bool en)
{

    if (en)
    {
        if (background->isHighlighted())
        {
            geometryNode->setStateSet(highlightState.get());
        }
        else
        {
            geometryNode->setStateSet(state.get());
        }
    }
    else
    {
        geometryNode->setStateSet(disabledState.get());
    }
}

void OSGVruiColoredBackground::setHighlighted(bool hl)
{
    if (background->isEnabled())
    {
        if (hl)
        {
            geometryNode->setStateSet(highlightState.get());
        }
        else
        {
            geometryNode->setStateSet(state.get());
        }
    }
    else
    {
        geometryNode->setStateSet(disabledState.get());
    }
}
}
