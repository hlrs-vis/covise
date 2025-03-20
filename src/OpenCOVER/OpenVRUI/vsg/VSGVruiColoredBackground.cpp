/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiColoredBackground.h>

#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/all.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{

/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
*/
VSGVruiColoredBackground::VSGVruiColoredBackground(coColoredBackground *background)
    : VSGVruiUIContainer(background)
{

    this->background = background;

    myDCS = 0;
    fancyDCS = 0;

}

/** Destructor
 */
VSGVruiColoredBackground::~VSGVruiColoredBackground()
{
}

void VSGVruiColoredBackground::resizeGeometry()
{

    createGeometry();

    float myHeight = background->getHeight();
    float myWidth = background->getWidth();

    (*coord)[3].set(0.0f, myHeight, 0.0f);
    (*coord)[2].set(myWidth, myHeight, 0.0f);
    (*coord)[1].set(myWidth, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);
    //vruiRendererInterface::the()->addToTransfer(coord);
}

/** create geometry elements shared by all VSGVruiColoredBackgrounds
 */
void VSGVruiColoredBackground::createSharedLists()
{
    normal = vsg::vec3Value::create(vsg::vec3(0.0f, 0.0f, 1.0f));
    config->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_INSTANCE, normal);
}

/** create the geometry node
 */
void VSGVruiColoredBackground::createGeometry()
{

    if (myDCS)
        return;

    ref_ptr<MatrixTransform> transform = MatrixTransform::create();

    myDCS = new VSGVruiTransformNode(transform);
    fancyDCS = new MatrixTransform();

    transform->addChild(fancyDCS);

    material = vsg::PhongMaterialValue::create(VSGVruiPresets::instance()->materials[coUIElement::GREY]);

    config = vsg::GraphicsPipelineConfigurator::create(VSGVruiPresets::instance()->getOrCreatePhongShaderSet());
    config->descriptorConfigurator = vsg::DescriptorConfigurator::create();
    config->descriptorConfigurator->shaderSet = VSGVruiPresets::instance()->getOrCreatePhongShaderSet();
    auto options = VSGVruiPresets::instance()->options;
    if (options) config->assignInheritedState(options->inheritedState);


    auto indices = vsg::ushortArray::create(6);

    coord = new vec3Array(4);

    (*coord)[3].set(0.0f, 60.0f, 0.0f);
    (*coord)[2].set(60.0f, 60.0f, 0.0f);
    (*coord)[1].set(60.0f, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);
    config->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord);


    createSharedLists(); // create normals



    auto vid = vsg::VertexIndexDraw::create();
    vid->assignArrays(vertexArrays);
    vid->assignIndices(indices);
    vid->indexCount = static_cast<uint32_t>(indices->valueCount());
    vid->instanceCount = 1;
    //if (!name.empty()) vid->setValue("name", name);

    bool blending = false;
    bool two_sided = false;
    // set the GraphicsPipelineStates to the required values.
    struct SetPipelineStates : public vsg::Visitor
    {
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        bool blending = false;
        bool two_sided = false;

        SetPipelineStates(VkPrimitiveTopology in_topology, bool in_blending, bool in_two_sided) :
            topology(in_topology), blending(in_blending), two_sided(in_two_sided) {
        }

        void apply(vsg::Object& object) { object.traverse(*this); }
        void apply(vsg::RasterizationState& rs)
        {
            if (two_sided) rs.cullMode = VK_CULL_MODE_NONE;
        }
        void apply(vsg::InputAssemblyState& ias) { ias.topology = topology; }
        void apply(vsg::ColorBlendState& cbs) { cbs.configureAttachments(blending); }
    } sps(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, blending, two_sided);

    config->accept(sps);
    if (options)
    {
        VSGVruiPresets::instance()->sharedObjects=options->sharedObjects;
    }
    if (VSGVruiPresets::instance()->sharedObjects)
    {
        VSGVruiPresets::instance()->sharedObjects->share(config, [](auto gpc) { gpc->init(); });
    }
    else
    {
        config->init();
    }

    // create StateGroup as the root of the scene/command graph to hold the GraphicsPipeline, and binding of Descriptors to decorate the whole graph
    auto stateGroup = vsg::StateGroup::create();

    config->copyTo(stateGroup, VSGVruiPresets::instance()->sharedObjects);

    stateGroup->addChild(vid);
    /*
    if (material->blending)
    {
        vsg::ComputeBounds computeBounds;
        vid->accept(computeBounds);
        vsg::dvec3 center = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
        double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.5;

        auto depthSorted = vsg::DepthSorted::create();
        depthSorted->binNumber = 10;
        depthSorted->bound.set(center[0], center[1], center[2], radius);
        depthSorted->child = stateGroup;

        node = depthSorted;
    }
    else*/
    {
        node = stateGroup;
    }

    fancyDCS->addChild(node);

    resizeGeometry();
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
*/
void VSGVruiColoredBackground::setEnabled(bool en)
{
    if (en)
    {
        if (background->isHighlighted())
        {
            material->value()= VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_HIGHLIGHTED];
        }
        else
        {
            material->value() = VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_NORMAL];
        }
    }
    else
    {
        material->value() = VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_DISABLED];
    }
    //vruiRendererInterface::the()->addToTransfer(material);
}

void VSGVruiColoredBackground::setHighlighted(bool hl)
{
    if (background->isEnabled())
    {
        if (hl)
        {
            material->value() = VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_HIGHLIGHTED];
        }
        else
        {
            material->value() = VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_NORMAL];
        }
    }
    else
    {
        material->value() = VSGVruiPresets::instance()->materials[coUIElement::ITEM_BACKGROUND_DISABLED];
    }
    //vruiRendererInterface::the()->addToTransfer(material);
}
}
