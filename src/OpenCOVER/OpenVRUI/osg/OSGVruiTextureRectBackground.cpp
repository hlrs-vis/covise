/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiTextureRectBackground.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/coTextureRectBackground.h>

#include <osg/Image>
#include <osg/Material>

#include <OpenVRUI/util/vruiLog.h>

using namespace osg;
using namespace std;

namespace vrui
{

ref_ptr<Vec3Array> OSGVruiTextureRectBackground::normal = 0;

/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
 */
OSGVruiTextureRectBackground::OSGVruiTextureRectBackground(coTextureRectBackground *background)
    : OSGVruiUIContainer(background)
{

    this->background = background;

    state = 0;
    highlightState = 0;
    disabledState = 0;
    repeat = background->getRepeat();
    texXSize = background->getTexXSize();
    texYSize = background->getTexYSize();
}

/** Destructor
 */
OSGVruiTextureRectBackground::~OSGVruiTextureRectBackground()
{
}

/*!
 * removed REPEAT, since rectangular textures do not support repeat!
 *
 */
void OSGVruiTextureRectBackground::update()
{

    geometryNode->dirtyBound();
    geometry->dirtyBound();
    geometry->dirtyDisplayList();
    if (tex != background->getCurrentTextures() || background->getUpdated())
    {
        background->setUpdated(false);

        tex = background->getCurrentTextures();

        Image *image = texNormal->getImage();

        GLenum pixel_format = tex->comp == 3 ? GL_RGB : GL_RGBA;

        image->setImage(tex->s, tex->t, tex->r, pixel_format, pixel_format, GL_UNSIGNED_BYTE,
                        (unsigned char *)tex->normalTextureImage, Image::NO_DELETE, tex->comp);
        image->dirty();
        texNormal->dirtyTextureObject();

        ref_ptr<TexEnv> texEnv = OSGVruiPresets::getTexEnvModulate();

        state->setTextureAttributeAndModes(0, texNormal.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        state->setTextureAttributeAndModes(0, texEnv.get(), StateAttribute::ON | StateAttribute::PROTECTED);

        texNormal->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texNormal->setWrap(Texture::WRAP_T, Texture::CLAMP);
    }

    if (texXSize != background->getTexXSize() || texYSize != background->getTexYSize())
        rescaleTexture();
}

void OSGVruiTextureRectBackground::resizeGeometry()
{

    createGeometry();

    float myHeight = background->getHeight();
    float myWidth = background->getWidth();

    (*coord)[3].set(0.0f, myHeight, 0.0f);
    (*coord)[2].set(myWidth, myHeight, 0.0f);
    (*coord)[1].set(myWidth, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);

    rescaleTexture();

	coord->dirty();
    geometryNode->dirtyBound();
    geometry->dirtyBound();
    geometry->dirtyDisplayList();
}

/*!
 * rescales the texture; Texture coordinates are non-normalized, therefore
 * they are set to ( 0, w, h, 0 )
 * see http://www.opengl.org/registry/specs/ARB/texture_rectangle.txt
 */
void OSGVruiTextureRectBackground::rescaleTexture()
{
    float xmin = 0.0f;
    float xmax = background->getTexXSize();
    float ymin = background->getTexYSize();
    float ymax = 0.0f;

    (*texcoord)[0].set(xmin, ymin);
    (*texcoord)[1].set(xmax, ymin);
    (*texcoord)[2].set(xmax, ymax);
    (*texcoord)[3].set(xmin, ymax);
}

/** create geometry elements shared by all OSGVruiTextureRectBackgrounds
 */
void OSGVruiTextureRectBackground::createSharedLists()
{
    if (normal == 0)
    {
        normal = new Vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }
}

/** create the geometry
 */
void OSGVruiTextureRectBackground::createGeometry()
{

    if (myDCS)
        return;

    myDCS = new OSGVruiTransformNode(new MatrixTransform());

    tex = background->getCurrentTextures();

    if (tex)
    {
        createTexturesFromArrays(tex->normalTextureImage,
                                 tex->comp, tex->s, tex->t, tex->r);
    }
    else
    {
        createTexturesFromFiles();
    }
    createGeode();
    if (tex->comp == 4) // texture with alpha
    { //FIXME
        //     state->setMode(PFSTATE_TRANSPARENCY, PFTR_BLEND_ALPHA);
        //     highlightState->setMode(PFSTATE_TRANSPARENCY, PFTR_BLEND_ALPHA);
        //     disabledState->setMode(PFSTATE_TRANSPARENCY, PFTR_BLEND_ALPHA);
    }

    myDCS->getNodePtr()->asGroup()->addChild(geometryNode.get());
}

/** create texture from files
 */
void OSGVruiTextureRectBackground::createTexturesFromFiles()
{
#if 0
   OSGVruiTexture * oTex = dynamic_cast<OSGVruiTexture*>(vruiRendererInterface::the()->createTexture(background->getNormalTexName()));
   texNormal = oTex->getTexture();
   vruiRendererInterface::the()->deleteTexture(oTex);

   if(texNormal.valid())
   {
      texNormal->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
      texNormal->setWrap(Texture::WRAP_S, Texture::CLAMP);
      texNormal->setWrap(Texture::WRAP_T, Texture::CLAMP);
   }
   else
   {
      VRUILOG("OSGVruiTransformNode::createTexturesFromFiles err: normal texture image '"
            << background->getNormalTexName() << "' not loaded")
   }
#endif
    VRUILOG("OSGVruiTextureRectBackground::createTexturesFromFiles(): Functionality not available!")
}

/** create texture from arrays
 */
void OSGVruiTextureRectBackground::createTexturesFromArrays(uint *normalImage,
                                                            int comp, int ns, int nt, int nr)
{

    texNormal = new TextureRectangle();

    GLenum pixel_format = comp == 3 ? GL_RGB : GL_RGBA;

    if (texNormal.valid())
    {
        ref_ptr<Image> image = new Image();

        image->setImage(ns, nt, nr, pixel_format, pixel_format, GL_UNSIGNED_BYTE,
                        (unsigned char *)normalImage, Image::NO_DELETE, comp);
        texNormal->setImage(image.get());
        texNormal->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texNormal->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texNormal->setWrap(Texture::WRAP_T, Texture::CLAMP);
    }
    else
    {
        VRUILOG("OSGVruiTransformNode::createTexturesFromArrays err: normal texture image creation error")
    }

    texHighlighted = texNormal;
    texDisabled = texNormal;
}

/** create geometry and texture objects
 */
void OSGVruiTextureRectBackground::createGeode()
{

    createSharedLists();

    coord = new Vec3Array(4);

    (*coord)[0].set(0, 0, 0);
    (*coord)[1].set(60, 0, 0);
    (*coord)[2].set(60, 60, 0);
    (*coord)[3].set(0, 60, 0);

    texcoord = new Vec2Array(4);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(1.0, 0.0);
    (*texcoord)[2].set(1.0, 1.0);
    (*texcoord)[3].set(0.0, 1.0);

    state = OSGVruiPresets::makeStateSet(coUIElement::WHITE);
    highlightState = OSGVruiPresets::makeStateSet(coUIElement::WHITE);
    disabledState = OSGVruiPresets::makeStateSet(coUIElement::WHITE);

    state->setTextureAttributeAndModes(0, texNormal.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    state->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), StateAttribute::ON | StateAttribute::PROTECTED);

    highlightState->setTextureAttributeAndModes(0, texHighlighted.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    highlightState->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), StateAttribute::ON | StateAttribute::PROTECTED);

    disabledState->setTextureAttributeAndModes(0, texDisabled.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    disabledState->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), StateAttribute::ON | StateAttribute::PROTECTED);

    state->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    highlightState->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    disabledState->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    geometry = new Geometry();
    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texcoord.get());
    geometryNode = new Geode();
    geometryNode->setStateSet(state.get());
    geometryNode->addDrawable(geometry.get());
    resizeGeometry();
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
 */
void OSGVruiTextureRectBackground::setEnabled(bool en)
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

void OSGVruiTextureRectBackground::setHighlighted(bool hl)
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
