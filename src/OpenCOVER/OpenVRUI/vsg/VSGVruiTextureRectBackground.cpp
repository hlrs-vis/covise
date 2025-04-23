/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiTextureRectBackground.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/coTextureRectBackground.h>


#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{


    vsg::ref_ptr<vsg::vec3Array> VSGVruiTextureRectBackground::normal;
/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
 */
VSGVruiTextureRectBackground::VSGVruiTextureRectBackground(coTextureRectBackground *background)
    : VSGVruiUIContainer(background)
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
VSGVruiTextureRectBackground::~VSGVruiTextureRectBackground()
{
}

/*!
 * removed REPEAT, since rectangular textures do not support repeat!
 *
 */
void VSGVruiTextureRectBackground::update()
{

   /* geometryNode->dirtyBound();
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

        ref_ptr<TexEnv> texEnv = VSGVruiPresets::getTexEnvModulate();

        state->setTextureAttributeAndModes(0, texNormal.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        state->setTextureAttributeAndModes(0, texEnv.get(), StateAttribute::ON | StateAttribute::PROTECTED);

        texNormal->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texNormal->setWrap(Texture::WRAP_T, Texture::CLAMP);
    }

    if (texXSize != background->getTexXSize() || texYSize != background->getTexYSize())
        rescaleTexture();*/
}

void VSGVruiTextureRectBackground::resizeGeometry()
{

    createGeometry();

   /* float myHeight = background->getHeight();
    float myWidth = background->getWidth();

    (*coord)[3].set(0.0f, myHeight, 0.0f);
    (*coord)[2].set(myWidth, myHeight, 0.0f);
    (*coord)[1].set(myWidth, 0.0f, 0.0f);
    (*coord)[0].set(0.0f, 0.0f, 0.0f);

    rescaleTexture();

	coord->dirty();
    geometryNode->dirtyBound();
    geometry->dirtyBound();
    geometry->dirtyDisplayList();*/
}

/*!
 * rescales the texture; Texture coordinates are non-normalized, therefore
 * they are set to ( 0, w, h, 0 )
 * see http://www.opengl.org/registry/specs/ARB/texture_rectangle.txt
 */
void VSGVruiTextureRectBackground::rescaleTexture()
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

/** create geometry elements shared by all VSGVruiTextureRectBackgrounds
 */
void VSGVruiTextureRectBackground::createSharedLists()
{
    if (normal.get() == nullptr)
    {
        normal = new vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }
}

/** create the geometry
 */
void VSGVruiTextureRectBackground::createGeometry()
{

    if (myDCS)
        return;

    myDCS = new VSGVruiTransformNode(MatrixTransform::create());

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

    vsg::Group* group= dynamic_cast<vsg::Group*>(myDCS->getNodePtr());
    //group->addChild(geometryNode.get());
}

/** create texture from files
 */
void VSGVruiTextureRectBackground::createTexturesFromFiles()
{
#if 0
   VSGVruiTexture * oTex = dynamic_cast<VSGVruiTexture*>(vruiRendererInterface::the()->createTexture(background->getNormalTexName()));
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
      VRUILOG("VSGVruiTransformNode::createTexturesFromFiles err: normal texture image '"
            << background->getNormalTexName() << "' not loaded")
   }
#endif
    VRUILOG("VSGVruiTextureRectBackground::createTexturesFromFiles(): Functionality not available!")
}

/** create texture from arrays
 */
void VSGVruiTextureRectBackground::createTexturesFromArrays(uint *normalImage,
                                                            int comp, int ns, int nt, int nr)
{

  /*  texNormal = new TextureRectangle();

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
        VRUILOG("VSGVruiTransformNode::createTexturesFromArrays err: normal texture image creation error")
    }

    texHighlighted = texNormal;
    texDisabled = texNormal;*/
}

/** create geometry and texture objects
 */
void VSGVruiTextureRectBackground::createGeode()
{

    createSharedLists();

    coord = new vec3Array(4);

    (*coord)[0].set(0, 0, 0);
    (*coord)[1].set(60, 0, 0);
    (*coord)[2].set(60, 60, 0);
    (*coord)[3].set(0, 60, 0);

    texcoord = new vec2Array(4);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(1.0, 0.0);
    (*texcoord)[2].set(1.0, 1.0);
    (*texcoord)[3].set(0.0, 1.0);

   
    resizeGeometry();
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
 */
void VSGVruiTextureRectBackground::setEnabled(bool en)
{

   /* if (en)
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
    }*/
}

void VSGVruiTextureRectBackground::setHighlighted(bool hl)
{
   /* if (background->isEnabled())
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
    }*/
}
}
