/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coTextureRectBackground.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

namespace vrui
{

/** Action listener called when the action button was just clicked.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
 */
void coTextureRectBackgroundActor::texturePointerClicked(coTextureRectBackground *, float, float)
{
}

/** Action listener called when the action button was just released.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
 */
void coTextureRectBackgroundActor::texturePointerReleased(coTextureRectBackground *, float, float)
{
}

/** Action listener called when the pointer is moved with the action button down.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
 */
void coTextureRectBackgroundActor::texturePointerDragged(coTextureRectBackground *, float, float)
{
}

/** Action listener called when the pointer is moved with no button pressed.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
 */
void coTextureRectBackgroundActor::texturePointerMoved(coTextureRectBackground *, float, float)
{
}

/** Action listener called when the pointer left the textured background.
  @param tb   textured background object which triggered the event
*/
void coTextureRectBackgroundActor::texturePointerLeft(coTextureRectBackground *)
{
}

/** Constructor
  @param normalTexture normal texture name
  @param highlightTexture highlighted texture name
  @param disableTexture disabled texture name
 */
coTextureRectBackground::coTextureRectBackground(const std::string &normalTexture,
                                                 coTextureRectBackgroundActor *actor)
{
    myActor = actor;
    normalTexName = normalTexture;
    repeat = false;
    updated = false;
    texXSize = 0;
    texYSize = 0;
    currentTextures = 0;

    if (myActor != NULL)
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
}

/** Constructor
  @param normalImage texel array form normal appearance
  @param highlightImage texel array for highlighted appearance
  @param disableImage texel array for disabled appearance
 */
coTextureRectBackground::coTextureRectBackground(uint *normalImage,
                                                 int comp, int ns, int nt, int nr,
                                                 coTextureRectBackgroundActor *actor)
{
    myActor = actor;
    normalTexName = "";
    repeat = false;
    texXSize = 0;
    texYSize = 0;

    currentTextures = new TextureSet(normalImage, comp, ns, nt, nr);

    if (myActor != NULL)
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
}

/** Destructor
 */
coTextureRectBackground::~coTextureRectBackground()
{
    delete currentTextures;
}

void coTextureRectBackground::setImage(uint *normalImage,
                                       int comp, int ns, int nt, int nr)
{
    delete currentTextures;
    currentTextures = new TextureSet(normalImage, comp, ns, nt, nr);
    uiElementProvider->update();
}

/** Hit is called whenever this texture is intersected.
  @return ACTION_CALL_ON_MISS if you want miss to be called
          otherwise return ACTION_DONE
 */
int coTextureRectBackground::hit(vruiHit *hit)
{

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    static bool prevButtonPressed = false;

    float x = -1.0f;
    float y = -1.0f; // pointer coordiantes on textured background [0..1]

    bool isButtonPressed; // true = action button pressed

    if (myActor)
    {
        // Compute pointer location in textured background coordinates:
        coVector point = hit->getLocalIntersectionPoint();
        x = (float)point[0] / myWidth;
        y = (float)point[1] / myHeight;

        // Get button status:
        isButtonPressed = ((hit->isMouseHit() ? vruiRendererInterface::the()->getMouseButtons() : vruiRendererInterface::the()->getButtons())->getStatus() & vruiButtons::ACTION_BUTTON);

        // Always send pointer coordinates:

        if (isButtonPressed && !prevButtonPressed) // button just pressed?
        {
            myActor->texturePointerClicked(this, x, y);
        }
        // button just released?
        else if (!isButtonPressed && prevButtonPressed)
        {
            myActor->texturePointerReleased(this, x, y);
        }
        else if (isButtonPressed)
        {
            myActor->texturePointerDragged(this, x, y);
        }
        else
        {
            myActor->texturePointerMoved(this, x, y);
        }

        prevButtonPressed = isButtonPressed;
    }
    return ACTION_CALL_ON_MISS;
}

void coTextureRectBackground::miss()
{
    if (myActor)
    {
        myActor->texturePointerLeft(this);
    }
}

/** set the size of the texture
  @param x width 0 means fit to Background
  @param y height 0 means fit to Background
 */
void coTextureRectBackground::setTexSize(float x, float y)
{
    texXSize = x;
    texYSize = y;
    uiElementProvider->update();
}

/** set repeating of the texture
 @param r if set to true, texture will be repeated, otherwise it is clamped, default is false
 */
void coTextureRectBackground::setRepeat(bool r)
{
    repeat = r;
    uiElementProvider->update();
}

bool coTextureRectBackground::getRepeat() const
{
    return repeat;
}

/** Set activation state of this background and all its children.
  if this background is disabled, the texture is always the
  disabled texture, regardless of the highlighted state
  @param en true = elements enabled
 */
void coTextureRectBackground::setEnabled(bool en)
{
    coUIContainer::setEnabled(en);
    uiElementProvider->setEnabled(en);
}

void coTextureRectBackground::setHighlighted(bool hl)
{
    coUIContainer::setHighlighted(hl);
    uiElementProvider->setHighlighted(hl);
}

const char *coTextureRectBackground::getClassName() const
{
    return "coTextureRectBackground";
}

bool coTextureRectBackground::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coBackground::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
