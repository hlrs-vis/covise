/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coTexturedBackground.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

#include <list>
using std::list;

namespace vrui
{

/** Action listener called when the action button was just clicked.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
*/
void coTexturedBackgroundActor::texturePointerClicked(coTexturedBackground *, float, float)
{
}

/** Action listener called when the action button was just released.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
*/
void coTexturedBackgroundActor::texturePointerReleased(coTexturedBackground *, float, float)
{
}

/** Action listener called when the pointer is moved with the action button down.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
*/
void coTexturedBackgroundActor::texturePointerDragged(coTexturedBackground *, float, float)
{
}

/** Action listener called when the pointer is moved with no button pressed.
  @param tb   textured background object which triggered the event
  @param x,y  hit location on textured background when button was pressed [0..1]
*/
void coTexturedBackgroundActor::texturePointerMoved(coTexturedBackground *, float, float)
{
}

/** Action listener called when the pointer left the textured background.
  @param tb   textured background object which triggered the event
*/
void coTexturedBackgroundActor::texturePointerLeft(coTexturedBackground *)
{
}

/** Constructor
  @param normalTexture normal texture name
  @param highlightTexture highlighted texture name
  @param disableTexture disabled texture name
*/
coTexturedBackground::coTexturedBackground(const std::string &normalTexture,
                                           const std::string &highlightTexture,
                                           const std::string &disabledTexture,
                                           coTexturedBackgroundActor *actor)
{
    myActor = actor;
    normalTexName = normalTexture;
    highlightTexName = highlightTexture;
    disabledTexName = disabledTexture;
    repeat = false;
    updated = false; // Pinkowski, 20.09.2007
    texXSize = 0;
    texYSize = 0;
    currentTextures = 0;
    active_ = true;
    myScale = 1.0;

    if (myActor != NULL)
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
}

/** Constructor
  @param normalImage texel array form normal appearance
  @param highlightImage texel array for highlighted appearance
  @param disableImage texel array for disabled appearance
*/
coTexturedBackground::coTexturedBackground(uint *normalImage,
                                           uint *highlightImage,
                                           uint *disabledImage,
                                           int comp, int ns, int nt, int nr,
                                           coTexturedBackgroundActor *actor)
{
    myActor = actor;
    normalTexName = "";
    highlightTexName = "";
    disabledTexName = "";
    repeat = false;
    texXSize = 0;
    texYSize = 0;
    myScale = 1.0;

    currentTextures = new TextureSet(normalImage, highlightImage, disabledImage, comp, ns, nt, nr);

    if (myActor != NULL)
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
}

/** Destructor
 */
coTexturedBackground::~coTexturedBackground()
{
    delete currentTextures;
}

void coTexturedBackground::setImage(uint *normalImage,
                                    uint *highlightImage,
                                    uint *disabledImage,
                                    int comp, int ns, int nt, int nr)
{
    delete currentTextures;
    currentTextures = new TextureSet(normalImage, highlightImage, disabledImage, comp, ns, nt, nr);
    updated = true;
    uiElementProvider->update();
}

/** Hit is called whenever this texture is intersected.
  @return ACTION_CALL_ON_MISS if you want miss to be called
          otherwise return ACTION_DONE
*/
int coTexturedBackground::hit(vruiHit *hit)
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

void coTexturedBackground::miss()
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
void coTexturedBackground::setTexSize(float x, float y)
{
    texXSize = x;
    texYSize = y;
    uiElementProvider->update();
}

/** set repeating of the texture
 @param r if set to true, texture will be repeated, otherwise it is clamped, default is false
*/
void coTexturedBackground::setRepeat(bool r)
{
    repeat = r;
    uiElementProvider->update();
}

bool coTexturedBackground::getRepeat() const
{
    return repeat;
}

void coTexturedBackground::setScale(float s)
{
    myScale = s;
    uiElementProvider->update();
}

float coTexturedBackground::getScale()
{
    return myScale;
}

/** Set activation state of this background and all its children.
  if this background is disabled, the texture is always the
  disabled texture, regardless of the highlighted state
  @param en true = elements enabled
*/
void coTexturedBackground::setEnabled(bool en)
{
    coUIContainer::setEnabled(en);
    uiElementProvider->setEnabled(en);
}

void coTexturedBackground::setHighlighted(bool hl)
{
    coUIContainer::setHighlighted(hl);
    uiElementProvider->setHighlighted(hl);
}

//set if item is active
void coTexturedBackground::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
        setEnabled(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        setEnabled(a);
    }
    active_ = a;
}

const char *coTexturedBackground::getClassName() const
{
    return "coTexturedBackground";
}

bool coTexturedBackground::isOfClassName(const char *classname) const
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

void coTexturedBackground::shrinkToMin()
{

    float mw = 0;
    float mh = 0;
    float md = 0;

    // walk through elements

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        // shrink them
        (*i)->shrinkToMin();

        // get max
        if ((*i)->getWidth() > mw)
        {
            mw = (*i)->getWidth();
        }

        if ((*i)->getHeight() > mh)
        {
            mh = (*i)->getHeight();
        }

        if ((*i)->getDepth() > md)
        {
            md = (*i)->getDepth();
        }
    }

    //fprintf(stderr,"coBackground(%x)::shrinkToMin mw=%f mh=%f\n", this, mw, mh);

    if ((mw == 0) && (mh == 0) && (md == 0)) // if there are no elements, set max of minSize and realSize
    {
        myWidth = coMax(minWidth, myWidth);
        myHeight = coMax(minHeight, myHeight);
        myDepth = coMax(minDepth, myDepth);
    }
    else
    {
        // set max of minSize and elementsSize
        myWidth = coMax(minWidth, mw);
        myHeight = coMax(minHeight, mh);
        myDepth = coMax(minDepth, md);
    }
    resizeGeometry();
}
}
