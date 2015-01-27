/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coMenuContainer.h>

using std::list;

namespace vrui
{

/// Constructor
coMenuContainer::coMenuContainer(Orientation orientation)
    : coRowContainer(orientation)
{
    numAlignedMin = 2;
}

/// Destructor
coMenuContainer::~coMenuContainer()
{
}

/// Change number of elements that are aligned MIN
void coMenuContainer::setNumAlignedMin(int n)
{
    numAlignedMin = n;
    childResized();
}

void coMenuContainer::resizeToParent(float newWidth, float newHeight, float, bool shrink)
{

    list<coUIElement *>::iterator element;
    float mw = 0.0f;
    float mh = 0.0f;
    float md = 0.0f;

    float sw = 0.0f;
    float sh = 0.0f;
    float sd = 0.0f;

    float ch = 0.0f;
    float cw = 0.0f;

    float old_sw = 0.0f;
    float old_sh = 0.0f;
    float old_sd = 0.0f;

    float el_w, el_h;

    bool geometryChanged;

    if (shrink)
    {
        shrinkToMin();
    }

    // get bounding box and maximum width and height
    sh = Vgap;
    sw = Hgap;

    // catch empty container size

    if (elements.empty())
    {
        sh += Vgap;
        sw += Hgap;
    }

    for (element = elements.begin(); element != elements.end(); ++element)
    {

        if ((*element)->getWidth() > mw)
        {
            mw = (*element)->getWidth();
        }

        if ((*element)->getHeight() > mh)
        {
            mh = (*element)->getHeight();
        }

        if ((*element)->getDepth() > md)
        {
            md = (*element)->getDepth();
        }

        sh += (*element)->getHeight();
        sh += Vgap;

        sw += (*element)->getWidth();
        sw += Hgap;
    }

    if (orientation == HORIZONTAL)
    {
        sw = coMax(sw, newWidth);
        sh = coMax(mh + 2 * Vgap, newHeight);
        sd = md + 2 * Dgap;
    }
    else if (orientation == VERTICAL)
    {
        sh = coMax(sh, newHeight);
        sw = coMax(mw + 2 * Hgap, newWidth);
        sd = md + 2 * Dgap;
    }

    // 2nd loop: resize elements

    geometryChanged = true;

    for (int ctr = 0; (ctr < 3) && geometryChanged; ++ctr)
    {
        old_sw = sw;
        old_sh = sh;
        old_sd = sd;

        // use cw,ch as bounding box
        cw = Hgap;
        ch = Vgap;

        // catch empty container size
        if (elements.empty())
        {
            ch += Vgap;
            cw += Hgap;
        }

        for (element = elements.begin(); element != elements.end(); ++element)
        {
            if (orientation == HORIZONTAL)
            {
                // die links ausgerichteten elemente
                (*element)->resizeToParent((*element)->getWidth(), sh - 2 * Vgap, md, false);
            }
            else if (orientation == VERTICAL)
            {
                // die links ausgerichteten elemente
                (*element)->resizeToParent(sw - 2 * Hgap, (*element)->getHeight(), md, false);
            }

            if ((*element)->getWidth() > mw)
            {
                mw = (*element)->getWidth();
            }

            if ((*element)->getHeight() > mh)
            {
                mh = (*element)->getHeight();
            }

            cw += (*element)->getWidth();
            cw += Hgap;

            ch += (*element)->getHeight();
            ch += Vgap;
        }

        // copy element width and height
        el_w = cw;
        el_h = ch;

        // adjust sw,sh to new numbers
        if (orientation == HORIZONTAL)
        {
            sw = coMax(cw, newWidth);
            sh = coMax(mh + 2 * Vgap, sh);
        }
        else if (orientation == VERTICAL)
        {
            sh = coMax(ch, newHeight);
            sw = coMax(mw + 2 * Hgap, sw);
        }

        // track changes
        geometryChanged = (sh != old_sh) || (sw != old_sw) || (sd != old_sd);
    }

    // 3rd loop: position elements
    int numElements = (int)elements.size();
    int elemNum = 0;

    ch = sh;
    cw = Hgap;

    // take care about the special situation of numAlignedMin=0
    // the first loop is skipped automatically.
    // the second should go and use 'alignment' for placement.

    for (element = elements.begin();
         (element != elements.end()) && (elemNum < numAlignedMin);
         ++element)
    {

        if (orientation == HORIZONTAL)
        {

            float h;

            switch (alignment)
            {
            case CENTER:
                h = (sh - Vgap - (*element)->getHeight()) / 2.0f;
                break;

            case MAX:
                h = sh - (*element)->getHeight();
                break;

            default:
                h = Vgap;
                break;
            }

            (*element)->setPos(cw, h, 0);

            cw += (*element)->getWidth();
            cw += Hgap;
        }
        else if (orientation == VERTICAL)
        {
            ch -= (*element)->getHeight();
            ch -= Vgap;

            float w;
            switch (alignment)
            {
            case CENTER:
                w = (sw - Hgap - (*element)->getWidth()) / 2.0f;
                break;

            case MAX:
                w = sw - (*element)->getWidth();
                break;

            default:
                w = Hgap;
                break;
            }

            (*element)->setPos(w, ch, 0);
        }

        ++elemNum;
    }

    // skip first elements
    element = elements.begin();
    for (int ctr = 0; ctr < numElements - 1; ++ctr)
        ++element;

    elemNum = numElements - 1;

    // take care about numAlignedMin=0

    // normal operation
    ch = Vgap;
    cw = sw; // alle >=numAlignedMin von rechts/unten

    if (numAlignedMin == 0)
    {
        if (orientation == HORIZONTAL)
        {
            // adjust x start
            cw = sw - ((sw - el_w) / 2.0f);
        }
        else if (orientation == VERTICAL)
        {
            // adjust y start
            ch = (sh - el_h) / 2.0f + Vgap;
        }
    }

    element = elements.end();

    while ((element != elements.begin()) && (elemNum >= numAlignedMin))
    {
        --element; // end contains no value...
        if (orientation == HORIZONTAL)
        {
            cw -= (*element)->getWidth();
            cw -= Hgap;

            float h;
            switch (alignment)
            {
            case CENTER:
                h = (sh - Vgap - (*element)->getHeight()) / 2.0f;
                break;

            case MAX:
                h = sh - (*element)->getHeight();
                break;

            default:
                h = Vgap;
                break;
            }

            (*element)->setPos(cw, h, 0);
        }
        else if (orientation == VERTICAL)
        {
            float w;
            switch (alignment)
            {
            case CENTER:
                w = (sw - Hgap - (*element)->getWidth()) / 2.0f;
                break;

            case MAX:
                w = sw - (*element)->getWidth();
                break;

            default:
                w = Hgap;
                break;
            }

            (*element)->setPos(w, ch, 0);

            ch += (*element)->getHeight();
            ch += Vgap;
        }
        --elemNum;
    }

    myWidth = sw;
    myHeight = sh;
    myDepth = sd;
}

const char *coMenuContainer::getClassName() const
{
    return "coMenuContainer";
}

bool coMenuContainer::isOfClassName(const char *classname) const
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
            return coRowContainer::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
