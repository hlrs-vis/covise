/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coDialog.h>

#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPopupHandle.h>

using namespace std;

namespace vrui
{

coDialog::coDialog(const string &name)
{
    itemsFrame = new coFrame("UI/Frame"); // name is texture name!
    itemsFrame->fitToParent();

    handle = new coPopupHandle(name);
    handle->addElement(itemsFrame);
}

/// Destructor.
coDialog::~coDialog()
{
    delete itemsFrame;
    delete handle;
}

void coDialog::addElement(coUIElement *el)
{
    itemsFrame->addElement(el);
}

void coDialog::showElement(coUIElement *el)
{
    itemsFrame->showElement(el);
}

void coDialog::removeElement(coUIElement *el)
{
    itemsFrame->removeElement(el);
}

void coDialog::childResized(bool shrink)
{
    itemsFrame->childResized(shrink);
}

void coDialog::resizeToParent(float x, float y, float z, bool shrink)
{
    itemsFrame->resizeToParent(x, y, z, shrink);
}

void coDialog::setEnabled(bool en)
{
    itemsFrame->setEnabled(en);
}

void coDialog::setHighlighted(bool hl)
{
    itemsFrame->setHighlighted(hl);
}

void coDialog::setVisible(bool newState)
{
    handle->setVisible(newState);
}

vruiTransformNode *coDialog::getDCS()
{
    return handle->getDCS();
}

void coDialog::setTransformMatrix(vruiMatrix *matrix)
{
    handle->setTransformMatrix(matrix);
}

void coDialog::setTransformMatrix(vruiMatrix *matrix, float scale)
{
    handle->setTransformMatrix(matrix, scale);
}

void coDialog::setScale(float s)
{
    handle->setScale(s);
}

float coDialog::getScale() const
{
    return handle->getScale();
}

float coDialog::getWidth() const
{
    return handle->getWidth();
}

float coDialog::getHeight() const
{
    return handle->getHeight();
}

float coDialog::getDepth() const
{
    return handle->getDepth();
}

float coDialog::getXpos() const
{
    return handle->getXpos();
}

float coDialog::getYpos() const
{
    return handle->getYpos();
}

float coDialog::getZpos() const
{
    return handle->getZpos();
}

void coDialog::setPos(float x, float y, float z)
{
    handle->setPos(x, y, z);
}

void coDialog::setSize(float w, float h, float d)
{
    handle->setSize(w, h, d);
}

void coDialog::setSize(float size)
{
    handle->setSize(size);
}

bool coDialog::update()
{
    return handle->update();
}
}
