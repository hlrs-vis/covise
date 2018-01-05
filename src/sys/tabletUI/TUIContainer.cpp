/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>

#include <QLayout>

#include "TUIContainer.h"
#include <iostream>

/// Constructor
TUIContainer::TUIContainer(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
}

/// Destructor
TUIContainer::~TUIContainer()
{
    removeAllChildren();
}

void TUIContainer::removeAllChildren()
{
    while (!elements.empty())
    {
        TUIElement *el = elements.front();
        elements.pop_front();
        el->setParent(NULL);
        delete el;
    }
}

/** Appends a child widget to this elements layout.
  @param el element to add
*/
void TUIContainer::addElementToLayout(TUIElement *el)
{
    if (auto gl = gridLayout())
    {
        /*
      if((el->getHeight() != 0) && (el->getWidth() != 0))

      else
         layout->addWidget(el->getWidget(),el->getYpos(),el->getXpos());
      */

        if (!el->isHidden())
        {
            if (el->getWidget())
                gl->addWidget(el->getWidget(), el->getYpos(), el->getXpos(), el->getHeight(), el->getWidth());
            else if (el->getLayout())
                gl->addLayout(el->getLayout(), el->getYpos(), el->getXpos(), el->getHeight(), el->getWidth(), Qt::AlignBaseline);
        }

        for (int i = 0; i < gl->rowCount(); i++)
            gl->setRowStretch(i, 0);
        gl->setRowStretch(gl->rowCount(), 100);
        for (int i = 0; i < gl->columnCount(); i++)
            gl->setColumnStretch(i, 0);
        gl->setColumnStretch(gl->columnCount(), 100);
    }
}

/** Appends a child to this container.
  @param el element to add
*/
void TUIContainer::addElement(TUIElement *el)
{
    if (el->getParent() == this)
        return;
    TUIContainer *test = this;

    // Make sure this element is not its own parent:
    do
    {
        if (test == el)
        {
            std::cerr << "can't add a parent as child" << std::endl;
            return;
        }
    } while ((test = test->getParent()));

    elements.push_back(el);
    el->setParent(this);
}

/** Adds the specified element to the scenegraph
    if it has previously been removed.
  @param el element to add
*/
void TUIContainer::showElement(TUIElement *)
{
}

/** Removes a child from this container.
  @param el element to remove
*/
void TUIContainer::removeElement(TUIElement *el)
{
    ElementList::iterator it = std::find(elements.begin(), elements.end(), el);
    if (it != elements.end())
        elements.erase(it);
    el->setParent(NULL);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIContainer::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
    for (ElementList::iterator it = elements.begin(); it != elements.end(); ++it)
    {
        (*it)->setEnabled(en);
    }
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIContainer::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
    for (ElementList::iterator it = elements.begin(); it != elements.end(); ++it)
    {
        (*it)->setHighlighted(hl);
    }
}

const char *TUIContainer::getClassName() const
{
    return "TUIContainer";
}

QGridLayout *TUIContainer::gridLayout() const
{
    return dynamic_cast<QGridLayout *>(layout);
}
