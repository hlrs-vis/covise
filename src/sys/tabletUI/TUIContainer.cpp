/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>

#include <QLayout>
#include <QWidget>

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
    inLayout.insert(el);
    if (numberOfColumns > 0)
    {
        relayout();
        return;
    }

    if (auto gl = gridLayout())
    {
        if (!el->isHidden())
        {
            if (el->getWidget())
                gl->addWidget(el->getWidget(), el->getYpos(), el->getXpos(), el->getHeight(), el->getWidth(), Qt::AlignBaseline);
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
    inLayout.erase(el);
    if (numberOfColumns > 0)
        relayout();

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

void TUIContainer::setNumberOfColumns(int columns)
{
    numberOfColumns = columns;
    relayout();
}

int TUIContainer::maximumNumberOfColumns() const
{
    return numberOfColumns;
}

void TUIContainer::relayout()
{
    if (auto grid = gridLayout())
    {
        // remove everything from layout
        while (layout->takeAt(0))
            ;

        int minWidth = 0, maxWidth = 0;
        std::vector<int> rowLength;
        std::vector<std::vector<TUIElement *>> rowElements;
        for (auto el: inLayout)
        {
            if (el->isHidden())
                continue;

            int x = el->getXpos();
            int y = el->getYpos();
            int w = el->getWidth();
            if (x<0 || y<0)
                continue;

            if (minWidth < w)
                minWidth = w;
            if (y >= rowLength.size())
            {
                rowLength.resize(y+1);
                rowElements.resize(y+1);
            }
            rowElements[y].push_back(el);
            if (x+w > rowLength[y])
                rowLength[y] = x+w;
            if (maxWidth < rowLength[y])
                maxWidth = rowLength[y];
        }

        int numColumns = maxWidth;
        if (numberOfColumns > 0)
            numColumns = std::max(minWidth, numberOfColumns);
        numColumns = std::max(numColumns, 1);
        std::vector<int> row;
        row.resize(rowLength.size());
        std::vector<int> rowLengthWrapped;
        rowLengthWrapped.resize(rowLength.size());
        for (size_t i=0; i<rowLength.size(); ++i)
        {
            std::sort(rowElements[i].begin(), rowElements[i].end(), [](const TUIElement *e1, const TUIElement *e2){
                return e1->getXpos() < e2->getXpos();
            });
            for (auto el: rowElements[i])
            {
                int c = rowLengthWrapped[i];
                if (c % numColumns + el->getWidth() > numColumns)
                    c += numColumns-c%numColumns;
                rowLengthWrapped[i] = c + el->getWidth();
            }
            if (i > 0)
            {
                bool wrap = false;
                for (auto el: rowElements[i-1])
                {
                    if (el->getXpos() + el->getWidth() > numColumns)
                        wrap = true;
                }

                int l = wrap ? rowLengthWrapped[i-1] : rowLength[i-1];
                row[i] = row[i-1] + (l+numColumns-1)/numColumns;
            }
        }

        for (size_t i=0; i<rowLength.size(); ++i)
        {
            bool wrap = false;
            for (auto el: rowElements[i])
            {
                if (el->getXpos() + el->getWidth() > numColumns)
                    wrap = true;
            }

            int pos = 0;
            for (auto el: rowElements[i])
            {
                int x = el->getXpos();
                int c = x % numColumns;
                if (wrap)
                {
                    c = pos % numColumns;
                    if (c + el->getWidth() > numColumns)
                    {
                        pos += numColumns-c%numColumns;
                        c = 0;
                    }
                }
                int r = row[el->getYpos()] + pos/numColumns;
                if (el->getWidget())
                    grid->addWidget(el->getWidget(), r, c, el->getHeight(), el->getWidth(), Qt::AlignBottom);
                else if (el->getLayout())
                    grid->addLayout(el->getLayout(), r, c, el->getHeight(), el->getWidth(), Qt::AlignBottom);
                pos += el->getWidth();
            }
        }

        for (int i = 0; i < grid->rowCount(); i++)
            grid->setRowStretch(i, 0);
        grid->setRowStretch(grid->rowCount(), 100);
        for (int i = 0; i < grid->columnCount(); i++)
            grid->setColumnStretch(i, 0);
        grid->setColumnStretch(grid->columnCount(), 100);
    }
}
