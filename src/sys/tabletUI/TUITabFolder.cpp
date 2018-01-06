/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QTabWidget>
#include <QStackedWidget>
#include <QComboBox>
#include <QGridLayout>

#include "TUITabFolder.h"
#include "TUITab.h"
#include "TUIApplication.h"

const bool StackToplevelTabs = false;
const bool StackTabs = false;

/// Constructor
TUITabFolder::TUITabFolder(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{
    width = -1;
    bool toplevel = parent == 1;
    if (!StackTabs && !(toplevel && StackToplevelTabs))
    {
        tabWidget = new QTabWidget(w);
        tabWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
        tabWidget->setMovable(true);
        widget = tabWidget;
        connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(valueChanged(int)));
    }
    else
    {
        auto frame = new QFrame(w);
        widget = frame;
        switchWidget = new QComboBox(frame);
        switchWidget->setMaxVisibleItems(20);

        stackWidget = new QStackedWidget(frame);
        stackWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

        connect(switchWidget, SIGNAL(activated(int)), this, SLOT(setCurrentIndex(int)));

        auto grid = new QGridLayout(frame);
        layout = grid;
        grid->addWidget(switchWidget, 0, 0);
        grid->addWidget(stackWidget, 1, 0);

        frame->setLayout(grid);
        frame->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
        frame->setFrameShape(QFrame::StyledPanel);
    }
}

/// Destructor
TUITabFolder::~TUITabFolder()
{
    removeAllChildren();
    delete tabWidget;
    delete switchWidget;
    delete stackWidget;
}

void TUITabFolder::valueChanged(int index)
{
    QWidget *tab = nullptr;
    if (tabWidget)
        tab = tabWidget->widget(index);
    else if (stackWidget)
        tab = stackWidget->widget(index);
    if (!tab)
        return;
    for (ElementList::iterator it = elements.begin(); it != elements.end(); ++it)
    {
        TUIElement *el = *it;
        if (el->getWidget() == tab)
        {
            if (strcmp(el->getClassName(), "TUITab") == 0)
            {
                assert(dynamic_cast<TUITab *>(el));
                ((TUITab *)el)->activated();
            }
            // else cant activate other elements, just tabs
        }
    }
}

void TUITabFolder::setCurrentIndex(int index)
{
    if (tabWidget)
    {
        tabWidget->setCurrentIndex(index);
    }
    else  if (stackWidget)
    {
        switchWidget->setCurrentIndex(index);
        stackWidget->setCurrentIndex(index);
    }

    valueChanged(index);
}

/** Appends a child widget to this elements layout.
  @param el element to add
*/
void TUITabFolder::addElementToLayout(TUIElement *el)
{
    if (!el->getWidget())
        return;

    if (tabWidget)
    {
        if (tabWidget->indexOf(el->getWidget()) < 0)
        {
            if (!el->isHidden())
                tabWidget->addTab(el->getWidget(), el->getLabel());
        }
        else
        {
            tabWidget->setTabText(tabWidget->indexOf(el->getWidget()), el->getLabel());
        }
    }
    else if (stackWidget)
    {
        if (stackWidget->indexOf(el->getWidget()) < 0)
        {
            if (!el->isHidden())
            {
                stackWidget->addWidget(el->getWidget());
                switchWidget->addItem(el->getLabel());
            }
        }
        else
        {
            switchWidget->setItemText(stackWidget->indexOf(el->getWidget()), el->getLabel());
        }

    }
}

const char *TUITabFolder::getClassName() const
{
    return "TUITabFolder";
}

int TUITabFolder::indexOf(QWidget *widget) const
{
    if (tabWidget)
        return tabWidget->indexOf(widget);
    else if (stackWidget)
        return stackWidget->indexOf(widget);

    return -1;
}

void TUITabFolder::addTab(QWidget *widget, QString label)
{
    if (tabWidget)
    {
        tabWidget->addTab(widget, label);
    }
    else if (stackWidget)
    {
        stackWidget->addWidget(widget);
        switchWidget->addItem(label);
    }
}

void TUITabFolder::removeTab(int index)
{
    if (tabWidget)
    {
        tabWidget->removeTab(index);
    }
    else if (stackWidget)
    {
        stackWidget->removeWidget(stackWidget->widget(index));
        switchWidget->removeItem(index);
    }
}
