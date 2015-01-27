/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHBoxLayout>

#include "MEExtendedPart.h"
#include "MEFileBrowserPort.h"
#include "MEColorMapPort.h"
#include "MEFileBrowser.h"
#include "color/MEColorMap.h"

int stretch = 4;

//------------------------------------------------------------------------
MEExtendedPart::MEExtendedPart(QWidget *parent, MEParameterPort *p)
    //------------------------------------------------------------------------
    : QFrame(parent),
      port(p)
{

    // create second widget and layout for browser
    setFrameStyle(QFrame::Box | QFrame::Raised);
    hide();

    // create layout
    extendedLayout = new QHBoxLayout(this);
    extendedLayout->setMargin(2);
    extendedLayout->setSpacing(2);

    hide();
}

//------------------------------------------------------------------------
void MEExtendedPart::addBrowser()
//------------------------------------------------------------------------
{
    MEFileBrowser *browser = static_cast<MEFileBrowserPort *>(port)->getBrowser();
    if (browser && browser->parentWidget() == 0)
    {
        if (extendedLayout->indexOf(browser) == -1)
            extendedLayout->addWidget(browser, stretch);
        browser->show();
    }

    show();
}

//------------------------------------------------------------------------
void MEExtendedPart::addColorMap()
//------------------------------------------------------------------------
{
    MEColorMap *map = static_cast<MEColorMapPort *>(port)->getColorMap();
    if (map && map->parentWidget() == 0)
    {
        if (extendedLayout->indexOf(map) == -1)
            extendedLayout->addWidget(map, stretch);
        map->show();
    }

    show();
}

//------------------------------------------------------------------------
void MEExtendedPart::removeBrowser()
//------------------------------------------------------------------------
{
    MEFileBrowser *browser = static_cast<MEFileBrowserPort *>(port)->getBrowser();
    if (browser)
    {
        if (extendedLayout->indexOf(browser) != -1)
            extendedLayout->removeWidget((QWidget *)browser);
        browser->setParent(0);
        browser->hide();
    }

    hide();
}

//------------------------------------------------------------------------
void MEExtendedPart::removeColorMap()
//------------------------------------------------------------------------
{
    MEColorMap *map = static_cast<MEColorMapPort *>(port)->getColorMap();
    if (map)
    {
        if (extendedLayout->indexOf(map) != -1)
            extendedLayout->removeWidget((QWidget *)map);
        map->setParent(0);
        map->hide();
    }

    hide();
}

//------------------------------------------------------------------------
MEExtendedPart::~MEExtendedPart()
//------------------------------------------------------------------------
{
}
