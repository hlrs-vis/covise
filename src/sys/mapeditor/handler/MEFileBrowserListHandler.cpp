/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MEFileBrowserListHandler.h"
#include "widgets/MEUserInterface.h"
#include "MEFileBrowser.h"

//======================================================================

MEFileBrowserListHandler::MEFileBrowserListHandler()
    : QObject()
{
}

MEFileBrowserListHandler *MEFileBrowserListHandler::instance()
{
    static MEFileBrowserListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MEFileBrowserListHandler();

    return singleton;
}

//======================================================================
MEFileBrowserListHandler::~MEFileBrowserListHandler()
//======================================================================
{
    qDeleteAll(browserList);
    browserList.clear();
}

//------------------------------------------------------------------------
// add a file browser
//------------------------------------------------------------------------
void MEFileBrowserListHandler::addFileBrowser(MEFileBrowser *fb)
{
    browserList.append(fb);
}

//------------------------------------------------------------------------
// remove a file browser
//------------------------------------------------------------------------
void MEFileBrowserListHandler::removeFileBrowser(MEFileBrowser *fb)
{
    browserList.remove(browserList.indexOf(fb));
}
