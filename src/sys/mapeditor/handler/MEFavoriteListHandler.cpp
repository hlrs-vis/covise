/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QToolBar>
#include <QDebug>

#include "MEFavoriteListHandler.h"
#include "MEMainHandler.h"
#include "widgets/MEUserInterface.h"

//======================================================================

MEFavoriteListHandler::MEFavoriteListHandler()
    : QObject()

{
}

MEFavoriteListHandler *MEFavoriteListHandler::instance()
{
    static MEFavoriteListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MEFavoriteListHandler();

    return singleton;
}

//======================================================================
MEFavoriteListHandler::~MEFavoriteListHandler()
//======================================================================
{
    qDeleteAll(favoriteList);
    favoriteList.clear();
}

//======================================================================
// store a list of favorites given by default ot readin from config
//======================================================================
void MEFavoriteListHandler::storeFavoriteList(const QStringList &list)
{
    favorites = list;
}

//======================================================================
// get the number of favorites
//======================================================================
int MEFavoriteListHandler::getNoOfFavorites()
{
    return favoriteList.count();
}

//======================================================================
// set favorites invisible
//======================================================================
void MEFavoriteListHandler::setVisible(bool state)
{
    foreach (MEFavorites *fav, favoriteList)
        fav->getAction()->setVisible(state);
}

//======================================================================
// create the favorite buttons for the toolbar
//======================================================================
void MEFavoriteListHandler::createFavorites()
{

    QToolBar *tb = MEUserInterface::instance()->getToolBar();

    for ( int k = 0; k < favorites.count(); k++)
        favoriteList.append(new MEFavorites(tb, favorites[k]));
    MEMainHandler::instance()->getConfig()->setValue("System.MapEditor.General.FavoritesList", favorites.join(" "));
    if (!favoriteList.isEmpty())
        MEUserInterface::instance()->hideFavoriteLabel();
}

//======================================================================
// insert a new favorite name
//======================================================================
void MEFavoriteListHandler::insertFavorite(const QString &newname, const QString &insertname)
{

    // insert new name to list
    // create a new toolbutton
    if (!favorites.contains(newname))
    {
        favorites.insert(favorites.indexOf(insertname), newname);

        QToolBar *tb = MEUserInterface::instance()->getToolBar();
        favoriteList.append(new MEFavorites(tb, newname));
        updateFavorites();
        MEMainHandler::instance()->getConfig()->setValue("System.MapEditor.General.FavoritesList", favorites.join(" "));
        MEUserInterface::instance()->hideFavoriteLabel();
    }
}

//======================================================================
// add a new favorite name
//======================================================================
void MEFavoriteListHandler::addFavorite(const QString &newname)
{
    if (!favorites.contains(newname))
    {
        favorites.append(newname);

        QToolBar *tb = MEUserInterface::instance()->getToolBar();
        favoriteList.append(new MEFavorites(tb, newname));
        updateFavorites();
        MEMainHandler::instance()->getConfig()->setValue("System.MapEditor.General.FavoritesList", favorites.join(" "));
        MEUserInterface::instance()->hideFavoriteLabel();
    }
}

//======================================================================
// remove a favorite object from list
//======================================================================
void MEFavoriteListHandler::removeFavorite(const QString &name)
{
    QToolBar *tb = MEUserInterface::instance()->getToolBar();
    foreach (MEFavorites *fav, favoriteList)
    {
        if (fav->getModuleName() == name)
        {
            favoriteList.remove(favoriteList.indexOf(fav));
            tb->removeAction(fav->getAction());
        }
    }
    favorites.removeAt(favorites.indexOf(name));

    MEMainHandler::instance()->getConfig()->setValue("System.MapEditor.General.FavoritesList", favorites.join(" "));
    if (favoriteList.isEmpty())
        MEUserInterface::instance()->showFavoriteLabel();
}

//======================================================================
// sort the favorite list
//======================================================================
void MEFavoriteListHandler::sortFavorites()
{
    favorites.sort();
    updateFavorites();
}

//======================================================================
// sort the favorite list
//======================================================================
void MEFavoriteListHandler::setEnabled(bool state)
{

    foreach (MEFavorites *fav, favoriteList)
        fav->setEnabled(state);
}

//======================================================================
// modify the favorite label list
//======================================================================
void MEFavoriteListHandler::updateFavorites()
{
    for (int i = 0; i < favoriteList.count(); i++)
    {
        MEFavorites *fav = favoriteList[i];
        fav->setModuleName(favorites[i]);
    }
}
