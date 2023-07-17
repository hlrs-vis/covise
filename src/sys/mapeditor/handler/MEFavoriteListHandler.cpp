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
    favorites = MEMainHandler::instance()->getConfig().array<std::string>("General", "FavoritesList", std::vector<std::string>{"RWCovise:IO", "Colors:Mapper", "IsoSurface:Mapper", "CuttingSurface:Filter", "Collect:Tools", "Renderer:Renderer"});
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
    std::vector<std::string> f;
    for (const auto &s : list)
    {
        std::cerr << "storeFavoriteList " << s.toStdString() << std::endl;
        f.push_back(s.toStdString());
    }
    *favorites = f;
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

    for ( int k = 0; k < favorites->size(); k++)
    {
        std::string favorite = (*favorites)[k];
        favoriteList.append(new MEFavorites(tb, favorite.c_str()));
    }
    if (!favoriteList.isEmpty())
        MEUserInterface::instance()->hideFavoriteLabel();
}

//======================================================================
// insert a new favorite name after the insertname
//======================================================================
void MEFavoriteListHandler::insertFavorite(const QString &newname, const QString &insertname)
{
    addOrInsertFavorite(newname, insertname);
}

//======================================================================
// add a new favorite name
//======================================================================
void MEFavoriteListHandler::addFavorite(const QString &newname)
{
    addOrInsertFavorite(newname);
}

void MEFavoriteListHandler::addOrInsertFavorite(const QString &newname, const QString &insertname)
{
    // insert new name to list
    // create a new toolbutton
    auto f = favorites->value();
    auto it = std::find(f.begin(), f.end(), newname.toStdString());
    if (auto it = std::find(f.begin(), f.end(), newname.toStdString()) == f.end())
    {
        auto pos = insertname.isNull() ? std::find(f.begin(), f.end(), insertname.toStdString()) : f.end();
        
        f.insert(pos, newname.toStdString());

        QToolBar *tb = MEUserInterface::instance()->getToolBar();
        favoriteList.append(new MEFavorites(tb, newname));
        *favorites = f;
        updateFavorites();
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
    auto f = favorites->value();
    f.erase(std::remove(f.begin(), f.end(), name.toStdString()));
    *favorites = f;

    if (favoriteList.isEmpty())
        MEUserInterface::instance()->showFavoriteLabel();
}

//======================================================================
// sort the favorite list
//======================================================================
void MEFavoriteListHandler::sortFavorites()
{
    auto f = favorites->value();
    std::sort(f.begin(), f.end());
    *favorites = f;
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
        std::string f = (*favorites)[i];
        fav->setModuleName(f.c_str());
    }
}
