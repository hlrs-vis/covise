/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FAVORITELISTHANDLER_H
#define ME_FAVORITELISTHANDLER_H

#include <QObject>

#include "widgets/MEFavorites.h"
#include <config/array.h>

class QString;
class QColor;
class QToolBar;

class MEFavoriteListHandler : public QObject
{
    Q_OBJECT

public:
    MEFavoriteListHandler();
    ~MEFavoriteListHandler();

    static MEFavoriteListHandler *instance();

    void sortFavorites();
    void updateFavorites();
    void createFavorites();
    void insertFavorite(const QString &newname, const QString &insertname);
    void addFavorite(const QString &newname);
    void removeFavorite(const QString &name);
    void storeFavoriteList(const QStringList &);
    void setVisible(bool state);
    void setEnabled(bool state);
    int getNoOfFavorites();

private:
    QVector<MEFavorites *> favoriteList;
    std::unique_ptr<covise::config::Array<std::string>> favorites;

    void addOrInsertFavorite(const QString &newname, const QString &insertname = QString());
};
#endif
