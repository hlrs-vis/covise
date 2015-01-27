/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FAVORITES_H
#define ME_FAVORITES_H

#include <QToolButton>

class QDropEvent;
class QDragEnterEvent;
class QMouseEvent;
class QAction;

//================================================
class MEFavorites : public QToolButton
//================================================
{
    Q_OBJECT

public:
    MEFavorites(QWidget *parent = 0, QString sname = QString::null);
    ~MEFavorites();

    void setModuleName(const QString &);
    QString getModuleName();
    QString getStartText();
    QAction *getAction()
    {
        return m_action;
    };

private:
    QString m_label, m_category;
    QAction *m_action;

signals:

    void initModule(const QString &);

protected:
    void mouseMoveEvent(QMouseEvent *e);
    void mouseDoubleClickEvent(QMouseEvent *e);
    void dropEvent(QDropEvent *e);
    void dragEnterEvent(QDragEnterEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);
    void dragMoveEvent(QDragMoveEvent *e);
};
#endif
