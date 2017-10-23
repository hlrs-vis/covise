#include "QtMainWindow.h"
#include <QMenu>

namespace opencover
{

QtMainWindow::QtMainWindow(QWidget *parent)
: QMainWindow (parent)
{

}

QMenu *QtMainWindow::createPopupMenu()
{
    auto menu = new QMenu(this);
    for (auto act: m_contextActions)
    {
        menu->addAction(act);
    }
    return menu;
}

void QtMainWindow::addContextAction(QAction *act)
{
    m_contextActions.push_back(act);
}

}

#include "moc_QtMainWindow.cpp"
