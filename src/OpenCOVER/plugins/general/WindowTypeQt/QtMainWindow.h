#ifndef UI_QT_MAINWINDOW_H
#define UI_QT_MAINWINDOW_H

#include <QMainWindow>
#include <QList>

class QMenu;
class QAction;

namespace opencover {

//! store the data for the representation of a UI Element within a QtView
class QtMainWindow: public QMainWindow
{
    Q_OBJECT

public:
    //! create for @param elem which has a corresponding @param obj
    QtMainWindow(QWidget *parent = nullptr);
    QMenu *createPopupMenu() override;
    void addContextAction(QAction *act);
    void closeEvent(QCloseEvent *ev) override;
    void changeEvent(QEvent *ev) override;

signals:
    void fullScreenChanged(bool);
    void closing();

private:
    QList<QAction *> m_contextActions;
    bool m_fullscreen = false;

};

}
#endif
