/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MODULEPANEL_H
#define ME_MODULEPANEL_H

#include <QDialog>

class QCloseEvent;
class QTabWidget;
class QPushButton;

class MEModuleParameter;
class MENode;

//================================================
class MEModulePanel : public QDialog
//================================================
{
    Q_OBJECT

public:
    MEModulePanel(QWidget *parent = 0);
    ~MEModulePanel();

    static MEModulePanel *instance();

    void init();
    void closeWindow();
    void closeLastWindow();
    void resetButtons(MENode *node);
    void changeModuleInfoTitle(MEModuleParameter *info, const QString &title);
    void showModuleInfo(MEModuleParameter *info);
    void hideModuleInfo(MEModuleParameter *info);
    void setMasterState(bool);
    void setCancelButtonState(bool state);
    void setDetailText(const QString &text);
    MEModuleParameter *addModuleInfo(MENode *node);

private:
    bool m_firsttime;
    QPushButton *m_helpPB, *m_cancelPB, *m_detailPB, *m_okPB, *m_executePB;
    QTabWidget *m_tabWidget;
    void fitWindow(int x, int y);

protected:
    void closeEvent(QCloseEvent *);

public slots:

    void enableExecCB(bool);
    void execCB();
    void cancelCB();
    void helpCB();
    void okCB();
    void detailCB();
    void tabChanged(int index);
    void paramChanged(bool);
};
#endif
