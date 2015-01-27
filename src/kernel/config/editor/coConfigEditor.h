/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGEDITOR_H
#define COCONFIGEDITOR_H

//Added by qt3to4:

#include <config/coConfigConstants.h>
#include <config/coConfigEntryString.h>

#include "coConfigEditorImport.h"
#include "coConfigEditorListView.h"
#include <Q3FileDialog>

class Q3Action;
class Q3ButtonGroup;
class QComboBox;
class Q3DockArea;
class Q3GridLayout;
class Q3ListView;
class Q3MainWindow;
class Q3ToolBar;

class CONFIGEDITOREXPORT coConfigEditor : public QWidget, protected coConfigConstants
{

    Q_OBJECT

public:
    coConfigEditor(QWidget *parent = 0);
    ~coConfigEditor();

    void plug(Q3DockArea *dock);

public slots:
    void save();
    void reload();
    void setActiveHost(const QString &name);
    void import();

protected:
    void createActions();
    void createToolbar(Q3MainWindow *parent = 0);

private:
    void buildDOM();
    void buildDomBranch(QDomNode parent, coConfigEntryString scope);

    void makeHostComboBox();

private:
    QDomDocument *config;

    Q3GridLayout *layout;
    coConfigEditorListView *tree;

    Q3ToolBar *toolbar;
    QComboBox *hostSectionCB;

    Q3Action *importAction;
    Q3Action *reloadAction;
    Q3Action *saveAction;
};

class coConfigSaveDialog : public Q3FileDialog, protected coConfigConstants
{

    Q_OBJECT

public:
    coConfigSaveDialog(QWidget *parent = 0, bool modal = false);
    virtual ~coConfigSaveDialog()
    {
    }

    QString getConfigScope();

private slots:
    void dirChanged(const QString &dir = 0);
    void selectScope(int id);

private:
    bool switchLocation;
    Q3ButtonGroup *buttonGroup;
};

#endif
