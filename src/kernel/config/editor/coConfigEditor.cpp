/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigEditor.h"

#include <qapplication.h>
#include <q3action.h>
#include <q3buttongroup.h>
#include <qcolor.h>
#include <qcombobox.h>
#include <q3dockarea.h>
#include <qdom.h>
#include <qlayout.h>
#include <q3listview.h>
#include <q3mainwindow.h>
#include <qpainter.h>
#include <qpixmap.h>
#include <qradiobutton.h>
#include <q3toolbar.h>
//Added by qt3to4:
#include <Q3GridLayout>

#include "coConfigIcons.h"

coConfigEditor::coConfigEditor(QWidget *parent)
    : QWidget(parent, "Configuration Editor")
{

    config = 0;
    tree = 0;
    layout = 0;
    toolbar = 0;
    hostSectionCB = 0;

    reloadAction = 0;
    importAction = 0;
    saveAction = 0;

    createActions();

    layout = new Q3GridLayout(this);
    layout->setAutoAdd(true);

    buildDOM();
    QDomNode rootNode = config->firstChild();
    tree = new coConfigEditorListView(rootNode, this);

    coConfig::getInstance()->setAdminMode(false);
}

coConfigEditor::~coConfigEditor()
{

    delete config;
}

void coConfigEditor::createActions()
{

    reloadAction = new Q3Action(QString(), QPixmap(qembed_findImage("reload")),
                                tr("Reload"), Qt::Key_F5,
                                this, "reload");
    connect(reloadAction, SIGNAL(activated()),
            this, SLOT(reload()));

    importAction = new Q3Action(QString(), QPixmap(qembed_findImage("fileimport")),
                                tr("Import"), Qt::CTRL + Qt::Key_I,
                                this, "import");
    connect(importAction, SIGNAL(activated()),
            this, SLOT(import()));

    saveAction = new Q3Action(QString(), QPixmap(qembed_findImage("filesave")),
                              tr("Save Configuration"), Qt::CTRL + Qt::Key_S,
                              this, "save");
    connect(saveAction, SIGNAL(activated()),
            this, SLOT(save()));
}

void coConfigEditor::createToolbar(Q3MainWindow *parent)
{

    toolbar = new Q3ToolBar(parent, "cfg_editor_toolbar");

    reloadAction->addTo(toolbar);
    saveAction->addTo(toolbar);
    importAction->addTo(toolbar);

    makeHostComboBox();
}

void coConfigEditor::plug(Q3DockArea *dock)
{

    if (!toolbar)
    {
        if (dock && dock->parent()->isA("QMainWindow"))
        {
            createToolbar(static_cast<Q3MainWindow *>(dock->parent()));
        }
        else
        {
            createToolbar();
        }
    }

    dock->moveDockWindow(toolbar);
}

void coConfigEditor::setActiveHost(const QString &name)
{

    if (coConfig::getInstance()->setActiveHost(name))
    {

        delete tree;
        tree = 0;

        buildDOM();
        QDomNode rootNode = config->firstChild();
        tree = new coConfigEditorListView(rootNode, this);

        tree->show();
    }
}

void coConfigEditor::import()
{

    coConfigEditorImport().importWizard();
}

void coConfigEditor::reload()
{

    coConfig::getInstance()->reload();
    qApp->setStyle(coConfig::getInstance()->getValue("style", "UICONFIG.QTSTYLE").lower());

    delete tree;

    tree = 0;

    makeHostComboBox();

    buildDOM();

    QDomNode rootNode = config->firstChild();
    tree = new coConfigEditorListView(rootNode, this);

    tree->show();
}

void coConfigEditor::buildDOM()
{

    delete config;
    config = new QDomDocument();

#ifdef YAC
    QDomElement rootNode = config->createElement("YAC Config");
#else
    QDomElement rootNode = config->createElement("COVISE Config");
#endif

    config->appendChild(rootNode);

    buildDomBranch(rootNode, QString());
}

void coConfigEditor::buildDomBranch(QDomNode parent, coConfigEntryString scope)
{

    coConfig *configuration = coConfig::getInstance();

    //   if (!scope.isNull()) {
    //     cerr << scope << ": ";
    //   }

    coConfigEntryStringList attributes = configuration->getVariableList(scope);
    if (!attributes.empty())
    {

        for (coConfigEntryStringList::iterator item = attributes.begin();
             item != attributes.end(); item++)
        {

            //cerr << *item << " ";
            if (*item == "scope")
                continue;
            QDomElement node = config->createElement(*item);
            node.setAttribute("type", "attribute");
            node.setAttribute("scope", scope);
            node.setAttribute("value", configuration->getValue(*item, scope));

            if ((*item).getConfigScope() == coConfigConstants::Global)
            {
                if ((*item).getConfigName() == "global")
                {
                    node.setAttribute("config", "global");
                }
                else
                {
                    node.setAttribute("config", "user");
                }
            }
            else if ((*item).getConfigScope() == coConfigConstants::Host)
            {
                if ((*item).getConfigName() == "global")
                {
                    node.setAttribute("config", "host");
                }
                else
                {
                    node.setAttribute("config", "userhost");
                }
            }

            if ((*item).isListItem())
            {
                node.setAttribute("listitem", "1");
            }

            parent.appendChild(node);
        }
    }

    //cerr << endl;

    //   cerr << "coConfigEditor::buildDomBranch info: querying scope "
    //        << scope << endl;

    coConfigEntryStringList entries = configuration->getScopeList(scope);
    if (entries.empty())
        return;

    for (coConfigEntryStringList::iterator item = entries.begin();
         item != entries.end(); item++)
    {

        coConfigEntryString newScope = scope + (scope.isNull()
                                                    ? coConfigEntryString(*item)
                                                    : coConfigEntryString("." + *item));

        QDomElement node = config->createElement(*item);
        node.setAttribute("type", "element");
        node.setAttribute("scope", scope);
        if ((*item).getConfigScope() == coConfigConstants::Global)
        {
            node.setAttribute("config", (*item).getConfigName());
        }
        else if ((*item).getConfigScope() == coConfigConstants::Host)
        {
            if ((*item).getConfigName() == "global")
            {
                node.setAttribute("config", "host");
            }
            else
            {
                node.setAttribute("config", "userhost");
            }
        }
        newScope.setConfigScope((*item).getConfigScope());

        buildDomBranch(node, newScope);
        parent.appendChild(node);
    }
}

void coConfigEditor::makeHostComboBox()
{

    if (!toolbar)
        return;

    delete hostSectionCB;
    hostSectionCB = 0;

    hostSectionCB = new QComboBox(toolbar);
    hostSectionCB->insertStringList(coConfig::getInstance()->getHostnameList());
    connect(hostSectionCB, SIGNAL(activated(const QString &)),
            this, SLOT(setActiveHost(const QString &)));

    QString activeHost = coConfig::getInstance()->getActiveHost();
    for (int ctr = 0; ctr < hostSectionCB->count(); ctr++)
    {
        if (hostSectionCB->text(ctr) == activeHost)
        {
            hostSectionCB->setCurrentItem(ctr);
            break;
        }
    }
}

void coConfigEditor::save()
{

    coConfig::getInstance()->save();

#if 0
   coConfigSaveDialog *dialog = new coConfigSaveDialog(this, true);
   if (dialog->exec() == coConfigSaveDialog::Accepted)
   {
      coConfig::getInstance()->save(dialog->selectedFile(),
         dialog->getConfigScope());
   }
#endif
}

coConfigSaveDialog::coConfigSaveDialog(QWidget *parent, bool modal)
    : Q3FileDialog(parent, 0, modal)
    , coConfigConstants()
{

    buttonGroup = new Q3ButtonGroup(1, Qt::Horizontal, "Config Scope", this);
    new QRadioButton("global", buttonGroup);
    new QRadioButton("local", buttonGroup);
    buttonGroup->setButton(0);
    connect(buttonGroup, SIGNAL(clicked(int)),
            this, SLOT(selectScope(int)));

    addRightWidget(buttonGroup);

    setMode(AnyFile);

    QDir beginDir(coConfigDefaultPaths::getDefaultGlobalConfigFileName());
    beginDir.cdUp();
    setDir(beginDir);

    connect(this, SIGNAL(dirEntered(const QString &)),
            this, SLOT(dirChanged(const QString &)));

    switchLocation = true;
}

void coConfigSaveDialog::dirChanged(const QString &)
{
    switchLocation = false;
}

void coConfigSaveDialog::selectScope(int)
{
    if (switchLocation)
    {
        // cerr << "coConfigSaveDialog::selectScope info: switching to " << id << endl;
        QDir beginDir;
        if (getConfigScope() == "global")
        {
            beginDir.setPath(coConfigDefaultPaths::getDefaultGlobalConfigFileName());
        }
        else
        {
            beginDir.setPath(coConfigDefaultPaths::getDefaultLocalConfigFileName());
        }
        beginDir.cdUp();
        setDir(beginDir);
        // Set to false via dirChanged
        switchLocation = true;
    }
    else
    {
        // cerr << "coConfigSaveDialog::selectScope info: not switching" << endl;
    }
}

QString coConfigSaveDialog::getConfigScope()
{

    int id = buttonGroup->id(buttonGroup->selected());

    if (id == 0)
        return "global";
    if (id == 1)
        return "user";
    return "global";
}
