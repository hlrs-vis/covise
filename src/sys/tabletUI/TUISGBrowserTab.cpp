/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <assert.h>
#include <iostream>

#include "TUISGBrowserTab.h"
#include "TUITab.h"
#include "TUIApplication.h"

#include <QLabel>
#include <QTabWidget>

#include <QGridLayout>
#include <QFrame>
#include <QImage>
#include <QPixmap>
#include <QIcon>
#include <QShortcut>
#include <QColorDialog>
#include <QAction>
#include <QTimer>
#include <QDateTime>
#include <util/unixcompat.h>

#include <net/covise_connect.h>
#include <net/message.h>
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#else

#define GL_POINTS 0x0000
#define GL_LINES 0x0001
#define GL_LINE_LOOP 0x0002
#define GL_LINE_STRIP 0x0003
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TRIANGLE_FAN 0x0006
#define GL_QUADS 0x0007
#define GL_QUAD_STRIP 0x0008
#define GL_POLYGON 0x0009
#endif

#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

#include "icons/find.xpm"
#include "icons/recSel.xpm"
#include "icons/wireon.xpm"
#include "icons/wireoff.xpm"
#define MY_GL_LINES_ADJACENCY_EXT 0x000A
#define MY_GL_LINE_STRIP_ADJACENCY_EXT 0x000B
#define MY_GL_TRIANGLES_ADJACENCY_EXT 0x000C
#define MY_GL_TRIANGLE_STRIP_ADJACENCY_EXT 0x000D

using std::string;

//Constructor
TUISGBrowserTab::TUISGBrowserTab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
    , simu(0)
{
    numItems = 0;
    receivedTextures = 0;
    currentTexture = false;

    texturePluginDir = QDir::homePath() + "/" + ".texturePlugin" + "/";
    texturePluginTempDir = texturePluginDir + "temp" + "/";
    textureDir = texturePluginDir + "textures" + "/";
    currentDir = textureDir;
    //agottsimshown=false;
    load = NULL;

    for (int j = 0; j < 21; j++)
    {
        textureModes[j] = 1;
        textureTexGenModes[j] = 0;
        textureIndices[j] = -1;
    }

    QDir *path = new QDir();

    if (!path->exists(texturePluginDir))
        path->mkdir(texturePluginDir);
    if (!path->exists(texturePluginTempDir))
        path->mkdir(texturePluginTempDir);
    if (!path->exists(textureDir))
        path->mkdir(textureDir);
    delete path;

    frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);

    auto grid = new QGridLayout(frame);
    layout = grid;
    widget = frame;

    Treelayout = new QHBoxLayout();
    grid->addLayout(Treelayout, 0, 0, Qt::AlignLeft);

    treeWidget = new nodeTree(frame);
    treeWidget->init();

    QSize Bsize = QSize(35, 35);
    QSize Isize = QSize(25, 25);

    treeWidget->resize(frame->width() - 200, frame->height() - 150);
    Treelayout->addWidget(treeWidget);
    connect(treeWidget, SIGNAL(currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *)), this, SLOT(itemChangedSlot(QTreeWidgetItem *, QTreeWidgetItem *)));
    connect(treeWidget, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(updateItemState(QTreeWidgetItem *, int)));
    connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(updateSelection()));
    connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(itemProperties()));
    connect(treeWidget, SIGNAL(itemExpanded(QTreeWidgetItem *)), this, SLOT(updateExpand(QTreeWidgetItem *)));
    connect(treeWidget, SIGNAL(itemCheckStateChanged(QTreeWidgetItem *,bool)), this, SLOT(showNode(QTreeWidgetItem *,bool)));

    QHBoxLayout *Hlayout = new QHBoxLayout();
    grid->addLayout(Hlayout, 1, 0, 1, 1, Qt::AlignLeft);
    QGridLayout *findlayout = new QGridLayout();

    findEdit = new QLineEdit(frame);
    findEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(findEdit, SIGNAL(editingFinished()), this, SLOT(findItemSLOT()));

    Hlayout->addWidget(findEdit);
    Hlayout->addLayout(findlayout);

    findButton = new QPushButton(frame);
    findButton->setToolTip("Find");
    findButton->setIconSize(Isize);
    findButton->setIcon(QPixmap(find_xpm));
    findButton->setMinimumSize(Bsize);
    findButton->setMaximumSize(Bsize);
    findlayout->addWidget(findButton, 0, 0, Qt::AlignLeft);
    connect(findButton, SIGNAL(clicked()), this, SLOT(findItemSLOT()));
    QShortcut *shortcut0 = new QShortcut(QKeySequence(tr("Alt+F", "Find")), frame);
    connect(shortcut0, SIGNAL(activated()), this, SLOT(findItemSLOT()));

    recursiveButton = new QPushButton(frame);
    recursiveButton->setToolTip("Recursive node selection");
    recursiveButton->setIconSize(Isize);
    recursiveButton->setIcon(QPixmap(recsel_xpm));
    recursiveButton->setCheckable(true);
    recursiveButton->setMinimumSize(Bsize);
    recursiveButton->setMaximumSize(Bsize);
    findlayout->addWidget(recursiveButton, 0, 1, Qt::AlignLeft);
    connect(recursiveButton, SIGNAL(toggled(bool)), this, SLOT(setRecursiveSel(bool)));

    QColor forButton = QColor(255, 0, 0);
    colorButton = new QPushButton(frame);
    colorButton->setToolTip("select color");
    colorButton->setMinimumSize(Bsize);
    colorButton->setMaximumSize(Bsize);
    colorButton->setPalette(QPalette(forButton));
    findlayout->addWidget(colorButton, 0, 5, Qt::AlignLeft);
    connect(colorButton, SIGNAL(clicked()), this, SLOT(setColor()));

    ColorR = (float)forButton.red() / 255.0;
    ColorG = (float)forButton.green() / 255.0;
    ColorB = (float)forButton.blue() / 255.0;

    wireON = QIcon(QPixmap(wireon_xpm));
    wireOFF = QIcon(QPixmap(wireoff_xpm));

    polyMode = 2;

    selectionModeCB = new QComboBox(frame);
    selectionModeCB->setToolTip("Selection Mode");
    selectionModeCB->setEditable(false);
    selectionModeCB->addItem("Filled");
    selectionModeCB->addItem("Wireframe selection color");
    selectionModeCB->addItem("Wireframe object color");
    selectionModeCB->addItem("Outline");
    selectionModeCB->setCurrentIndex(polyMode);
    grid->addWidget(selectionModeCB, 2, 0, Qt::AlignRight);
    connect(selectionModeCB, SIGNAL(activated(int)), this, SLOT(changeSelectionMode(int)));

    selModeCBox = new QCheckBox(frame);
    selModeCBox->setText("Show selection");
    selModeCBox->setToolTip("show /hide selection in OpenCover");
    selModeCBox->setChecked(true);
    grid->addWidget(selModeCBox, 2, 0, Qt::AlignLeft);
    connect(selModeCBox, SIGNAL(toggled(bool)), this, SLOT(setSelMode(bool)));

    selMode = 1;
    setSelMode(true);

    selectCBox = new QCheckBox(frame);
    selectCBox->setText("Select &All");
    grid->addWidget(selectCBox, 3, 0, Qt::AlignLeft);
    connect(selectCBox, SIGNAL(toggled(bool)), this, SLOT(selectAllNodes(bool)));

    QPushButton *propButton = new QPushButton(frame);
    propButton->setText(" &Properties ");
    grid->addWidget(propButton, 3, 0, Qt::AlignRight);
    connect(propButton, SIGNAL(clicked()), this, SLOT(showhideDialog()));

    showCBox = new QCheckBox(frame);
    showCBox->setText("Show all nodes");
    grid->addWidget(showCBox, 4, 0, Qt::AlignLeft);
    connect(showCBox, SIGNAL(toggled(bool)), this, SLOT(showAllNodes(bool)));

    updateButton = new QPushButton(frame);
    updateButton->setText(" &Update ");
    grid->addWidget(updateButton, 4, 0, Qt::AlignRight);
    connect(updateButton, SIGNAL(clicked()), this, SLOT(updateScene()));

    showNodes = showCBox->isChecked();
    recursiveSel = recursiveButton->isChecked();

    find = false;
    root = false;

    QAction *newAct = new QAction("Update", this);
    connect(newAct, SIGNAL(triggered()), this, SLOT(updateItem()));
    propAct = new QAction("Properties", this);
    propAct->setCheckable(true);
    propAct->setChecked(false);
    connect(propAct, SIGNAL(triggered()), this, SLOT(showhideDialog()));

    firsttime = true;

    QAction *center = new QAction("Center Object", this);
    connect(center, SIGNAL(triggered()), this, SLOT(centerObject()));

    treeWidget->setContextMenuPolicy(Qt::ActionsContextMenu);
    treeWidget->addAction(center);
    treeWidget->addAction(propAct);
    treeWidget->addAction(newAct);
    //treeWidget->addAction(load);

    propertyDialog = new PropertyDialog(this);
    //Treelayout->addWidget(propertyDialog); (not possible to hide on init?! Qt-bug?)

    restraint = new covise::coRestraint();

    updateScene();

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif


    receivingTextures = true;
    thread = new SGTextureThread(this);
    thread->start();
    updateTimer = new QTimer();
    connect(updateTimer, SIGNAL(timeout()), this, SLOT(updateTextureButtons()));
    updateTimer->setSingleShot(false);
    updateTimer->start(250);
    buttonList = QStringList();
    indexList.clear();
}

TUISGBrowserTab::~TUISGBrowserTab()
{
    receivingTextures = false;
    thread->terminateTextureThread();

    simu.clear();
    // Remove all Files from temp directory
    QDir temp(texturePluginTempDir);
    QStringList fileList = temp.entryList();
    QStringList::Iterator it;
    for (it = fileList.begin(); it != fileList.end(); ++it)
    {
        QFile::remove(texturePluginTempDir + *it);
    }

    int count = 0;
    while (!thread->isFinished())
    {
        count++;
#ifndef _WIN32_WCE
        usleep(5);
#endif
    }
    delete thread;

    delete propertyDialog;
    delete restraint;
}

void TUISGBrowserTab::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    int texNumber;
    int width;
    int height;
    int depth;
    int dataLength;
    //gottlieb<
    if (type == TABLET_SIM_SETSIMPAIR)
    {
        char *nodePath;
        char *simPath;
        char *simName;
        tb >> nodePath;
        tb >> simPath;
        tb >> simName;
        //cout<<"new MAP:   "<<nodePath<<"---"<<simPath<<"---"<<simName<<endl;
        SimPath_SimName[simPath] = simName;
        SIM_Status[simPath] = true;
        if (new_Sim(nodePath, simPath))
        {
            //cout<<"create a new item in the CAD_SIM_Node map"<<endl;
            CAD_SIM_Node[nodePath].push_back(simPath);
            CAD_Status[nodePath] = true;
            //SIM_Status[simPath]=true;
        }
    }

    if (type == TABLET_SIM_SHOW_HIDE)
    {
        int state;
        tb >> state;
        char *nodePath;
        char *simPath;
        tb >> nodePath;
        tb >> simPath;

        //cout<<"-**Calling showhideSimItems="<<state<<" ||| "<<nodePath<<" ||| "<<simPath<<endl;
        showhideSimItems(state, simPath);
    }

    if (type == TABLET_LOADFILE_SATE)
    {
        int state;

        tb >> state;
        if (state != 0 && load == NULL)
        {
            std::cout << "Message from SG-Browser-Plugin !!Loadfiles-Button will be created for right mouse menue in SGBrowser" << std::endl;
            load = new QAction("Load Files", this);
            connect(load, SIGNAL(triggered()), this, SLOT(loadFiles()));
            treeWidget->addAction(load);
        }
        else
        {
        } //>gottlieb
    }
    if (type == TABLET_BROWSER_NODE)
    {
        int nodetype, numChildren;
        char *name;
        char *nodeClassName;
        char *path, *parentPath;
        int nodeMode;

        tb >> nodetype;
        tb >> name;
        tb >> nodeClassName;
        tb >> nodeMode;
        tb >> path;
        tb >> parentPath;
        tb >> numChildren;

        QString nodeName(name);
        QString itemPath(path);
        QString className(nodeClassName);

        QString parentItemPath(parentPath);

        itemPath = itemPath.trimmed();
        parentItemPath = parentItemPath.trimmed();

        bool showItem;
        QString typeName, nodeTip;
        QColor nodeColor;

        nodeTip = QString("Type: %1 \n Name: %2 \n Path: %3").arg(nodeClassName).arg(name).arg(path);

        if (nodeName == "OBJECTS_ROOT" && !root)
        {
            treeWidget->clear();
            showItem = decision(nodetype, typeName, nodeColor);
            new nodeTreeItem(treeWidget, nodeName, className, typeName, nodeColor, nodeTip, nodeMode, path, numChildren);
            root = true;
        }
        else
        {
            if (treeWidget->findParent(itemPath) == NULL)
            {
                if (is_SimNode(itemPath.toUtf8().data()))
                {
                    nodetype = SG_SIM_NODE;
                }
                showItem = decision(nodetype, typeName, nodeColor);
                nodeTreeItem *parentItem = treeWidget->findParent(parentItemPath);
                if (parentItem)
                {
                    if (nodeMode == 0)
                    {
                        int parentMode = parentItem->text(6).toInt();
                        nodeMode = parentMode;
                    }
                }
                nodeTreeItem *item = new nodeTreeItem(parentItem, nodeName, className, typeName, nodeColor, nodeTip, nodeMode, path, numChildren);
                if (!showNodes)
                {
                    if (!showItem)
                        treeWidget->setItemHidden(item, true);
                }
            }
        }
    }
    else if (type == TABLET_BROWSER_END)
    {

        if (find)
        {
            findItem();
            find = false;
        }
    }
    else if (type == TABLET_BROWSER_CURRENT_NODE)
    {
        char *path;
        tb >> path;
        QString itemPath(path);

        disconnect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(updateSelection()));

        treeWidget->clearSelection();
        selectCBox->setCheckState(Qt::Unchecked);
        nodeTreeItem *currentItem = treeWidget->findParent(itemPath);
        treeWidget->scrollToItem(currentItem);
        treeWidget->setItemSelected(currentItem, true);
        connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(updateSelection()));
    }
    else if (type == TABLET_BROWSER_REMOVE_NODE)
    {
        char *path;
        char *pPath;
        tb >> path;
        tb >> pPath;
        QString itemPath(path);
        QString parentPath(pPath);

        nodeTreeItem *removeItem = treeWidget->findParent(itemPath);
        if (removeItem)
        {
            nodeTreeItem *parentItem = treeWidget->findParent(parentPath);
            if (!parentItem)
                parentItem = (nodeTreeItem *)removeItem->parent();

            if (parentItem)
            {
                //parentItem->removeChild(removeItem);  //only QT 4.3
                int i = parentItem->indexOfChild(removeItem);
                QTreeWidgetItem *item = parentItem->takeChild(i);
                if (item)
                    delete item;
            }
        }
    }
    else if (type == TABLET_BROWSER_PROPERTIES)
    {
        // getProperties
        QTreeWidgetItem *item = treeWidget->currentItem();

        if (item)
        {
            char *path;
            char *pPath;
            int mode, trans;
            tb >> path;
            tb >> pPath;
            tb >> mode;
            tb >> trans;
            QString itemPath(path);
            QString parentPath(pPath);

            itemProps prop;

            if (item->text(8) == itemPath)
            {
                prop.name = item->text(0);
                prop.type = item->text(1);
                prop.mode = mode;
                prop.remove = 0;
                prop.trans = trans;
                prop.numChildren = item->text(4);

                tb >> prop.diffuse[0];
                tb >> prop.diffuse[1];
                tb >> prop.diffuse[2];
                tb >> prop.diffuse[3];

                tb >> prop.specular[0];
                tb >> prop.specular[1];
                tb >> prop.specular[2];
                tb >> prop.specular[3];

                tb >> prop.ambient[0];
                tb >> prop.ambient[1];
                tb >> prop.ambient[2];
                tb >> prop.ambient[3];

                tb >> prop.emissive[0];
                tb >> prop.emissive[1];
                tb >> prop.emissive[2];
                tb >> prop.emissive[3];

                if (prop.type == "MatrixTransform")
                {
                    for (int i = 0; i < 16; ++i)
                    {
                        tb >> prop.matrix[i];
                    }
                }

                propertyDialog->setProperties(prop);
            }

            QTreeWidgetItem *item = treeWidget->currentItem();

            if (is_SimNode(item->text(8).toUtf8().data()))
            {
                if (simu.size() != 0)
                {
                    for (std::vector<QAction *>::iterator iter = simu.begin(); iter != simu.end(); iter++)
                    {
                        treeWidget->removeAction(*iter);
                    }
                    treeWidget->removeAction(cad);
                    delete cad;
                    simu.clear();
                    cad = 0;
                }
                //cout<<endl<<"Current Item"<<item->text ( 8 ).toUtf8().data() <<endl;

                cad = new QAction("View CAD-Data", 0);
                cad->setCheckable(true);
                cad->setChecked(CAD_Status[item->text(8).toUtf8().data()]);
                connect(cad, SIGNAL(triggered()), this, SLOT(viewCad()));
                treeWidget->addAction(cad);

                std::map<std::string, std::vector<std::string> >::iterator i = CAD_SIM_Node.find(item->text(8).toUtf8().data());
                if (i != CAD_SIM_Node.end())
                    for (int index = 0; index < i->second.size(); index++)
                    {
                        QAction *simAct = new QAction(SimPath_SimName[((i->second)[index])].c_str(), 0);
                        simAct->setCheckable(true);
                        simAct->setChecked(SIM_Status[(i->second[index])]);
                        connect(simAct, SIGNAL(triggered()), this, SLOT(viewSim()));
                        treeWidget->addAction(simAct);
                        Sim_Act_Pair[((i->second)[index]).c_str()] = simAct;
                        simu.push_back(simAct);
                    }
            }
            else
            {
                if (simu.size() != 0)
                {
                    for (std::vector<QAction *>::iterator iter = simu.begin(); iter != simu.end(); iter++)
                    {
                        treeWidget->removeAction(*iter);
                    }
                    treeWidget->removeAction(cad);
                    simu.clear();
                    delete cad;
                    cad = 0;
                }
            }
        }
    }
    else if (type == TABLET_TRAVERSED_TEXTURES)
    {
        receivingTextures = false;
        receivedTextures = numItems;
    }
    else if (type == TABLET_NODE_TEXTURES)
    {
        receivingTextures = false;
        receivedTextures = numItems;
        currentTexture = true;
    }
    else if (type == TABLET_NO_TEXTURES)
    {
        receivingTextures = false;
    }
    else if (type == TABLET_TEX_CHANGE)
    {
        int buttonNumber;
        char *currentPath;
        tb >> buttonNumber;
        tb >> currentPath;
        std::string path = std::string(currentPath);

        thread->enqueueGeode(buttonNumber, path);
        propertyDialog->setTextureUpdateBtn(false);
    }
    else if (type == TABLET_TEX_MODE)
    {

        tb >> texNumber;
        tb >> textureModes[texNumber];
        tb >> textureTexGenModes[texNumber];
        tb >> textureIndices[texNumber];
        if (texNumber == propertyDialog->getTexNumber())
        {
            propertyDialog->setTexMode(textureModes[texNumber]);
            propertyDialog->setTexGenMode(textureTexGenModes[texNumber]);
            propertyDialog->setView(textureIndices[texNumber]);
        }
    }
    else if (type == TABLET_TEX)
    {
        int index;
        tb >> height;
        tb >> width;
        tb >> depth;
        tb >> index;
        tb >> dataLength;

        if (depth == 24)
            dataLength = (dataLength * 4) / 3;

        char *sendData = new char[dataLength];

        for (int i = 0; i < dataLength; i++)
        {
            if ((i % 4) == 3)
                if (depth == 24)
                    sendData[i] = 1;
                else
                    tb >> sendData[i];
            else if ((i % 4) == 2)
                tb >> sendData[i - 2];
            else if ((i % 4) == 1)
                tb >> sendData[i];
            else if ((i % 4) == 0)
                tb >> sendData[i + 2];
        }
        QImage image;
        if (dataLength > 0)
        {
            if (depth == 32)
                image = QImage(reinterpret_cast<unsigned char *>(sendData), width, height, QImage::Format_RGB32);
            else
                image = QImage(reinterpret_cast<unsigned char *>(sendData), width, height, QImage::Format_ARGB32);
            image = image.mirrored();
        }
        int num = numItems;
        QString dateTime = QString("%1_%2").arg(QDateTime::currentDateTime().toTime_t()).arg(num);

        QString fileName = texturePluginTempDir + "texture" + dateTime + ".png";
        if (image.save(fileName, "PNG"))
        {
            thread->lock();
            buttonList.append(fileName);
            indexList.push_back(index);
            numItems++;
            //std::cerr << "Button saved : " << fileName << "\n";
            thread->unlock();
        }
        delete[] sendData;
    }
    else if (type == GET_SHADER)
    {
        char *name;
        tb >> name;
        QString shaderName(name);
        propertyDialog->addShader(shaderName);
    }
    else if (type == GET_UNIFORMS)
    {
        char *name, *type, *value, *min, *max, *textureFile;
        tb >> name;
        tb >> type;
        tb >> value;
        tb >> min;
        tb >> max;
        tb >> textureFile;
        QString uniName(name);
        QString uniType(type);
        QString uniValue(value);
        QString uniMin(min);
        QString uniMax(max);
        QString uniTexFile(textureFile);
        propertyDialog->addUniform(uniName, uniType, uniValue, uniMin, uniMax, uniTexFile);
    }
    else if (type == GET_SOURCE)
    {
        char *vertex, *fragment, *geometry, *tessControl, *tessEval;
        tb >> vertex;
        tb >> fragment;
        tb >> geometry;
        tb >> tessControl;
        tb >> tessEval;
        QString vertexS(vertex);
        QString fragmentS(fragment);
        QString geometryS(geometry);
        QString tessControlS(tessControl);
        QString tessEvalS(tessEval);
        propertyDialog->addSource(vertexS, fragmentS, geometryS, tessControlS, tessEvalS);
    }
    else if (type == UPDATE_UNIFORM)
    {
        char *Sname, *Uname, *value, *textureFile;
        tb >> Sname;
        tb >> Uname;
        tb >> value;
        tb >> textureFile;
        QString SName(Sname);
        QString UName(Uname);
        QString Value(value);
        QString STexFile(textureFile);

        propertyDialog->updateUniform(SName, UName, Value, STexFile);
    }
    else if (type == UPDATE_VERTEX)
    {
        char *Sname, *vertex;
        tb >> Sname;
        tb >> vertex;
        QString SName(Sname);
        QString Value(vertex);

        propertyDialog->updateVertex(SName, Value);
    }
    else if (type == UPDATE_TESSCONTROL)
    {
        char *Sname, *tessControl;
        tb >> Sname;
        tb >> tessControl;
        QString SName(Sname);
        QString Value(tessControl);

        propertyDialog->updateTessControl(SName, Value);
    }
    else if (type == UPDATE_TESSEVAL)
    {
        char *Sname, *tessEval;
        tb >> Sname;
        tb >> tessEval;
        QString SName(Sname);
        QString Value(tessEval);

        propertyDialog->updateTessEval(SName, Value);
    }
    else if (type == UPDATE_FRAGMENT)
    {
        char *Sname, *fragment;
        tb >> Sname;
        tb >> fragment;
        QString SName(Sname);
        QString Value(fragment);

        propertyDialog->updateFragment(SName, Value);
    }
    else if (type == UPDATE_GEOMETRY)
    {
        char *Sname, *geometry;
        tb >> Sname;
        tb >> geometry;
        QString SName(Sname);
        QString Value(geometry);

        propertyDialog->updateGeometry(SName, Value);
    }
    else if (type == SET_NUM_VERT)
    {
        char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        QString SName(Sname);
        ;

        propertyDialog->setNumVert(SName, value);
    }
    else if (type == SET_INPUT_TYPE)
    {
        char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        QString SName(Sname);
        ;

        propertyDialog->setInputType(SName, value);
    }
    else if (type == SET_OUTPUT_TYPE)
    {
        char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        QString SName(Sname);
        ;

        propertyDialog->setOutputType(SName, value);
    }

    TUITab::setValue(type, tb);
}

void TUISGBrowserTab::centerObject()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << CENTER_OBJECT;
    TUIMainWindow::getInstance()->send(tb);
}

//gottlieb<
void TUISGBrowserTab::loadFiles()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_LOAD_FILES;
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
        tb << item->text(8).toUtf8().data();
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::viewSim()
{
    QTreeWidgetItem *item = treeWidget->currentItem();
    std::map<std::string, std::vector<std::string> >::iterator iter = CAD_SIM_Node.find(item->text(8).toUtf8().data());
    if (iter != CAD_SIM_Node.end())
    {
        for (int index = 0; index < iter->second.size(); index++)
        {
            if (Sim_Act_Pair[iter->second[index].c_str()]->isChecked())
            {
                TUISGBrowserTab::showhideSimItems(1, iter->second[index].c_str());
                SIM_Status[(iter->second[index])] = true;
            }
            else
            {
                TUISGBrowserTab::showhideSimItems(0, iter->second[index].c_str());
                SIM_Status[(iter->second[index])] = false;
            }
        }
    }
}

void TUISGBrowserTab::viewCad()
{
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (cad->isChecked())
    {
        TUISGBrowserTab::showhideSimItems(1, item->text(8).toUtf8().data());
        CAD_Status[item->text(8).toUtf8().data()] = true;
    }
    else
    {
        TUISGBrowserTab::showhideSimItems(0, item->text(8).toUtf8().data());
        CAD_Status[item->text(8).toUtf8().data()] = false;
    }
}

//>gottlieb

void TUISGBrowserTab::itemChangedSlot(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    if (!current)
    {
        treeWidget->setCurrentItem(previous);
    }
}

void TUISGBrowserTab::handleSetFragment()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString value = propertyDialog->getFragmentValue();
        QByteArray ba1 = value.toUtf8();
        const char *valueF = ba1.data();

        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_FRAGMENT;
        tb << nameS;
        tb << valueF;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleSetGeometry()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString value = propertyDialog->getGeometryValue();
        QByteArray ba1 = value.toUtf8();
        const char *valueG = ba1.data();

        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_GEOMETRY;
        tb << nameS;
        tb << valueG;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleNumVertChanged(int n)
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_NUM_VERT;
        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();
        tb << nameS;
        tb << n;
        TUIMainWindow::getInstance()->send(tb);
    }
}
void TUISGBrowserTab::handleInputTypeChanged(int n)
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_INPUT_TYPE;
        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();
        tb << nameS;

        if (n == 0)
            tb << GL_POINTS;
        if (n == 1)
            tb << GL_LINES;
        if (n == 2)
            tb << MY_GL_LINES_ADJACENCY_EXT;
        if (n == 3)
            tb << MY_GL_TRIANGLES_ADJACENCY_EXT;
        if (n == 4)
            tb << GL_TRIANGLES;
        TUIMainWindow::getInstance()->send(tb);
    }
}
void TUISGBrowserTab::handleOutputTypeChanged(int n)
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_OUTPUT_TYPE;
        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();
        tb << nameS;
        if (n == 0)
            tb << GL_POINTS;
        if (n == 1)
            tb << GL_LINE_STRIP;
        if (n == 2)
            tb << GL_TRIANGLE_STRIP;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleSetVertex()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString value = propertyDialog->getVertexValue();
        QByteArray ba1 = value.toUtf8();
        const char *valueV = ba1.data();

        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_VERTEX;
        tb << nameS;
        tb << valueV;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleSetTessControl()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString value = propertyDialog->getTessControlValue();
        QByteArray ba1 = value.toUtf8();
        const char *valueV = ba1.data();

        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_TESSCONTROL;
        tb << nameS;
        tb << valueV;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleSetTessEval()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString value = propertyDialog->getTessEvalValue();
        QByteArray ba1 = value.toUtf8();
        const char *valueV = ba1.data();

        QByteArray ba2 = Sname.toUtf8();
        const char *nameS = ba2.data();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_TESSEVAL;
        tb << nameS;
        tb << valueV;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::handleSetUniform()
{
    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {
        QString Uname = propertyDialog->getUniformName();
        if (!Uname.isEmpty())
        {
            QString value = propertyDialog->getUniformValue();
            QByteArray ba1 = value.toUtf8();
            const char *valueU = ba1.data();

            QByteArray ba2 = Sname.toUtf8();
            const char *nameS = ba2.data();
            QByteArray ba3 = Uname.toUtf8();
            const char *nameU = ba3.data();

            QString text = propertyDialog->getUniformTextureFile();
            QByteArray ba4 = text.toUtf8();
            const char *textS = ba4.data();

            covise::TokenBuffer tb;
            tb << ID;
            tb << TABLET_BROWSER_PROPERTIES;
            tb << SET_UNIFORM;
            tb << nameS;
            tb << nameU;
            tb << valueU;
            tb << textS;
            TUIMainWindow::getInstance()->send(tb);
        }
    }
}

void TUISGBrowserTab::sendRemoveShaderRequest(QTreeWidgetItem *item)
{
    QString str = item->text(8);
    QByteArray ba = str.toUtf8();
    const char *path = ba.data();

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << REMOVE_SHADER;
    tb << path;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::handleRemoveShader()
{
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
    {
        if ((item->text(0) != "OBJECTS_ROOT") && (item->parent()->text(1) == "Switch"))
        {
            QTreeWidget *tree = item->treeWidget();
            QTreeWidgetItem *top = tree->topLevelItem(0);
            while (top != NULL)
            {
                sendRemoveShaderRequest(top);
                top = tree->itemBelow(top);
            }
        }
        else
            sendRemoveShaderRequest(item);
    }
}

void TUISGBrowserTab::handleStoreShader()
{

    QString Sname = propertyDialog->getShaderName();
    if (!Sname.isEmpty())
    {

        QByteArray ba2 = Sname.toUtf8();
        const char *name = ba2.data();
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << STORE_SHADER;
        tb << name;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::sendSetShaderRequest(QTreeWidgetItem *item, const char *shaderName)
{
    QString str = item->text(8);
    QByteArray ba = str.toUtf8();
    const char *path = ba.data();

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << SET_SHADER;
    tb << path;
    tb << shaderName;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::handleSetShader()
{
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
    {
        QString Sname = propertyDialog->getShaderName();
        if (!Sname.isEmpty())
        {
            QByteArray ba2 = Sname.toUtf8();
            const char *name = ba2.data();

            if ((item->text(0) != "OBJECTS_ROOT") && (item->parent()->text(1) == "Switch"))
            {
                QTreeWidget *tree = item->treeWidget();
                QTreeWidgetItem *top = tree->topLevelItem(0);
                while (top != NULL)
                {
                    sendSetShaderRequest(top, name);
                    top = tree->itemBelow(top);
                }
            }
            else
                sendSetShaderRequest(item, name);
        }
    }
}

void TUISGBrowserTab::handleGetShader()
{
    propertyDialog->clearShaderList();
    propertyDialog->clearUniformList();
    propertyDialog->clearSource();

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << GET_SHADER;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::handleShaderList(QListWidgetItem *item)
{
    QString str = item->text();
    QByteArray ba = str.toUtf8();
    const char *name = ba.data();

    propertyDialog->clearUniformList();
    propertyDialog->clearSource();
    propertyDialog->setShaderName(item->text());

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << GET_UNIFORMS;
    tb << name;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::updateTextureButtons()
{

    if (currentTexture && (receivedTextures == 0))
    {
        // set view
        int index = propertyDialog->getTexNumber();
        propertyDialog->setView(textureIndices[index]);
        currentTexture = false;
    }
    if (receivedTextures > 0)
    {
        //std::cerr << "timeout & receiving\n";
        thread->lock();
        QStringList::Iterator it;
        for (it = buttonList.begin(); it != buttonList.end(); ++it)
        {
            QString fileName = *it;
            //std::cerr << "Button taken : " << fileName << "\n";
            QPixmap map;
            int index = indexList.front();

            if (map.load(fileName))
            {
                propertyDialog->setListWidgetItem(map.scaled(96, 96), fileName, index);
                receivedTextures--;
            }
            indexList.pop_front();
        }
        //std::cerr << "\n";
        buttonList.clear();
        indexList.clear();
        thread->unlock();
    }
    if (!thread->isSending() && receivedTextures == 0)
    {
        propertyDialog->setTextureUpdateBtn(true);
        return;
    }
}

void TUISGBrowserTab::loadTexture()
{
    QString file = QFileDialog::getOpenFileName(frame, tr("get texture file"), textureDir, "*.*");
    QFileInfo info(file);
    if (!file.isEmpty())
    {
        QString newName = currentDir + info.fileName();
        if (newName != file)
        {
            QFile oldFile(file);
            QDataStream inStream(&oldFile);
            QFile newFile(newName);
            QDataStream outStream(&newFile);

            int fileSize = oldFile.size();
            if (oldFile.open(QIODevice::ReadOnly) && oldFile.isOpen() && newFile.open(QIODevice::WriteOnly) && newFile.isOpen())
            {
                char *c = new char[fileSize];
                while (!inStream.atEnd())
                {
                    inStream.readRawData(c, fileSize);
                    outStream.writeRawData(c, fileSize);
                }
                oldFile.close();
                newFile.close();
                //oldFile.remove();
                delete[] c;
            }
        }
        QString indexStr = QString("%1").arg(QDateTime::currentDateTime().toTime_t());
        int index = indexStr.toInt();
        QPixmap map;
        if (map.load(newName))
        {
            propertyDialog->setListWidgetItem(map.scaled(96, 96), newName, index);
        }
    }
}

void TUISGBrowserTab::closeEvent(QCloseEvent *ce)
{
    (void)ce;
    // closing the connection in the destructor should be enough, right?
    //ce->accept();
}

void TUISGBrowserTab::send(covise::TokenBuffer &tb)
{
    if (!getClient())
        return;

    covise::Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    getClient()->send_msg(&m);
}

void TUISGBrowserTab::handleClient(const covise::Message *msg)
{
    covise::TokenBuffer tb(msg);
    switch (msg->type)
    {
    case covise::COVISE_MESSAGE_SOCKET_CLOSED:
    case covise::COVISE_MESSAGE_CLOSE_SOCKET:
    {
        if (!connectionClosed)
            std::cerr << "TUISGBrowserTab: socket closed: ignored" << std::endl;
        connectionClosed = true;
    }
    break;
    case covise::COVISE_MESSAGE_TABLET_UI:
    {
        connectionClosed = false;

        int tablettype;
        tb >> tablettype;
        int ID;

        std::cerr.flush();
        switch (tablettype)
        {
        case TABLET_SET_VALUE:
        {
            int typeInt;
            tb >> typeInt;
            tb >> ID;
            auto type = static_cast<TabletValue>(typeInt);
            this->setValue(type, tb);
        }
        break;
        default:
        {
            std::cerr << "unhandled Message!!" << tablettype << std::endl;
        }
        break;
        }
    }
    break;
    default:
    {
        if (msg->type > 0)
        {
            std::cerr << "unhandled Message!! type=" << msg->type << " - ";
            if (msg->type < covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE)
                std::cerr << covise::covise_msg_types_array[msg->type];
            else
                std::cerr << "UNKNOWN (" << msg->type << " out of range!)";
            std::cerr << std::endl;
        }
    }
    break;
    }
}

void TUISGBrowserTab::handleRemoveTex()
{

    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
    {
        QString str = item->text(8);
        QByteArray ba = str.toUtf8();
        const char *path = ba.data();

        propertyDialog->clearView();
        int index = propertyDialog->getTexNumber();
        textureIndices[index] = -1;

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << REMOVE_TEXTURE;
        tb << path;
        tb << index;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUISGBrowserTab::sendChangeTextureRequest()
{
    int index = propertyDialog->getTexNumber();
    textureModes[index] = propertyDialog->getTexMode();
    textureTexGenModes[index] = propertyDialog->getTexGenMode();
    textureIndices[index] = propertyDialog->getCurrentIndex();

    int buttonNumber = propertyDialog->getCurrentIndex();
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_TEX_CHANGE;
    tb << buttonNumber;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::changeTexMode(int index)
{
    propertyDialog->setTexMode(textureModes[index]);
    propertyDialog->setTexGenMode(textureTexGenModes[index]);
    propertyDialog->setView(textureIndices[index]);
}

void TUISGBrowserTab::changeTexture(int listindex, std::string geode)
{
    QImage image;
    if (image.load(propertyDialog->getFilename(listindex)))
    {
        image = image.mirrored();
        int hasAlpha;
        if (image.format() == QImage::Format_ARGB32)
            hasAlpha = 1;
        else
            hasAlpha = 0;
        int dataLength = (image.height() * image.width() * image.depth()) / 8;
        int texNumber = propertyDialog->getTexNumber();
        const char *currentPath = geode.c_str();
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_TEX_CHANGE;
        tb << texNumber;
        tb << propertyDialog->getTexMode();
        tb << propertyDialog->getTexGenMode();
        tb << hasAlpha;
        tb << image.height();
        tb << image.width();
        tb << image.depth();
        tb << dataLength;
        tb << currentPath;
        tb << listindex;
        tb.addBinary(reinterpret_cast<char *>(image.bits()), dataLength);

        send(tb);
    }
}

covise::Connection *TUISGBrowserTab::getClient()
{
    return TUIMainWindow::getInstance()->toCOVERSG;
}

covise::Connection *TUISGBrowserTab::getServer()
{
    return TUIMainWindow::getInstance()->toCOVERSG;
}

void TUISGBrowserTab::updateTextures()
{
    //  propertyDialog->clearList();
    numItems = 0;

    /*
      // Remove all Files from temp directory
      QDir temp(texturePluginTempDir);
      QStringList fileList = temp.entryList();
      QStringList::Iterator it;
      for(it = fileList.begin();it != fileList.end();++it)
      {
             QFile::remove(texturePluginTempDir+*it);
      }
   */
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_TEX_UPDATE;
    TUIMainWindow::getInstance()->send(tb);
    propertyDialog->setTextureUpdateBtn(false);
    receivingTextures = true;
}

void TUISGBrowserTab::setColor()
{
    QColor color = QColorDialog::getColor(Qt::red, frame);
    if (color.isValid())
    {

        covise::TokenBuffer tb;

        tb << ID;
        tb << TABLET_BROWSER_CLEAR_SELECTION;
        TUIMainWindow::getInstance()->send(tb);

        colorButton->setPalette(QPalette(color));

        ColorR = (float)color.red() / 255.0;
        ColorG = (float)color.green() / 255.0;
        ColorB = (float)color.blue() / 255.0;

        tb.reset();
        tb << ID;
        tb << TABLET_BROWSER_COLOR;
        tb << ColorR;
        tb << ColorG;
        tb << ColorB;
        TUIMainWindow::getInstance()->send(tb);

        updateSelection();
    }
}

void TUISGBrowserTab::changeSelectionMode(int mode)
{
    covise::TokenBuffer tb;

    tb << ID;
    tb << TABLET_BROWSER_CLEAR_SELECTION;
    TUIMainWindow::getInstance()->send(tb);

    tb.reset();
    tb << ID;
    tb << TABLET_BROWSER_WIRE;
    tb << mode;
    TUIMainWindow::getInstance()->send(tb);

    updateSelection();
}

void TUISGBrowserTab::setSelMode(bool mode)
{
    covise::TokenBuffer tb;

    tb << ID;
    tb << TABLET_BROWSER_CLEAR_SELECTION;
    TUIMainWindow::getInstance()->send(tb);

    if (mode)
    {
        selMode = 1;
    }
    else
    {
        selMode = 0;
    }

    tb.reset();
    tb << ID;
    tb << TABLET_BROWSER_SEL_ONOFF;
    tb << selMode;
    TUIMainWindow::getInstance()->send(tb);

    updateSelection();
}

void TUISGBrowserTab::updateScene()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_BROWSER_CLEAR_SELECTION;
    TUIMainWindow::getInstance()->send(tb);

    tb.reset();
    treeWidget->clearSelection();

    root = false;

    tb << ID;
    tb << TABLET_BROWSER_UPDATE;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::updateItem()
{
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
    {
        item->setText(7, "no");
        updateExpand(item);
    }
}

void TUISGBrowserTab::updateExpand(QTreeWidgetItem *item)
{

    covise::TokenBuffer tb;

    if (item->text(7) == "no")
    {

        QString str = item->text(8);
        QByteArray ba = str.toUtf8();
        const char *path = ba.data();

        tb << ID;
        tb << TABLET_BROWSER_EXPAND_UPDATE;
        tb << path;
        TUIMainWindow::getInstance()->send(tb);

        item->setText(7, "expanded");
    }
}

void TUISGBrowserTab::showhideDialog()
{
    if (firsttime)
    {
        Treelayout->addWidget(propertyDialog);
        firsttime = false;
    }
    if (propertyDialog->isVisible())
    {
        propertyDialog->hide();
    }
    else
    {
        propertyDialog->show();
    }
}

void TUISGBrowserTab::handleCloseDialog()
{
    propAct->setChecked(false);
}

void TUISGBrowserTab::itemProperties()
{
    QList<QTreeWidgetItem *> selList;
    selList = treeWidget->selectedItems();
    if (!selList.isEmpty())
    {
        QTreeWidgetItem *item = selList.at(0);

        if (item)
        {
            treeWidget->setCurrentItem(item);
            propertyDialog->clearView();
            propertyDialog->setTextureUpdateBtn(false);
            numItems = 0;

            for (int j = 0; j < 21; j++)
            {
                textureModes[j] = 1;
                textureTexGenModes[j] = 0;
                textureIndices[j] = -1;
            }

            receivingTextures = true;

            covise::TokenBuffer tb;
            QTreeWidgetItem *pI = NULL;
            QString str = item->text(8);
            QByteArray ba = str.toUtf8();
            const char *path = ba.data();
            QByteArray baP;
            const char *parentPath = path;

            pI = item->parent();
            if (pI)
            {
                str = pI->text(8);
                baP = str.toUtf8();
                parentPath = baP.data();
            }

            tb << ID;
            tb << TABLET_BROWSER_PROPERTIES;
            tb << GET_PROPERTIES;
            tb << path;
            tb << parentPath;

            TUIMainWindow::getInstance()->send(tb);
        }
    }
}

void TUISGBrowserTab::handleApply()
{
    itemProps prop = propertyDialog->getProperties();
    int i;
    QTreeWidgetItem *item = treeWidget->currentItem();
    if (item)
    {
        covise::TokenBuffer tb;
        QTreeWidgetItem *pI = NULL;
        QString str = item->text(8);
        QByteArray ba = str.toUtf8();
        const char *path = ba.data();
        const char *parentPath;

        pI = item->parent();
        if (pI)
        {
            str = pI->text(8);
            QByteArray baP = str.toUtf8();
            parentPath = baP.data();
        }
        else
        {
            parentPath = path;
        }

        QString strC = prop.children;
        QByteArray baC = strC.toUtf8();
        const char *children = baC.data();

        tb << ID;
        tb << TABLET_BROWSER_PROPERTIES;
        tb << SET_PROPERTIES;
        tb << path;
        tb << parentPath;
        tb << prop.mode;
        tb << children;
        tb << prop.allChildren;
        tb << prop.remove;
        tb << prop.trans;

        tb << prop.diffuse[0];
        tb << prop.diffuse[1];
        tb << prop.diffuse[2];
        tb << prop.diffuse[3];

        tb << prop.specular[0];
        tb << prop.specular[1];
        tb << prop.specular[2];
        tb << prop.specular[3];

        tb << prop.ambient[0];
        tb << prop.ambient[1];
        tb << prop.ambient[2];
        tb << prop.ambient[3];

        tb << prop.emissive[0];
        tb << prop.emissive[1];
        tb << prop.emissive[2];
        tb << prop.emissive[3];

        for (i = 0; i < 16; ++i)
        {
            tb << prop.matrix[i];
        }

        TUIMainWindow::getInstance()->send(tb);

        if (prop.type == "Switch")
        {
            if (prop.allChildren)
            {
                for (i = 0; i < item->childCount(); i++)
                {
                    if (item->text(3) == "hide")
                        setColorState(item->child(i), true, false);
                    else
                        setColorState(item->child(i), true);
                }
            }
            else if (prop.children != "NOCHANGE")
            {

                restraint->clear();
                restraint->add(children);
                covise::coRestraint res = *restraint;
                for (i = 0; i < item->childCount(); i++)
                {
                    if (res(i + 1))
                    {
                        if (item->text(3) == "hide")
                            setColorState(item->child(i), true, false);
                        else
                            setColorState(item->child(i), true);
                    }
                    else
                    {
                        setColorState(item->child(i), false);
                    }
                }
            }
        }
    }
}

void TUISGBrowserTab::updateSelection()
{
    QList<QTreeWidgetItem *> selList;

    int i;
    QTreeWidgetItem *pI = NULL;

    covise::TokenBuffer tb;

    selList = treeWidget->selectedItems();

    if (!selList.isEmpty())
    {
        tb << ID;
        tb << TABLET_BROWSER_CLEAR_SELECTION;
        TUIMainWindow::getInstance()->send(tb);
    }

    for (i = 0; i < selList.size(); i++)
    {
        covise::TokenBuffer tb2;

        QString str = selList.at(i)->text(8);
        QByteArray ba = str.toUtf8();
        const char *path = ba.data();

        pI = selList.at(i)->parent();
        if (pI)
        {
            str = pI->text(8);
            QByteArray baP = str.toUtf8();
            const char *parentPath = baP.data();

            tb2 << ID;
            tb2 << TABLET_BROWSER_SELECTED_NODE;
            tb2 << path;
            tb2 << parentPath;

            TUIMainWindow::getInstance()->send(tb2);
        }
    }
}

void TUISGBrowserTab::showAllNodes(bool checked)
{
    showNodes = checked;
    updateScene();
}

void TUISGBrowserTab::selectAllNodes(bool checked)
{
    int i;
    bool help;
    help = recursiveSel;
    recursiveSel = true;

    if (checked)
    {
        for (i = 0; i < treeWidget->topLevelItemCount(); i++)
        {
            updateItemState(treeWidget->topLevelItem(i), 1);
        }
    }
    else
        treeWidget->clearSelection();

    recursiveSel = help;
}

void TUISGBrowserTab::setRecursiveSel(bool checked)
{
    recursiveSel = checked;
}

void TUISGBrowserTab::findItemSLOT()
{
    QString str;
    str = findEdit->text();
    QByteArray ba = str.toUtf8();
    const char *name = ba.data();

    find = true;

    treeWidget->clearSelection();

    covise::TokenBuffer tb;

    tb << ID;
    tb << TABLET_BROWSER_CLEAR_SELECTION;
    TUIMainWindow::getInstance()->send(tb);
    tb.reset();

    tb << ID;
    tb << TABLET_BROWSER_FIND;
    tb << name;
    TUIMainWindow::getInstance()->send(tb);
}

void TUISGBrowserTab::findItem()
{
    int i;
    QString str;
    QList<QTreeWidgetItem *> findlist;

    treeWidget->collapseAll();

    str = findEdit->text();
    findlist = treeWidget->findString(str);
    treeWidget->clearSelection();
    for (i = 0; i < findlist.size(); i++)
    {
        treeWidget->setItemSelected(findlist.at(i), true);
    }
    if (findlist.isEmpty())
        findEdit->setText("not found");
    else
        treeWidget->scrollToItem(findlist.at(0));
}

void TUISGBrowserTab::setColorState(QTreeWidgetItem *item, bool show, bool setColor)
{
    int i;
    QColor color;
    QTreeWidgetItem *child;

    if (show)
    {
        if (setColor)
            color.setNamedColor(item->text(2));
        else
            color = QColor(60, 60, 60);

        if (item->text(5) != "switch")
            item->setText(3, "show");
    }
    else
    {
        color = QColor(125, 125, 125);

        if (item->text(5) != "switch")
            item->setText(3, "hide");
    }
    item->setTextColor(0, color);

    for (i = 0; i < item->childCount(); i++)
    {
        child = item->child(i);
        if (!((child->text(5) == "switch") && (child->text(3) == "hide")))
            setColorState(item->child(i), show, setColor);
    }
}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void TUISGBrowserTab::CheckedItems(QString searchStr, bool show)
{
    int i, mode;

    QList<QTreeWidgetItem *> list;
    QTreeWidgetItem *pI = NULL;

    list = treeWidget->findString(searchStr, 0);

    for (i = 0; i < list.size(); i++)
    {
        pI = list.at(i)->parent();

        if (pI)
        {
            if (show)
            {
                mode = TABLET_BROWSER_SHOW_NODE;
                if (list.at(i)->parent()->text(3) == "hide")
                    setColorState(list.at(i), true, false);
                else
                {
                    setColorState(list.at(i), true);
                    pI->setCheckState(i, Qt::Unchecked);
                }
                list.at(i)->setText(3, "show");
            }
            else
            {
                mode = TABLET_BROWSER_HIDE_NODE;
                setColorState(list.at(i), false);
                pI->setCheckState(i, Qt::Unchecked);
                list.at(i)->setText(3, "hide");
            }

            list.at(i)->setText(5, "switch");

            covise::TokenBuffer tb;

            QString str = list.at(i)->text(8);
            QByteArray ba = str.toUtf8();
            const char *path = ba.data();

            str = pI->text(8);
            QByteArray baP = str.toUtf8();
            const char *parentPath = baP.data();

            tb << ID;
            tb << mode;
            tb << path;
            tb << parentPath;
            TUIMainWindow::getInstance()->send(tb);
        }
    }
}

void TUISGBrowserTab::showNode(QTreeWidgetItem *item, bool visible)
{
    QTreeWidgetItem *pI = item->parent();
    if (pI)
    {
        int mode = 0;
        if (visible)
        {
            mode = TABLET_BROWSER_SHOW_NODE;
            if (item->parent()->text(3) == "hide")
                setColorState(item, true, false);
            else
                setColorState(item, true);
            item->setText(3, "show");
        }
        else
        {
            mode = TABLET_BROWSER_HIDE_NODE;
            setColorState(item, false);
            item->setText(3, "hide");
        }

        item->setText(5, "switch");

        covise::TokenBuffer tb;

        QString str = item->text(8);
        QByteArray ba = str.toUtf8();
        const char *path = ba.data();

        str = pI->text(8);
        QByteArray baP = str.toUtf8();
        const char *parentPath = baP.data();

        tb << ID;
        tb << mode;
        tb << path;
        tb << parentPath;
        TUIMainWindow::getInstance()->send(tb);
    }
}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void TUISGBrowserTab::CheckedItems(bool show)
{
    int i;

    QList<QTreeWidgetItem *> list;
    QTreeWidgetItem *pI = NULL;

    list = treeWidget->findString("", 1);

    for (i = 0; i < list.size(); i++)
    {
        QTreeWidgetItem *item = list.at(i);
        showNode(item, show);
    }
}

void TUISGBrowserTab::showCheckedItems()
{
    CheckedItems(true);
}

void TUISGBrowserTab::hideCheckedItems()
{
    CheckedItems(false);
}

void TUISGBrowserTab::showhideCheckedItems()
{
    int i;

    QList<QTreeWidgetItem *> list;
    QTreeWidgetItem *pI = NULL;

    list = treeWidget->findString("", 1);

    for (i = 0; i < list.size(); i++)
    {
        if (list.at(i)->text(3) == "show")
        {

            setColorState(list.at(i), false);
            list.at(i)->setText(3, "hide");
            list.at(i)->setText(5, "switch");
            // send path to SG HIDE!!!
            covise::TokenBuffer tb;

            QString str = list.at(i)->text(8);
            QByteArray ba = str.toUtf8();
            const char *path = ba.data();

            pI = list.at(i)->parent();
            if (pI)
            {
                str = pI->text(8);
                QByteArray baP = str.toUtf8();
                const char *parentPath = baP.data();

                tb << ID;
                tb << TABLET_BROWSER_HIDE_NODE;
                tb << path;
                tb << parentPath;
            }
        }
        else
        {

            if (list.at(i)->parent())
            {
                if (list.at(i)->parent()->text(3) == "hide")
                    setColorState(list.at(i), true, false);
                else
                    setColorState(list.at(i), true);
            }
            else
                setColorState(list.at(i), true);

            list.at(i)->setText(3, "show");
            list.at(i)->setText(5, "switch");
            // send path to SG SHOW!!!
            covise::TokenBuffer tb2;

            QString str = list.at(i)->text(8);
            QByteArray ba = str.toUtf8();
            const char *path = ba.data();

            pI = list.at(i)->parent();
            if (pI)
            {
                str = pI->text(8);
                QByteArray baP = str.toUtf8();
                const char *parentPath = baP.data();

                tb2 << ID;
                tb2 << TABLET_BROWSER_SHOW_NODE;
                tb2 << path;
                tb2 << parentPath;
            }
        }
    }
}

void TUISGBrowserTab::showhideSimItems(int state, const char *nodePath)
{
    QString searchStr = nodePath;
    TUISGBrowserTab::CheckedItems(searchStr, state);
}

void TUISGBrowserTab::updateItemState(QTreeWidgetItem *item, int column)
{
    if (!column)
        selectCBox->setCheckState(Qt::Unchecked);

    if ((item->text(0) != "OBJECTS_ROOT") && (item->parent()->text(1) == "Switch"))
    {

        QTreeWidget *tree = item->treeWidget();
        QTreeWidgetItem *top = tree->topLevelItem(0);
        QTreeWidgetItem *parent = item->parent();
        while (top != NULL)
        {
            if ((top != item) && (parent == top->parent()) && (top->checkState(0) == Qt::Checked))
                top->setCheckState(0, Qt::Unchecked);
            top = tree->itemBelow(top);
        }
    }

    if (recursiveSel)
    {
        for (int j = 0; j < item->childCount(); j++)
        {
            treeWidget->setItemSelected(item->child(j), true);
            updateItemState(item->child(j), column);
        }
    }
    if (item->text(0) != "OBJECTS_ROOT")
    {
        for (int j = 0; j <= item->childCount(); j++)
        {
            if (item->checkState(0) == Qt::Unchecked)
            {
                //hideCheckedItems();
            }
            else
            {
                //showCheckedItems();
            }
        }
    }
}

const char *TUISGBrowserTab::getClassName() const
{
    return "TUISGBrowserTab";
}

bool TUISGBrowserTab::decision(int nodetype, QString &name, QColor &color)
{
    bool show = true;

    switch (nodetype)
    {
    case SG_NODE:
        color = QColor(0, 0, 255);
        name = "NODE";
        show = true;
        break;
    case SG_GEODE:
        color = QColor(0, 0, 0);
        name = "GEODE";
        show = true;
        break;
    case SG_GROUP:
        color = QColor(200, 0, 0);
        name = "GROUP";
        show = true;
        break;
    case SG_CLIP_NODE:
        color = QColor(200, 0, 200);
        name = "CLIP";
        show = true;
        break;
    case SG_CLEAR_NODE:
        color = QColor(0, 200, 200);
        name = "CLEAR";
        show = true;
        break;
    case SG_LIGHT_SOURCE:
        color = QColor(200, 200, 0);
        name = "LIGHT";
        show = true;
        break;
    case SG_TEX_GEN_NODE:
        color = QColor(0, 255, 255);
        name = "TEX";
        show = true;
        break;
    case SG_TRANSFORM:
        color = QColor(0, 200, 0);
        name = "TRANSFORM";
        show = true;
        break;
    case SG_MATRIX_TRANSFORM:
        color = QColor(0, 0, 255);
        name = "MATRIX";
        show = true;
        break;
    case SG_SWITCH:
        color = QColor(0, 255, 0);
        name = "SWITCH ON/OFF";
        show = true;
        break;
    case SG_LOD:
        color = QColor(50, 255, 50);
        name = "LOD";
        show = true;
        break;
    case SG_SIM_NODE:
        color = QColor(0, 75, 0);
        name = "node with simulation data";
        show = true;
        break;
    default:
        color = QColor(255, 0, 0);
        name = "UNKNOWN";
        show = true;
    }

    return show;
}

bool TUISGBrowserTab::new_Sim(std::string nodePath, std::string simPath)
{
    std::map<string, std::vector<string> >::iterator i = CAD_SIM_Node.find(nodePath);
    if (i != CAD_SIM_Node.end())
    {
        for (int index = 0; index < i->second.size(); index++)
        {
            string tmp1 = (i->second)[index].substr(0, (i->second)[index].find_last_of("_"));
            string tmp2 = simPath.substr(0, (simPath).find_last_of("_"));
            //cout <<(i->second)[index]<<" -.-.-.- "<<simPath<<endl;
            if (tmp1 == tmp2)
            {
                //cout<<"name of old/new Simulation: "<<(i->second)[index]<<" --- "<<simPath<<endl;
                std::map<string, QAction *>::iterator k = Sim_Act_Pair.find((i->second)[index]);
                if (k != Sim_Act_Pair.end())
                {
                    QAction *tmp = k->second;
                    Sim_Act_Pair.erase(k);
                    Sim_Act_Pair[simPath] = tmp;
                }
                i->second[index] = simPath;
                return false;
            }
        }
        //cout<<"Simulation "<<simPath<<" added to "<<nodePath<<endl;
        i->second.push_back(simPath);
        return false;
    }

    return (true);
}

bool TUISGBrowserTab::is_SimNode(std::string nodePath)
{
    std::map<string, std::vector<string> >::iterator i = CAD_SIM_Node.find(nodePath);
    if (i != CAD_SIM_Node.end())
        return true;
    return false;
}

/*****************************************************************************
 *
 * Class nodeTreeItem
 *
 *
 * setText(x, QString)
 * 0 = name
 * 1 = className
 * 2 = color.name() (show color for item)
 * 3 = "show" / "hide"
 * 4 = numChildren
 * 5 = "switch" (node ist switch)
 * 6 = nodeMode (state for color)
 * 7 = "expanded" / "no"   (needs to update or not)
 * 8 = path   (node path from root)
 *
 *****************************************************************************/

nodeTreeItem::nodeTreeItem(nodeTree *item, const QString &text, QString className, QString name, QColor color, QString tip, int nodeMode, QString path, int numChildren)
    : QTreeWidgetItem(item)
{
    QString str, pStr, modeStr;
    QColor hideColor = QColor(125, 125, 125);

    setToolTip(0, tip);

    setText(1, className);

    if ((nodeMode == 1) || (nodeMode == 0))
    {
        setTextColor(0, color);
        setText(3, "show");
    }
    else if (nodeMode == 2)
    {
        setTextColor(0, hideColor);
        setText(3, "hide");
    }

    if (text.isEmpty())
        setText(0, name);
    else
        setText(0, text);

    setText(2, color.name());

    setText(4, pStr.setNum(numChildren));

    if (className == "Switch")
        setText(5, "");
    else
        setText(5, "switch");

    setText(6, modeStr.setNum(nodeMode));
    setText(7, "no");
    setText(8, path);
}

nodeTreeItem::nodeTreeItem(nodeTreeItem *item, const QString &text, QString className, QString name, QColor color, QString tip, int nodeMode, QString path, int numChildren)
    : QTreeWidgetItem(item)
{
    QString str, pStr, modeStr;
    setCheckState(0, Qt::Checked);
    QColor hideColor = QColor(125, 125, 125);
    QColor parentColor, isColor;

    setToolTip(0, tip);

    setText(1, className);

    if (item)
    {
        parentColor.setNamedColor(item->text(2));
        isColor = item->textColor(0);
    }

    if ((nodeMode == 1) || (nodeMode == 0))
    {
        if (item)
        {
            if (isColor == parentColor)
                setTextColor(0, color);
            else
                setTextColor(0, isColor);
            setText(3, item->text(3));
        }
        else
        {
            setTextColor(0, color);
            setText(3, "show");
            setCheckState(0, Qt::Checked);
        }
    }
    else if (nodeMode == 2)
    {
        setTextColor(0, hideColor);
        setText(3, "hide");
        setCheckState(0, Qt::Unchecked);
    }

    if (text.isEmpty())
        setText(0, name);
    else
        setText(0, text);

    setText(2, color.name());

    setText(4, pStr.setNum(numChildren));

    if (className == "Switch")
        setText(5, "");
    else
        setText(5, "switch");

    setText(6, modeStr.setNum(nodeMode));
    setText(7, "no");
    setText(8, path);

    setFlags(flags() | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
}

nodeTreeItem::~nodeTreeItem()
{
}

// see http:://stackoverflow.com/a/32403843
void nodeTreeItem::setData(int column, int role, const QVariant &value)
{
    const bool isCheckChange = column==0
        && role==Qt::CheckStateRole
        && data(column, role).isValid()
        && checkState(0) != value;
    QTreeWidgetItem::setData(column, role, value);
    if (isCheckChange)
    {
        nodeTree *tree = static_cast<nodeTree *>(treeWidget());
		if(tree)
		{
			emit tree->itemCheckStateChanged(this, checkState(0) == Qt::Checked);
		}
    }
}

/*****************************************************************************
 *
 *  Class nodeTree
 *
 *****************************************************************************/

nodeTree::nodeTree(QWidget *parent)
    : QTreeWidget(parent)
{
    setRootIsDecorated(true);
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    //setSelectionBehavior(QAbstractItemView::SelectItems);
}

nodeTree::~nodeTree()
{
}

void nodeTree::init()
{
    setColumnCount(1);
    setHeaderLabel("Data Objects");
}

//------------------------------------------------------------------------
// get the (parent) tree item for a given QString
//------------------------------------------------------------------------
nodeTreeItem *nodeTree::findParent(QString parentStr)
{
    // find item (recursive loop)
    nodeTreeItem *it = NULL;
    QTreeWidgetItem *top;

    for (int i = 0; i < topLevelItemCount(); i++)
    {
        top = topLevelItem(i);
        QString str = top->text(8);

        if (str == parentStr)
            return static_cast<nodeTreeItem *>(top);

        it = search(top, parentStr);
        if (it != NULL)
            return it;
    }

    return NULL;
}

//------------------------------------------------------------------------
// search tree items for a certain QString
//------------------------------------------------------------------------
nodeTreeItem *nodeTree::search(QTreeWidgetItem *item, QString parent)
{
    QTreeWidgetItem *child = NULL;
    for (int i = 0; i < item->childCount(); i++)
    {
        child = item->child(i);
        QString str = child->text(8);

        if (str == parent)
            return static_cast<nodeTreeItem *>(child);

        else
        {
            nodeTreeItem *it = search(child, parent);
            if (it != NULL)
                return it;
        }
    }

    return NULL;
}

//------------------------------------------------------------------------
// get tree items for a given QString / find selected items
//------------------------------------------------------------------------
QList<QTreeWidgetItem *> nodeTree::findString(QString searchStr, int mode)
{
    int i;
    // find item (recursive loop)
    QTreeWidget *tree;
    QList<QTreeWidgetItem *> toplist;
    QList<QTreeWidgetItem *> childlist;
    QTreeWidgetItem *top;
    bool parentState = false;

    for (i = 0; i < topLevelItemCount(); i++)
    {
        top = topLevelItem(i);
        if (mode)
        {
            parentState = top->isSelected();
            if (parentState)
                toplist.append(top);
        }
        else
        {
            if (top->text(8).contains(searchStr))
                toplist.append(top);
        }

        childlist = searchString(top, searchStr, parentState, mode);

        if (!mode)
            if (!childlist.isEmpty())
            {
                tree = top->treeWidget();
                tree->expandItem(top);
            }

        toplist += childlist;
    }

    return toplist;
}

//------------------------------------------------------------------------
// search tree items for a certain QString / search selected items
//------------------------------------------------------------------------
QList<QTreeWidgetItem *> nodeTree::searchString(QTreeWidgetItem *item, QString searchStr, bool parentState, int mode)
{
    int i;
    QTreeWidget *tree;
    QList<QTreeWidgetItem *> childlist;
    QList<QTreeWidgetItem *> childchildlist;
    bool newParentState = false;

    QTreeWidgetItem *child;

    for (i = 0; i < item->childCount(); i++)
    {
        child = item->child(i);

        if (mode)
        {

            newParentState = child->isSelected();
            if (newParentState)
            {
                if (newParentState != parentState)
                    childlist.append(child);
                else if (child->text(5) == "switch")
                    childlist.append(child);
            }
        }
        else
        {
            if (child->text(8).contains(searchStr))
                childlist.append(child);
        }

        childchildlist = searchString(child, searchStr, newParentState, mode);

        if (!mode)
            if (!childchildlist.isEmpty())
            {
                tree = child->treeWidget();
                tree->expandItem(child);
            }

        childlist += childchildlist;
    }

    return childlist;
}

SGTextureThread::SGTextureThread(TUISGBrowserTab *tab)
{
    running = true;
    this->tab = tab;
}

void SGTextureThread::run()
{

    for (;;)
    {
        lock();
        bool r = running;
        unlock();
        if (!r)
            break;

        //std::cerr << "thread runs \n " ;

        if (isSending())
        {
            //std::cerr << "sending texture \n" ;
            lock();
            int index = buttonQueue.front();
            std::string geode = geodeQueue.front();
            buttonQueue.pop();
            geodeQueue.pop();
            unlock();
            tab->changeTexture(index, geode);
        }
        else if (tab->getClient()->check_for_input())
        {
            covise::Message msg;
            if (tab->getClient()->recv_msg(&msg))
            {
                if (tab->isReceivingTextures())
                    tab->handleClient(&msg);
                else
                    std::cerr << "SGTextureThread: received unexpected message" << std::endl;
            }
        }
        else
        {
            //std::cerr << "sleep \n" ;
            usleep(25000);
        }
    }

    //std::cerr << "SGTextureThread: finished" << std::endl;
}

void SGTextureThread::enqueueGeode(int number, std::string geode)
{
    lock();
    buttonQueue.push(number);
    geodeQueue.push(geode);
    unlock();
}

void SGTextureThread::setButtonNumber(int number)
{
    buttonNumber = number;
}

bool SGTextureThread::isSending()
{
    lock();
    bool ret = !buttonQueue.empty();
    unlock();
    return ret;
}

void SGTextureThread::terminateTextureThread()
{
    lock();
    running = false;
    unlock();
}

void SGTextureThread::lock()
{
    m_mutex.lock();
}

void SGTextureThread::unlock()
{
    m_mutex.unlock();
}
