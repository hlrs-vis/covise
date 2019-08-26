/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream>
#include <sstream>
#include <colormapListWidget.h>
//Added by qt3to4:
#include <QPixmap>
#include <QResizeEvent>
#include <QMoveEvent>
#include <QCloseEvent>
#include <QMenu>
#include <QMainWindow>
#include <QDockWidget>
#include <QListWidget>
#include <QListView>
#include <QTreeView>
using namespace std;

#include "qstring.h"
#include "qlayout.h"
#include "qtabwidget.h"
#include "qsplitter.h"
#include "qlineedit.h"
#include "qfont.h"
#include "qmenubar.h"
#include "qpushbutton.h"
#include "qcolordialog.h"
#include "qmessagebox.h"
#include "qstylefactory.h"
#include "qapplication.h"
#include "qpainter.h"
#include "qpixmap.h"
//
// Inventor stuff
//
#include <Inventor/actions/SoBoxHighlightRenderAction.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/Qt/editors/SoQtColorEditor.h>
#include <Inventor/nodes/SoSelection.h>

// debug stuff (local use)
//
#include <covise/covise.h>
#include <covise/covise_appproc.h>
#include <covise/covise_msg.h>
#include <net/covise_host.h>
#include <covise/Covise_Util.h>
#include <config/CoviseConfig.h>

#include "InvMain.h"
#include "InvMsgManager.h"
#include "InvCommunicator.h"
#include "InvObjectManager.h"
#include "InvSequencer.h"
#include "InvViewer.h"
#include "InvClipPlaneEditor.h"

#include "XPM/covise.xpm"
#include "XPM/checker.xpm"

InvMain *renderer = NULL;

//======================================================================

InvMain::InvMain(int argc, char *argv[])
    : QMainWindow()
{
    setWindowTitle("application main window");
    setAttribute(Qt::WA_DeleteOnClose);

    if ((argc < 7) || (argc > 8))
    {
        if (argc == 2 && 0 == strcmp(argv[1], "-d"))
        {
            set_module_description("Qt/Coin3D (Inventor) Renderer");
            add_port(INPUT_PORT,
                     "RenderData",
                     "Geometry|Points|"
                     "Text_Iv|UnstructuredGrid|RectilinearGrid|"
                     "StructuredGrid|UniformGrid|Polygons|TriangleStrips|"
                     "Lines|Spheres|Texture",
                     "render geometry or Inventor file");

            add_port(PARIN,
                     "AnnotationString",
                     "String",
                     "Annotation descr. string");

            set_port_default("AnnotationString", "empty");

            printDesc(argv[0]);
        }
        else
            cerr << "Application Module with inappropriate arguments called\n";
        exit(0);
    }

    renderer = this;
    synced = SYNC_LOOSE;
    rendererPropBlending = 0;
    rendererProp = 0;
    rendererProperties = 0;
    iCaptureWindow = 0;
    m_iShowWindowDecorations = 1;
    m_iShowFullSizeWindow = 0;

    //
    // parse arguments and store them in info class
    //
    m_name = proc_name = argv[0];
    port = atoi(argv[1]);
    h_name = host = argv[2];
    proc_id = atoi(argv[3]);
    instance = argv[4];
    //
    // contact controller
    //
    appmod = new ApplicationProcess(argv[0], argc, argv);

    set_module_description("Qt/Coin3D (Inventor) Renderer");

    // INPUT PORT
    add_port(INPUT_PORT,
             "RenderData",
             "Geometry|Points|"
             "Text_Iv|UnstructuredGrid|RectilinearGrid|"
             "StructuredGrid|Polygons|TriangleStrips|"
             "Lines|Spheres|Texture",
             "render geometry or Inventor file");

    // annotation parameter
    add_port(PARIN,
             "AnnotationString",
             "String",
             "Annotation descr. string");

    set_port_default("AnnotationString", "empty");
    set_port_immediate("AnnotationString", 1);

    QStringList tmplist;
    QString tmp;

    QString msg = get_description_message();
    QByteArray ba = msg.toLatin1();

    Message message{ COVISE_MESSAGE_FINISHED , DataHandle{ba.data() ,  ba.length() + 1, false} };
    renderer->appmod->send_ctl_msg(&message);

    m_username = "me";
    backgroundColorEditor = NULL;

    // get username from Controller
    hostname = appmod->get_hostname();
    modId = appmod->get_id();

    tmplist << "USERNAME" << hostname << m_name << instance << QString().setNum(modId);
    tmp = tmplist.join("\n");
    ba = tmp.toLatin1();
    message.data = DataHandle(ba.data(), ba.length() + 1, false);
    message.type = COVISE_MESSAGE_UI;
    appmod->send_ctl_msg(&message);
    tmplist.clear();

    print_comment(__LINE__, __FILE__, "Renderer Process succeeded");

    //
    // create a render manager
    //
    render_name = argv[0];
    render_name.append("_");
    render_name.append(argv[4]);
    render_name.append("@");
    QString tmphostname = QString::fromStdString(Host::lookupHostname(hostname.toLatin1()));
    render_name.append(tmphostname.section('.', 0, 0));
    setWindowTitle(render_name);

// Initializes SoQt library (and implicitly also the Coin and Qt
// libraries). Returns a top-level / shell Qt window to use.
//shell = SoQt::init(argc, argv, argv[0]);

#if !defined(_WIN32) && !defined(__APPLE__)
    // set the correct display
    if (getenv("DISPLAY") == NULL)
        setenv("DISPLAY", ":0", 0);

    tp_rate = 0.02;
    tp_lasttime = cam_lasttime = 0;
#endif

    // set a proper font & layout
    boldfont.setWeight(QFont::Bold);
    boldfont.setItalic(true);

    //
    // start an object manager & message manager for the renderer
    //
    om = new InvObjectManager();
    mm = new InvMsgManager();
    cm = new InvCommunicator();

    //
    // create the menubar and the toolbar
    // create a main layout
    //

    // yac/covise configuration environment
    coConfigGroup *mapConfig = new coConfigGroup("QtMapEditor");
    mapConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "mapqt.xml", "local", true);
    coConfig::getInstance()->addConfig(mapConfig);

    std::string style = coCoviseConfig::getEntry("System.UserInterface.QtStyle");
    if (!style.empty())
    {
        QStyle *s = QStyleFactory::create(style.c_str());
        if (s)
            QApplication::setStyle(s);
    }

    // read style from mapqt.xml
    renderConfig = new coConfigGroup("Renderer");
    renderConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "renderer.xml", "local", true);
    coConfig::getInstance()->addConfig(renderConfig);

    makeLayout();
    createMenubar();

    // set config parameter
    float rr = (renderConfig->getValue("red", "Renderer.BackgroundColor")).toFloat();
    float gg = (renderConfig->getValue("green", "Renderer.BackgroundColor")).toFloat();
    float bb = (renderConfig->getValue("blue", "Renderer.BackgroundColor")).toFloat();
    viewer->setBackgroundColor(SbColor(rr, gg, bb));

    int xpos = (renderConfig->getValue("ya", "Renderer.Windows")).toInt();
    int ypos = (renderConfig->getValue("ya", "Renderer.Windows")).toInt();
    int width = (renderConfig->getValue("width", "Renderer.Windows")).toInt();
    int height = (renderConfig->getValue("height", "Renderer.Windows")).toInt();
    move(xpos, ypos);
    resize(width, height);

    // set the logo

    setWindowIcon(QPixmap(logo));

    //
    // show the all widgets
    //
    show();
}

void InvMain::makeLayout()
{

    // create a docking window for several lists on the right side

    QDockWidget *dw = new QDockWidget(this);
    dw->setWindowTitle("Renderer Listbox ");
    /*qt3 dw->setFeatures(Qt::Horizontal);
   dw->setCloseMode(QDockWidget::Docked);
   dw->setResizeEnabled(true);*/

    listTabs = new QTabWidget(dw);
    listTabs->setMinimumWidth(250);

    /*colorListBox = new QListView(listTabs);
	  colorStringList = new QStringList();
      colorListModel = new QStringListModel(*colorStringList, NULL);
	  colorListBox->setModel(colorListModel);*/
    colorListWidget = new QListWidget(listTabs);
    colorListWidget->setViewMode(QListView::IconMode);
    colorListWidget->setIconSize(QSize(200,300));

    treeWidget = new objTreeWidget(this, listTabs);
    treeWidget->show();

    listTabs->addTab(treeWidget, "Object Tree");
    listTabs->addTab(colorListWidget, "ColorMap List");

    dw->setWidget(listTabs);
    addDockWidget(Qt::RightDockWidgetArea, dw, Qt::Horizontal);

    // make a central widget window for the render area

    QWidget *main = new QWidget(this);
    main->setMinimumSize(720, 574); // will give PAL size images
    viewer = new InvViewer(main);
    setCentralWidget(main);

    // create the lower docking window for sequencer

    dw = new QDockWidget(this);
    addDockWidget(Qt::BottomDockWidgetArea, dw, Qt::Vertical);
    dw->setWindowTitle("Renderer Sequencer");
    // qt3 dw->setHorizontallyStretchable(true);
    // qt3 dw->setResizeEnabled(true);
    sequencer = new InvSequencer(dw);
    dw->setWidget(sequencer);
    dw->hide();

    // set the view all mode for the renderer
    viewer->viewAll();

    //connect( colorListBox->selectionModel(), SIGNAL(selectionChanged ( QItemSelection,QItemSelection )),
    //         this, SLOT(colorSelectionChanged ( QItemSelection,QItemSelection )) );
}

//------------------------------------------------------------------------
// create all stuff for the menubar
//------------------------------------------------------------------------
void InvMain::createMenubar()
{

    // File Menu

    /*file = new QPopupMenu( this );
   menuBar()->insertItem( "&File",        file );

   fid[0] = file->insertItem("Save", 	    this, SLOT(file1()),  ALT+Key_S );
   fid[1] = file->insertItem("Save As",    this, SLOT(file1()) );
   fid[2] = file->insertItem("Snap",       this, SLOT(file1()),  CTRL+Key_S );
   fid[3] = file->insertItem("Snap All",   this, SLOT(file1()) );
   fid[4] = file->insertItem("Copy View",  this, SLOT(file1()) );
   fid[5] = file->insertItem("Print",      this, SLOT(file1()),  ALT+Key_P );
   file->insertSeparator();
   fid[6] = file->insertItem("Read Camera Env...",    this, SLOT(file1())	);
   fid[7] = file->insertItem("Save Camera Env...",    this, SLOT(file1())	);
   */

    //===============================================================

    //sync = new QMenu( this );
    sync = menuBar()->addMenu(tr("&Sync"));

    NoCouplingAction = new QAction("No Coupling", this);
    MasterSlaveAction = new QAction("Master/Slave", this);
    TightCouplingAction = new QAction("Tight Coupling", this);
    NoCouplingAction->setCheckable(true);
    NoCouplingAction->setChecked(false);
    MasterSlaveAction->setCheckable(true);
    MasterSlaveAction->setChecked(true);
    TightCouplingAction->setCheckable(true);
    TightCouplingAction->setChecked(false);
    connect(NoCouplingAction, SIGNAL(triggered(bool)), this, SLOT(sendSyncMode(bool)));
    connect(MasterSlaveAction, SIGNAL(triggered(bool)), this, SLOT(sendSyncMode(bool)));
    connect(TightCouplingAction, SIGNAL(triggered(bool)), this, SLOT(sendSyncMode(bool)));

    sync->addAction(NoCouplingAction);
    sync->addAction(MasterSlaveAction);
    sync->addAction(TightCouplingAction);

    /*connect( sync, SIGNAL(activated(int)),
      this, SLOT(sendSyncMode(int)) );*/

    //===============================================================

    //renderer_props = new QMenu( this );
    renderer_props = menuBar()->addMenu("&Renderer");

    BBAction[0] = new QAction("Billboards: manual on the CPU", this);
    BBAction[1] = new QAction("Billboards: ARB point sprites", this);
    BBAction[2] = new QAction("Billboards: CG vertex shader", this);
    BBAction[3] = new QAction("Billboards: Blending On/Off", this);
    BBAction[0]->setCheckable(true);
    BBAction[1]->setCheckable(true);
    BBAction[2]->setCheckable(true);
    BBAction[3]->setCheckable(true);
    BBAction[0]->setChecked(true);
    BBAction[1]->setChecked(false);
    BBAction[2]->setChecked(false);
    BBAction[3]->setChecked(false);
    renderer_props->addAction(BBAction[0]);
    renderer_props->addAction(BBAction[1]);
    renderer_props->addAction(BBAction[2]);
    renderer_props->addAction(BBAction[3]);

    renderer_props->insertSeparator(BBAction[3]);

    CaptureAction = new QAction("Capture Render Window On/Off", this);
    SnapshotAction = new QAction("Snapshot", this);

    CaptureAction->setCheckable(true);
    CaptureAction->setChecked(true);
    renderer_props->addAction(CaptureAction);
    renderer_props->addAction(SnapshotAction);

    connect(BBAction[0], SIGNAL(triggered(bool)), this, SLOT(rendererPropMode(bool)));
    connect(BBAction[1], SIGNAL(triggered(bool)), this, SLOT(rendererPropMode(bool)));
    connect(BBAction[2], SIGNAL(triggered(bool)), this, SLOT(rendererPropMode(bool)));
    connect(BBAction[3], SIGNAL(triggered(bool)), this, SLOT(rendererPropMode(bool)));
    connect(CaptureAction, SIGNAL(triggered(bool)), this, SLOT(setCapture(bool)));
    connect(SnapshotAction, SIGNAL(triggered(bool)), this, SLOT(doSnap(bool)));

    //===============================================================

    //viewing = new Q3PopupMenu( this );
    viewing = menuBar()->addMenu("&Viewing");
    EBAction = new QAction("Edit Background Color ...", this);

    SCAction = new QAction("Show Coordinate Axis", this);
    CPAction = new QAction("Clipping Plane", this);
    SWAction = new QAction("Show Window Decorations", this);
    SFAction = new QAction("Show Full Size Window", this);

    viewing->addAction(EBAction);
    viewing->addAction(SCAction);
    viewing->addAction(CPAction);
    viewing->insertSeparator(CPAction);
    viewing->addAction(SWAction);
    viewing->addAction(SFAction);

    SCAction->setCheckable(true);
    CPAction->setCheckable(true);
    SWAction->setCheckable(true);
    SFAction->setCheckable(true);
    SCAction->setChecked(renderConfig->isOn("Renderer.ShowAxis", "true"));
    CPAction->setChecked(false);
    SWAction->setChecked(true);
    SFAction->setChecked(false);

    connect(EBAction, SIGNAL(triggered(bool)), this, SLOT(doEditBackground(bool)));
    connect(SCAction, SIGNAL(triggered(bool)), this, SLOT(showCoordinateAxis(bool)));
    connect(CPAction, SIGNAL(triggered(bool)), this, SLOT(clippingPlane(bool)));
    connect(SWAction, SIGNAL(triggered(bool)), this, SLOT(showWindowDecoration(bool)));
    connect(SFAction, SIGNAL(triggered(bool)), this, SLOT(showFullSizeWindow(bool)));

    //connect( viewing, SIGNAL(activated(int)),
    //   this, SLOT(viewingMode(int)) );

    //===============================================================

    //edit = new Q3PopupMenu( this );
    edit = menuBar()->addMenu("&Editors");

    MaterialEditorAction = new QAction("Material Editor...", this);
    ColorEditorAction = new QAction("Color Editor...", this);
    ObjectTransformAction = new QAction("Object Transform...", this);
    PartsAction = new QAction("Parts ...", this);
    SnapHandleToAxisAction = new QAction("Snap handle to axis", this);
    FreeHandleMotionAction = new QAction("Free handle motion", this);
    NumericClipPlaneAction = new QAction("Numeric clip plane...", this);

    SnapHandleToAxisAction->setCheckable(true);
    SnapHandleToAxisAction->setChecked(false);
    FreeHandleMotionAction->setCheckable(true);
    FreeHandleMotionAction->setChecked(false);

    edit->addAction(MaterialEditorAction);
    edit->addAction(ColorEditorAction);
    edit->addAction(ObjectTransformAction);
    edit->addAction(PartsAction);
    edit->addAction(SnapHandleToAxisAction);
    edit->addAction(FreeHandleMotionAction);
    edit->addAction(NumericClipPlaneAction);

    connect(MaterialEditorAction, SIGNAL(triggered(bool)), this, SLOT(doMaterialEditor(bool)));
    connect(ColorEditorAction, SIGNAL(triggered(bool)), this, SLOT(doColorEditor(bool)));
    connect(ObjectTransformAction, SIGNAL(triggered(bool)), this, SLOT(doObjectTransform(bool)));
    connect(PartsAction, SIGNAL(triggered(bool)), this, SLOT(doParts(bool)));
    connect(SnapHandleToAxisAction, SIGNAL(triggered(bool)), this, SLOT(doSnapHandleToAxis(bool)));
    connect(FreeHandleMotionAction, SIGNAL(triggered(bool)), this, SLOT(doFreeHandleMotion(bool)));
    connect(NumericClipPlaneAction, SIGNAL(triggered(bool)), this, SLOT(doNumericClipPlane(bool)));

    //===============================================================

    //manip = new Q3PopupMenu( this );
    manip = menuBar()->addMenu("&Manips");

    mid[0] = new QAction("Trackball", this);
    mid[1] = new QAction("HandleBox", this);
    mid[2] = new QAction("Jack", this);
    mid[3] = new QAction("Centerball", this);
    mid[4] = new QAction("TransformBox", this);
    mid[5] = new QAction("TabBox", this);
    mid[6] = new QAction("None", this);
    manip->addAction(mid[0]);
    manip->addAction(mid[1]);
    manip->addAction(mid[2]);
    manip->addAction(mid[3]);
    manip->addAction(mid[4]);
    manip->addAction(mid[5]);
    viewing->insertSeparator(mid[5]);
    manip->addAction(mid[6]);

    for (unsigned int i = 0; i < 6; i++)
    {
        mid[i]->setCheckable(true);
        mid[i]->setChecked(false);
    }

    mid[6]->setChecked(true);

    connect(mid[0], SIGNAL(triggered(bool)), this, SLOT(ManipT(bool)));
    connect(mid[1], SIGNAL(triggered(bool)), this, SLOT(ManipH(bool)));
    connect(mid[2], SIGNAL(triggered(bool)), this, SLOT(ManipJ(bool)));
    connect(mid[3], SIGNAL(triggered(bool)), this, SLOT(ManipC(bool)));
    connect(mid[4], SIGNAL(triggered(bool)), this, SLOT(ManipTF(bool)));
    connect(mid[5], SIGNAL(triggered(bool)), this, SLOT(ManipTB(bool)));
    connect(mid[6], SIGNAL(triggered(bool)), this, SLOT(ManipNone(bool)));

    menuBar()->setFont(boldfont);
}

void InvMain::set_module_description(QString descr)
{
    module_description = descr;
}

void InvMain::add_port(enum appl_port_type type, QString name)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (!port_name[i].isNull())
            i++;
        port_type[i] = type;
        port_name[i] = name;
        port_required[i] = 1;
        port_immediate[i] = 0;
    }

    else
    {
        QByteArray ba = name.toLatin1();
        cerr << "wrong description type in add_port " << (const char *)ba << "\n";
        return;
    }
}

void InvMain::add_port(enum appl_port_type type, QString name, QString dt, QString descr)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (!port_name[i].isNull())
            i++;
        port_type[i] = type;
        port_name[i] = name;
        port_datatype[i] = dt;
        port_required[i] = 1;
        port_immediate[i] = 0;
        port_description[i] = descr;
    }
    else
    {
        cerr << "wrong description type in add_port " << (const char *)name.toLatin1() << "\n";
        return;
    }
}

void InvMain::set_port_description(QString name, QString descr)
{
    int i = 0;
    while (!port_name[i].isNull())
    {
        if (port_name[i] == name)
            break;
        i++;
    }
    if (port_name[i].isNull())
    {
        cerr << "wrong portname " << (const char *)name.toLatin1() << " in set_port_description\n";
        return;
    }
    port_description[i] = descr;
}

void InvMain::set_port_default(QString name, QString def)
{
    int i = 0;
    while (!port_name[i].isNull())
    {
        if (port_name[i] == name)
            break;
        i++;
    }
    if (port_name[i].isNull())
    {
        cerr << "wrong portname " << (const char *)name.toLatin1() << " in set_port_default\n";
        return;
    }

    if (port_type[i] != PARIN && port_type[i] != PAROUT)
    {
        cerr << "wrong port type in set_port_default " << (const char *)name.toLatin1() << "\n";
        return;
    }
    port_default[i] = def;
}

void InvMain::set_port_datatype(QString name, QString dt)
{
    int i = 0;
    while (!port_name[i].isNull())
    {
        if (port_name[i] == name)
            break;
        i++;
    }
    if (port_name[i].isNull())
    {
        cerr << "wrong portname " << (const char *)name.toLatin1() << " in set_port_datatype\n";
        return;
    }
    port_datatype[i] = dt;
}

void InvMain::set_port_required(QString name, int req)
{
    int i = 0;
    while (!port_name[i].isNull())
    {
        if (port_name[i] == name)
            break;
        i++;
    }
    if (port_name[i].isNull())
    {
        cerr << "wrong portname " << (const char *)name.toLatin1() << " in set_port_required\n";
        return;
    }
    if (port_type[i] != INPUT_PORT)
    {
        cerr << "wrong port type in set_port_required " << (const char *)name.toLatin1() << "\n";
        return;
    }
    port_required[i] = req;
}

void InvMain::set_port_immediate(QString name, int imm)
{
    int i = 0;
    while (!port_name[i].isNull())
    {
        if (port_name[i] == name)
            break;
        i++;
    }
    if (port_name[i].isNull())
    {
        cerr << "wrong portname " << (const char *)name.toLatin1() << " in set_port_immediate\n";
        return;
    }
    if (port_type[i] != PARIN)
    {
        cerr << "wrong port type in set_port_immediate " << (const char *)name.toLatin1() << "\n";
        return;
    }
    port_immediate[i] = imm;
}

QString InvMain::get_description_message()
{
    QString msg;
    msg += "DESC\n";
    msg += m_name;
    msg += "\n";
    msg += h_name;
    msg += "\n";
    if (!module_description.isNull())
        msg += module_description;
    else
        msg += m_name;
    msg += "\n";

    int i = 0, ninput = 0, noutput = 0, nparin = 0, nparout = 0;
    while (!port_name[i].isNull())
    {
        switch (port_type[i])
        {
        case INPUT_PORT:
            ninput++;
            break;
        case OUTPUT_PORT:
            noutput++;
            break;
        case PARIN:
            nparin++;
            break;
        case PAROUT:
            nparout++;
            break;
        default:
            break;
        }
        i++;
    }
    msg += QString::number(ninput); // number of parameters
    msg += "\n";
    msg += QString::number(noutput);
    msg += "\n";
    msg += QString::number(nparin);
    msg += "\n";
    msg += QString::number(nparout);
    msg += "\n";
    i = 0; // INPUT ports
    while (!port_name[i].isNull())
    {
        if (port_type[i] == INPUT_PORT)
        {
            msg += port_name[i];
            msg += "\n";
            if (port_datatype[i].isNull())
            {
                cerr << "no datatype for port " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_datatype[i];
            msg += "\n";
            if (!port_description[i].isNull())
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += "\n";
            if (port_required[i])
                msg += "req\n";
            else
                msg += "opt\n";
        }
        i++;
    }

    i = 0; // OUTPUT ports
    while (!port_name[i].isNull())
    {
        if (port_type[i] == OUTPUT_PORT)
        {
            msg += port_name[i];
            msg += "\n";
            if (port_datatype[i].isNull())
            {
                cerr << "no datatype for port " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_datatype[i];
            msg += "\n";
            if (port_description[i].isNull())
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += "\n";
            if (!port_dependency[i].isNull())
            {
                msg += port_dependency[i];
                msg += "\n";
            }
            else
                msg += "default\n";
        }
        i++;
    }

    i = 0; // PARIN ports
    while (!port_name[i].isNull())
    {
        if (port_type[i] == PARIN)
        {
            msg += port_name[i];
            msg += "\n";
            if (port_datatype[i].isNull())
            {
                cerr << "no datatype for port " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_datatype[i];
            msg += "\n";
            if (port_description[i].isNull())
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += "\n";
            if (port_default[i].isNull())
            {
                cerr << "no default value for parameter " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_default[i];
            msg += "\n";
            if (port_immediate[i])
                msg += "IMM\n";
            else
                msg += "START\n";
        }
        i++;
    }

    i = 0; // PAROUT ports
    while (!port_name[i].isNull())
    {
        if (port_type[i] == PAROUT)
        {
            msg += port_name[i];
            msg += "\n";
            if (port_datatype[i].isNull())
            {
                cerr << "no datatype for port " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_datatype[i];
            msg += "\n";
            if (port_description[i].isNull())
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += "\n";
            if (port_default[i].isNull())
            {
                cerr << "no default value for parameter " << (const char *)port_name[i].toLatin1() << "\n";
            }
            msg += port_default[i];
            msg += "\n";
        }
        i++;
    }

    return msg;
}

void InvMain::printDesc(const char *callname)
{
    // strip leading path from module name
    const char *modName = strrchr(callname, '/');
    if (modName)
        modName++;
    else
        modName = callname;

    cout << "Module:      \"" << modName << "\"" << std::endl;
    cout << "Desc:        \"" << (const char *)module_description.toLatin1() << "\"" << std::endl;

    int i, numItems;

    // count parameters
    numItems = 0;
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == PARIN)
            numItems++;
    cout << "Parameters:   " << numItems << std::endl;

    // print parameters
    numItems = 0;
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == PARIN)
        {
            char immediate[10];
            switch (port_immediate[i])
            {
            case 0:
                strcpy(immediate, "START");
                break;
            case 1:
                strcpy(immediate, "IMM");
                break;
            default:
                strcpy(immediate, "START");
            }
            cout << "  \"" << (const char *)port_name[i].toLatin1()
                 << "\" \"" << (const char *)port_datatype[i].toLatin1()
                 << "\" \"" << (const char *)port_default[i].toLatin1()
                 << "\" \"" << (const char *)port_description[i].toLatin1()
                 << "\" \"" << immediate << '"' << std::endl;
        }

    // count OutPorts
    numItems = 0;
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == OUTPUT_PORT)
            numItems++;
    cout << "OutPorts:     " << numItems << std::endl;

    // print outPorts
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == OUTPUT_PORT)
        {
            char *dependency;
            if (!port_dependency[i].isNull())
            {
                dependency = new char[1 + strlen(port_dependency[i].toLatin1())];
                strcpy(dependency, port_dependency[i].toLatin1());
            }
            else
            {
                dependency = new char[10];
                strcpy(dependency, "default");
            }
            cout << "  \"" << (const char *)port_name[i].toLatin1()
                 << "\" \"" << (const char *)port_datatype[i].toLatin1()
                 << "\" \"" << (const char *)port_description[i].toLatin1()
                 << "\" \"" << dependency << '"' << std::endl;
        }

    // count InPorts
    numItems = 0;
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == INPUT_PORT)
            numItems++;
    cout << "InPorts:      " << numItems << std::endl;

    // print InPorts
    for (i = 0; !port_name[i].isNull(); i++)
        if (port_type[i] == INPUT_PORT)
        {
            char *required = new char[10];
            if (port_required[i] == 0)
            {
                strcpy(required, "opt");
            }
            else
            {
                strcpy(required, "req");
            }

            cout << "  \"" << (const char *)port_name[i].toLatin1()
                 << "\" \"" << (const char *)port_datatype[i].toLatin1()
                 << "\" \"" << (const char *)port_description[i].toLatin1()
                 << "\" \"" << required << '"' << std::endl;
        }
}

InvMain::~InvMain()
{
}

//------------------------------------------------------------------------
// close the application
//------------------------------------------------------------------------
void InvMain::closeEvent(QCloseEvent *ce)
{
    cerr << "receive close event " << std::endl;
    ce->ignore();
}

void InvMain::ManipT(bool state)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[0]->setChecked(true);
    viewer->curManip = InvViewer::SV_TRACKBALL;
}
void InvMain::ManipH(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[1]->setChecked(true);
    viewer->curManip = InvViewer::SV_HANDLEBOX;
}
void InvMain::ManipJ(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[2]->setChecked(true);
    viewer->curManip = InvViewer::SV_JACK;
}
void InvMain::ManipC(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[3]->setChecked(true);
    viewer->curManip = InvViewer::SV_CENTERBALL;
}
void InvMain::ManipTF(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[4]->setChecked(true);
    viewer->curManip = InvViewer::SV_XFBOX;
}
void InvMain::ManipTB(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[5]->setChecked(true);
    viewer->curManip = InvViewer::SV_TABBOX;
}
void InvMain::ManipNone(bool)
{
    for (int i = 0; i < 6; i++)
    {
        mid[i]->setChecked(false);
    }
    mid[6]->setChecked(true);

    viewer->highlightRA->setVisible(false);
    if (viewer->curManipReplaces)
        viewer->replaceAllManips(viewer->curManip);
    if (viewer->isViewing())
        viewer->setViewing(false);

    viewer->curManip = InvViewer::SV_NONE;
    viewer->highlightRA->setVisible(true);
    if (viewer->curManipReplaces)
        viewer->detachManipFromAll();
    viewer->setViewing(true);
}
void InvMain::setCapture(bool state)
{
    if (state)
    {
        int width, height;
        viewer->setDecoration(SbBool(false));
        //viewer->setFullScreen(SbBool(true));
        //showFullScreen();
        width = viewer->getRenderAreaWidget()->width();
        height = viewer->getRenderAreaWidget()->height();
        //renderer_props->setItemChecked(rid[4], true);
        viewer->setRenderWindowCaptureSize(width, height);
        std::cout << "Size of Render Window: " << width << "x" << height << std::endl;
        viewer->enableRenderWindowCapture(true);
    }
    else
    {
        //renderer_props->setItemChecked(rid[4], false);
        viewer->enableRenderWindowCapture(false);
    }
}

void InvMain::doSnap(bool)
{
    int width, height;
    width = viewer->getRenderAreaWidget()->width();
    height = viewer->getRenderAreaWidget()->height();
    viewer->setRenderWindowCaptureSize(width, height);
    viewer->writeRenderWindowSnapshot();
}
void InvMain::rendererPropMode(bool checked)
{
    if (BBAction[0]->isChecked())
    {
        rendererProp = 0;
        BBAction[1]->setChecked(false);
        BBAction[2]->setChecked(false);
        BBAction[3]->setChecked(false);

        viewer->setBillboardRenderingMethod(0);
    }
    else if (BBAction[1]->isChecked())
    {
        rendererProp = 1;
        BBAction[0]->setChecked(false);
        BBAction[2]->setChecked(false);
        BBAction[3]->setChecked(false);

        viewer->setBillboardRenderingMethod(1);
    }
    else if (BBAction[2]->isChecked())
    {
        rendererProp = 2;
        BBAction[0]->setChecked(false);
        BBAction[1]->setChecked(false);
        BBAction[3]->setChecked(false);

        viewer->setBillboardRenderingMethod(2);
    }
    else if (BBAction[3]->isChecked())
    {
        BBAction[0]->setChecked(false);
        BBAction[1]->setChecked(false);
        BBAction[2]->setChecked(false);
        rendererPropBlending = 1 - rendererPropBlending;
        if (rendererPropBlending == 1)
        {
            BBAction[3]->setChecked(true);
            viewer->setBillboardRenderingBlending(false);
        }
        else
        {
            BBAction[3]->setChecked(false);
            viewer->setBillboardRenderingBlending(true);
        }
    }
}

void InvMain::doEditBackground(bool)
{
    float rr, gg, bb;
    int ir, ig, ib;
    const SbColor &c = viewer->getBackgroundColor();
    c.getValue(rr, gg, bb);
    ir = (int)(255. * rr);
    ig = (int)(255. * gg);
    ib = (int)(255. * bb);
    QColor col2 = QColor(ir, ig, ib);
    QColor col = QColorDialog::getColor(col2);
    rr = (float)col.red() / 255.;
    gg = (float)col.green() / 255.;
    bb = (float)col.blue() / 255.;
    viewer->setBackgroundColor(SbColor(rr, gg, bb));

    // store background color
    renderConfig->setValue("red", QString::number(rr), "Renderer.BackgroundColor");
    renderConfig->setValue("green", QString::number(gg), "Renderer.BackgroundColor");
    renderConfig->setValue("blue", QString::number(bb), "Renderer.BackgroundColor");
    renderConfig->save();

    char message[95];

    if (master && synced != SYNC_LOOSE)
    {
        sprintf(message, "%f %f %f", rr, gg, bb);
        cm->sendBackcolorMessage(message);
    }
}
void InvMain::showCoordinateAxis(bool)
{
    bool s = SCAction->isChecked();
    //viewing->setItemChecked( vid[1], s);
    viewer->setAxis(s);

    if (s)
    {
        cm->sendAxisMessage("1");
        renderConfig->setValue("Renderer.ShowAxis", "true");
    }

    else
    {
        cm->sendAxisMessage("0");
        renderConfig->setValue("Renderer.ShowAxis", "false");
    }
    renderConfig->save();
}
void InvMain::clippingPlane(bool)
{
    //viewing->setItemChecked( vid[2], s);
    if (CPAction->isChecked())
    {
        viewer->setClipping(CO_ON);
        //viewer->editClippingPlane();
        //viewer->sendClippingPlane(CO_ON, viewer->eqn);
        //viewing->setItemChecked( selectedId, true);
    }

    else
    {
        viewer->setClipping(CO_OFF);
        /*
         if (viewer->clippingPlaneEditor)
            viewer->clippingPlaneEditor->hide();
          */
        //viewer->sendClippingPlane(CO_OFF, viewer->eqn);
        //viewing->setItemChecked( selectedId, false);
    }
}
void InvMain::showWindowDecoration(bool)
{
    viewer->setDecoration(SWAction->isChecked());
    //viewer->setDecoration(SbBool(true));
}
void InvMain::showFullSizeWindow(bool)
{
    //viewing->setItemChecked( vid[4], s);
    if (!SFAction->isChecked())
    {
        showNormal();
        //this->menuBar()->show();
        listTabs->show();
    }
    else
    {
        showFullScreen();
        //this->menuBar()->hide();
        listTabs->hide();
    }
}

void InvMain::switchAxisMode(int mode)
{
    SCAction->setChecked(mode != 0);
}

void InvMain::sendSyncMode(bool checked)
{
    QString text;
    if (NoCouplingAction->isChecked())
    {
        synced = SYNC_LOOSE;
        text = "LOOSE";
    }

    else if (MasterSlaveAction->isChecked())
    {
        synced = SYNC_SYNC;
        text = "SYNC";
    }

    else if (MasterSlaveAction->isChecked())
    {
        synced = SYNC_TIGHT;
        text = "TIGHT";
    }
    else
    {
        return;
    }

    switchSyncMode();

    if (master)
    {
        cm->sendSyncModeMessage(text);
    }
}

void InvMain::setSyncMode(int mode)
{
    synced = mode;
    switchSyncMode();
}

void InvMain::setRendererPropMode(int mode)
{
    if (mode >= 0 && mode <= 3)
    {
        rendererProp = mode;
    }
    else if (mode == 4)
    {
        rendererPropBlending = 1 - rendererPropBlending;
    }
}

void InvMain::setMaster(int mode)
{
    // set new status
    master = mode;

    // reset items in menubar
    /*TODO  for(unsigned int i=0; i<sync->count(); i++)
   {
      menuBar()->setItemEnabled(sid[i], mode);
   }

   for(unsigned int i=0; i<viewing->count(); i++)
   {
      menuBar()->setItemEnabled(vid[i], mode);
   }*/
}

void InvMain::switchSyncMode()
{
    /* TODO  int   id;

  for ( int i = 0; i < int(sync->count()); i++ )
   {
      id = sync->idAt( i );
      sync->setItemChecked( id, false);
   }

   id = sync->idAt( synced );
   sync->setItemChecked( id, true );*/
}

void InvMain::objectListCB(const QModelIndex &index)
{
    if (viewer)
    {

        //TODO selection viewer->objectListCB(objListBox->indexWidget(index));
    }
}

void InvMain::doMaterialEditor(bool)
{
#ifdef HAVE_EDITORS
    viewer->createMaterialEditor();
#endif
}
void InvMain::doColorEditor(bool)
{
#ifdef HAVE_EDITORS
//viewer->createColorEditor();
#endif
}
void InvMain::doObjectTransform(bool)
{
}
void InvMain::doParts(bool)
{
}
void InvMain::doSnapHandleToAxis(bool)
{
    viewer->toggleHandleState();
}
void InvMain::doFreeHandleMotion(bool)
{
    viewer->toggleHandleState();
}
void InvMain::doNumericClipPlane(bool)
{
    viewer->toggleClippingPlaneEditor();
}

void InvMain::resizeEvent(QResizeEvent *e)
{
    renderConfig->setValue("width", QString::number(this->width()), "Renderer.Windows");
    renderConfig->setValue("height", QString::number(this->height()), "Renderer.Windows");
    renderConfig->save();
    QMainWindow::resizeEvent(e);
}

void InvMain::moveEvent(QMoveEvent *e)
{
    QPoint global = this->mapToGlobal(QPoint(0, 0));
    renderConfig->setValue("xa", QString::number(global.x()), "Renderer.Windows");
    renderConfig->setValue("ya", QString::number(global.y()), "Renderer.Windows");
    renderConfig->save();
    QMainWindow::moveEvent(e);
}

void InvMain::insertColorListItem(const char *name, const char *colormap)
{
    float my_min = 0.0f, my_max = 0.0f; 
    int num = 0, dummy_0;
    char mapname[256];
    char annotation[256];
    int index = 0;
    int numLeg = 11;
    int colorMapHeight = 250;

    memset(mapname, 0, sizeof(mapname));
    memset(annotation, 0, sizeof(annotation));

    stringstream stream(colormap);
    stream.getline(mapname, sizeof(mapname), '\n');
    //stream >> mapname;
    stream.getline(annotation, sizeof(annotation), '\n');
    //stream >> annotation;
    stream >> my_min;
    stream >> my_max;
    stream >> num;
    stream >> dummy_0;

    float *colorMap = new float[num * 4];
    memset(colorMap, 0, sizeof(float) * num * 4);

    for (index = 0; index < num * 4; index++)
        stream >> colorMap[index];

    QBrush check = QPixmap(checker);
    QPixmap *p = new QPixmap(100, colorMapHeight + 20);
    p->fill(Qt::white);
    QPainter painter(p);

    // draw colormap
    float scale = (float)colorMapHeight / (float)num;
    for (index = 0; index < num; index++)
    {
        QBrush brush(QColor((int)(colorMap[index * 4] * 255),
                            (int)(colorMap[index * 4 + 1] * 255),
                            (int)(colorMap[index * 4 + 2] * 255),
                            (int)(colorMap[index * 4 + 3] * 255)));
        painter.fillRect(0, colorMapHeight - (int)floor(((index + 1) * scale)) + 10, 32, (int)ceil(scale), check);
        painter.fillRect(0, colorMapHeight - (int)floor(((index + 1) * scale)) + 10, 32, (int)ceil(scale), brush);
    }

    // add values
    QFont font("Times", 10);
    QFontMetrics fm(font);
    painter.setWorldMatrixEnabled(false);
    painter.setPen(Qt::black);
    painter.setFont(font);
    float inc = (my_max - my_min) / (numLeg - 1);
    const char *format;
    if (((my_max - my_min) < 0.1) || ((my_max - my_min) > 1e5))
    {
        format = "%5.3e";
    }
    else
    {
        format = "%.3f";
    }
    for (index = 0; index < numLeg; index++)
    {
        char value[12];
        snprintf(value, 12, format, my_min + inc * index);
        int ypos = (int)(numLeg - 1 - index) * (colorMapHeight / (numLeg - 1)) + 10 + fm.descent();
        painter.drawText(QPoint(36, ypos), value);
    }

    painter.drawRect(0, 10, 32, colorMapHeight);

    string desc(annotation);
    desc += " ";
    desc += name;

    QIcon CIcon(*p);
    //CIcon.actualSize(QSize(p->width(),p->height()));

    QListWidgetItem *LWidgetItem
        = new QListWidgetItem(CIcon, desc.c_str(), colorListWidget, 0);
    //LWidgetItem->setSizeHint(QSize(p->width(), p->height()));
    colorListWidget->addItem(LWidgetItem);

    PixmapWidget *w = new PixmapWidget(*p, "test");
    listTabs->addTab(w, "ColorMap Widget");

    delete[] colorMap;
}

void InvMain::colorSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
    /*	QVariant data = colorListModel->data(selected.first().indexes().first(),Qt::DisplayRole);
	QString st = data.toString();
	*/
}
