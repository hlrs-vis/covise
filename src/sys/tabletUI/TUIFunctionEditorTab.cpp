/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TUITab.h"
#include "TUIApplication.h"
#include "TUIFunctionEditorTab.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif
#include <QLabel>
#include <QColor>
#include <QLineEdit>
#include <QSlider>
#include <QValidator>
#include <QGridLayout>
#include <QButtonGroup>
#include <QtGui>
#include <QSpinBox>

//#include <comm/msg/coSendBuffer.h>

#include <qpalette.h>

#include <stdio.h>
#include <assert.h>
#include <limits>
#include <math.h>

#include "TUITF2DEditor.h"

//Constructor
TUIFunctionEditorTab::TUIFunctionEditorTab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
    , initFlag(false)
    , userDefined(false)
    ,
    //immediateUpdate(true),
    //currAlpha(255),
    fmin(0.0f)
    , fmax(1.0f)
{
    label = name;

    QFrame *frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);
    widget = frame;
    layout = new QGridLayout(frame);

    udc.append("User Defined");
    icm.append("Initial ColorMap");

    // set a default map

    // read colormaps from standard color XML file & from local user (.covise/colormaps)
    //readConfigFile();

    // make main layout
    makeEditor();

    // set default tables
    initFlag = true;
    widget->setFocusPolicy(Qt::StrongFocus);
    widget->setVisible(!hidden);
}

TUIFunctionEditorTab::~TUIFunctionEditorTab()
{
}

void TUIFunctionEditorTab::widgetListMessage(covise::TokenBuffer &tb)
{
    uint32_t newEditorType;
    tb >> newEditorType;

    changeTFEDimension(newEditorType);

    functionEditor->parseMessage(tb);
}

void TUIFunctionEditorTab::histogramMessage(covise::TokenBuffer &tb)
{
    uint32_t dim;
    tb >> dim;

    if (dim == 1)
    {
        uint32_t histogramBuckets;
        tb >> histogramBuckets;

        //float xmax = 0.0;
        //float xmin = std::numeric_limits<float>::max();

        int *histogramData = new int[histogramBuckets];
        for (uint32_t i = 0; i < histogramBuckets; ++i)
        {
            tb >> histogramData[i];
            //xmin = std::min((float)histogramData[i], xmin);
            //xmax = std::max((float)histogramData[i], xmax);
        }
        updateHistogram(histogramBuckets, 0.0f, 1.0f, histogramData);
    }
    else
    {
        uint32_t histoBuckets1;
        uint32_t histoBuckets2;
        tb >> histoBuckets1;
        tb >> histoBuckets2;

        int *histogramData = new int[histoBuckets1 * histoBuckets2];
        for (uint32_t i = 0; i < histoBuckets1 * histoBuckets2; ++i)
        {
            tb >> histogramData[i];
        }

        TUITF2DEditor *func2D = static_cast<TUITF2DEditor *>(functionEditor);
        func2D->setHistogramData(histoBuckets1, histoBuckets2, histogramData);
    }
}

// receive a list of colors, rgba, and uses them to update the
// color map for the transfer function
void TUIFunctionEditorTab::setValue(int type, covise::TokenBuffer &tb)
{
    switch (type)
    {
    case TABLET_TF_WIDGET_LIST:
        widgetListMessage(tb);
        break;

    case TABLET_TF_HISTOGRAM:
        histogramMessage(tb);
        break;
    }

    TUIElement::setValue(type, tb);
}

void TUIFunctionEditorTab::valueChanged()
{
    covise::TokenBuffer tb;

    tb << ID;
    tb << TABLET_TF_WIDGET_LIST;
    tb << (int)this->editorType;
    functionEditor->valueChanged(tb);

    // finally send
    TUIMainWindow::getInstance()->send(tb);
}

char *TUIFunctionEditorTab::getClassName()
{
    return (char *)"TUIFunctionEditorTab";
}

void TUIFunctionEditorTab::make1DEditor()
{
    // add the color tables
    if (gb != NULL)
        delete gb;

    gb = new QGroupBox("RGB-Alpha Table", widget);
    QGridLayout *grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(0);

    // Set up our command bar
    drawAlphaFree = new QPushButton(widget);
    drawAlphaFree->setText("Draw Alpha Function");
    drawAlphaFree->setCheckable(true);
    eraseAlphaFree = new QPushButton(widget);
    eraseAlphaFree->setText("Erase Alpha Function");
    deleteMarker = new QPushButton(widget);
    deleteMarker->setText("Delete Current Marker");
    applyFunction = new QPushButton(widget);
    applyFunction->setText("Apply settings");

    //not used in our case
    chkOwnColor = NULL;

    QHBoxLayout *editorCmdLayout = new QHBoxLayout();
    editorCmdLayout->setMargin(2);
    editorCmdLayout->setSpacing(2);
    editorCmdLayout->addWidget(drawAlphaFree);
    editorCmdLayout->addWidget(eraseAlphaFree);
    editorCmdLayout->addWidget(deleteMarker);
    editorCmdLayout->addWidget(applyFunction);
    grid->addLayout(editorCmdLayout, 0, 0);

    // The function editor itself
    functionEditor = new TUITF1DEditor(this, gb);
    functionEditor->setToolTip("Current colormap");
    grid->addWidget(functionEditor, 1, 0);

    connect(functionEditor, SIGNAL(newPoint(TUITFEWidget *)), this, SLOT(pointAdded(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(pickPoint(TUITFEWidget *)), this, SLOT(pointPicked(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(deletePoint(TUITFEWidget *)), this, SLOT(pointRemoved(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(movePoint(TUITFEWidget *)), this, SLOT(valuesChanged(TUITFEWidget *)));

    connect(deleteMarker, SIGNAL(clicked()), functionEditor, SLOT(removeCurrentMarker()));
    connect(drawAlphaFree, SIGNAL(toggled(bool)), functionEditor, SLOT(setDrawAlphaFree(bool)));
    connect(eraseAlphaFree, SIGNAL(clicked()), functionEditor, SLOT(eraseAlphaFree()));

    connect(applyFunction, SIGNAL(clicked()), this, SLOT(valueChanged()));
    connect(functionEditor, SIGNAL(functionChanged()), this, SLOT(valueChanged()));

    gb->setLayout(grid);
    mainLayout->addWidget(gb);
    histo->setVisible(!hidden);

    //assure we have at least two control points
    functionEditor->loadDefault();
}

void TUIFunctionEditorTab::make2DEditor()
{
    // add the color tables
    if (gb != NULL)
        delete gb;

    histo->hide();

    gb = new QGroupBox("RGB-Alpha Table", widget);
    QGridLayout *grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(0);

    // Set up our command bar
    deleteMarker = new QPushButton(widget);
    deleteMarker->setText("Delete Current Marker");
    applyFunction = new QPushButton(widget);
    applyFunction->setText("Apply settings");

    editAlpha = new QLineEdit(widget);
    editAlpha->setValidator(new QIntValidator(1, 255, widget));
    editAlpha->setText("128");
    chkOwnColor = new QCheckBox(widget);

    QSpinBox *editWidth = new QSpinBox(widget);
    editWidth->setRange(1, 20);
    editWidth->setSingleStep(1);
    editWidth->setValue(4);

    QButtonGroup *backGroup = new QButtonGroup(widget);
    QHBoxLayout *backCmdLayout = new QHBoxLayout();

    QPushButton *btnBlack = new QPushButton("Black BG");
    btnBlack->setCheckable(true);
    backGroup->addButton(btnBlack, (int)Canvas::BackBlack);
    backCmdLayout->addWidget(btnBlack);

    QPushButton *btnCheck = new QPushButton("Checkers BG");
    btnCheck->setCheckable(true);
    backGroup->addButton(btnCheck, (int)Canvas::BackChecker);
    backCmdLayout->addWidget(btnCheck);

    QPushButton *btnHisto = new QPushButton("Histo BG");
    btnHisto->setCheckable(true);
    backGroup->addButton(btnHisto, (int)Canvas::BackHistogram);
    backCmdLayout->addWidget(btnHisto);

    // default is a black bg
    btnBlack->setChecked(true);

    // A group of buttons is more handy that a drop-down menu on a tablet
    QButtonGroup *groupBox = new QButtonGroup(widget);

    QPushButton *btnDrawFree = new QPushButton("Paint Alpha");
    //btnDrawFree->setAutoExclusive(true);
    btnDrawFree->setCheckable(true);
    groupBox->addButton(btnDrawFree, (int)TUITFEWidget::TF_MAP);

    QPushButton *btnDrawContours = new QPushButton("Draw Contours");
    //btnDrawFree->setAutoExclusive(true);
    btnDrawContours->setCheckable(true);
    groupBox->addButton(btnDrawContours, (int)TUITFEWidget::TF_CUSTOM_2D);

    QPushButton *btnExtrudeContours = new QPushButton("Extrude Contours");
    //btnDrawFree->setAutoExclusive(true);
    btnExtrudeContours->setCheckable(true);
    groupBox->addButton(btnExtrudeContours, (int)TUITFEWidget::TF_CUSTOM_2D_EXTRUDE);

    QPushButton *btnRampContours = new QPushButton("Ramp from Contours");
    //btnDrawFree->setAutoExclusive(true);
    btnRampContours->setCheckable(true);
    groupBox->addButton(btnRampContours, (int)TUITFEWidget::TF_CUSTOM_2D_TENT);

    QPushButton *btnAddColor = new QPushButton("Color");
    btnAddColor->setCheckable(true);
    groupBox->addButton(btnAddColor, (int)TUITFEWidget::TF_COLOR);

    QPushButton *btnAddPyramid = new QPushButton("Pyramid");
    btnAddPyramid->setCheckable(true);
    groupBox->addButton(btnAddPyramid, (int)TUITFEWidget::TF_PYRAMID);

    QPushButton *btnAddBell = new QPushButton("Bell");
    btnAddBell->setCheckable(true);
    groupBox->addButton(btnAddBell, (int)TUITFEWidget::TF_BELL);

    // default is add color
    btnAddColor->setChecked(true);

    QHBoxLayout *paintCmdLayout = new QHBoxLayout();
    paintCmdLayout->addWidget(new QLabel("Opacity: ", widget));
    paintCmdLayout->addWidget(editAlpha);
    paintCmdLayout->addWidget(new QLabel("Brush: ", widget));
    paintCmdLayout->addWidget(editWidth);
    paintCmdLayout->addWidget(new QLabel("Own color: ", widget));
    paintCmdLayout->addWidget(chkOwnColor);

    QVBoxLayout *editorCmdLayout = new QVBoxLayout();
    editorCmdLayout->setMargin(2);
    editorCmdLayout->setSpacing(2);
    editorCmdLayout->addLayout(backCmdLayout);
    editorCmdLayout->addSpacing(3);
    editorCmdLayout->addWidget(deleteMarker);
    editorCmdLayout->addWidget(applyFunction);
    //
    editorCmdLayout->addSpacing(3);
    editorCmdLayout->addLayout(paintCmdLayout);
    editorCmdLayout->addWidget(btnDrawFree);
    //editorCmdLayout->addWidget(btnDrawContours);
    //editorCmdLayout->addWidget(btnExtrudeContours);
    //editorCmdLayout->addWidget(btnRampContours);
    editorCmdLayout->addSpacing(1);
    editorCmdLayout->addWidget(btnAddColor);
    editorCmdLayout->addWidget(btnAddPyramid);
    editorCmdLayout->addWidget(btnAddBell);

    grid->addLayout(editorCmdLayout, 0, 1);

    // The function editor itself
    functionEditor = new TUITF2DEditor(this, gb);
    functionEditor->setToolTip("Current colormap");
    grid->addWidget(functionEditor, 0, 0);

    connect(functionEditor, SIGNAL(newPoint(TUITFEWidget *)), this, SLOT(pointAdded(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(pickPoint(TUITFEWidget *)), this, SLOT(pointPicked(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(deletePoint(TUITFEWidget *)), this, SLOT(pointRemoved(TUITFEWidget *)));
    connect(functionEditor, SIGNAL(movePoint(TUITFEWidget *)), this, SLOT(valuesChanged(TUITFEWidget *)));

    connect(deleteMarker, SIGNAL(clicked()), functionEditor, SLOT(removeCurrentMarker()));
    connect(applyFunction, SIGNAL(clicked()), this, SLOT(valueChanged()));
    connect(functionEditor, SIGNAL(functionChanged()), this, SLOT(valueChanged()));

    connect(editAlpha, SIGNAL(textEdited(const QString &)), this, SLOT(changedAlphaValue(const QString &)));
    connect(editWidth, SIGNAL(valueChanged(int)), functionEditor, SLOT(changedBrushWidth(int)));
    connect(chkOwnColor, SIGNAL(stateChanged(int)), functionEditor, SLOT(changedOwnColor(int)));

    connect(groupBox, SIGNAL(buttonClicked(int)), this, SLOT(changeWidgetType(int)));
    connect(backGroup, SIGNAL(buttonClicked(int)), functionEditor, SLOT(setBackType(int)));

    gb->setLayout(grid);
    mainLayout->addWidget(gb);
}

void TUIFunctionEditorTab::changedAlphaValue(const QString &text)
{
    if (editorType == TF_2D_EDITOR)
    {
        TUITF2DEditor *func2d = static_cast<TUITF2DEditor *>(functionEditor);
        if (editAlpha->hasAcceptableInput())
        {
            func2d->currentAlpha = text.toInt();
        }
    }
}

void TUIFunctionEditorTab::changeWidgetType(int id)
{
    if (editorType == TF_2D_EDITOR)
    {
        TUITF2DEditor *func2d = static_cast<TUITF2DEditor *>(functionEditor);
        int data = id;
        if (data >= 0)
            func2d->setCurrentWidgetType((TUITFEWidget::TFKind)data);
    }
}

/*
// make layout and widgets
*/
void TUIFunctionEditorTab::makeEditor()
{
    // set main layout
    mainLayout = new QVBoxLayout();
    mainLayout->setMargin(2);
    mainLayout->setSpacing(2);

    // add the color chooser
    chooser = new TUIColorWidget(widget);
    connect(chooser, SIGNAL(changedColor(QColor)), this, SLOT(showNewColor(QColor)));
    mainLayout->addWidget(chooser, 0);

    layout->addLayout(mainLayout, 0, 0);
    gb = NULL;
    functionEditor = NULL;
    chkOwnColor = NULL;

    QGroupBox *groupBox = new QGroupBox("Transfer Function type", widget);

    btn1D = new QPushButton("1D Function Editor");
    btn1D->setCheckable(true);
    btn2D = new QPushButton("2D Function Editor");
    btn2D->setCheckable(true);
    QButtonGroup *buttonGroup = new QButtonGroup(groupBox);
    buttonGroup->addButton(btn1D, (int)TF_1D_EDITOR);
    buttonGroup->addButton(btn2D, (int)TF_2D_EDITOR);

    groupBox->setLayout(new QHBoxLayout);
    groupBox->layout()->addWidget(btn1D);
    groupBox->layout()->addWidget(btn2D);
    mainLayout->addWidget(groupBox);

    connect(buttonGroup, SIGNAL(buttonClicked(int)), this, SLOT(changeTFEDimension(int)));

    //new TUITF1DEditor(this, gb);
    //functionEditor->setToolTip("Current colormap");
    //grid->addWidget(functionEditor, 1, 0);

    //connect(functionEditor, SIGNAL (newPoint(TUITFEWidget*)),     this, SLOT (pointAdded (TUITFEWidget*)));
    //connect(functionEditor, SIGNAL (pickPoint(TUITFEWidget*)),    this, SLOT (pointPicked(TUITFEWidget*)));
    //connect(functionEditor, SIGNAL (deletePoint(TUITFEWidget*)),  this, SLOT (pointRemoved(TUITFEWidget*)));
    //connect(functionEditor, SIGNAL (movePoint(TUITFEWidget*)),    this, SLOT (valuesChanged(TUITFEWidget*)));

    histo = new QGroupBox("Histogram", widget);
    // create a widget containing a data histogramm
    histogram = new TUIHistogram(this, histo);
    histogram->setMinimumHeight(96);
    histo->setLayout(new QVBoxLayout);
    histo->layout()->addWidget(histogram);
    mainLayout->addWidget(histo);

    // add widgets for value display
    QHBoxLayout *hb = new QHBoxLayout();

    // minimum
    minval = new QLabel(QString("Min: %1 ").arg(fmin, 0, 'f', 3), widget);
    hb->addWidget(minval);
    hb->addStretch(5);

    // current value
    hb->addWidget(new QLabel("Current:", widget));
    tfval = new QLineEdit(widget);
    tfval->setText("NONE");
    tfval->setToolTip("Current value");
    connect(tfval, SIGNAL(returnPressed()), this, SLOT(newWidgetValue()));
    hb->addWidget(tfval);
    hb->addStretch(5);

    // maximum
    maxval = new QLabel(QString("Max: %1 ").arg(fmax, 0, 'f', 3), widget);
    hb->addWidget(maxval);
    mainLayout->addLayout(hb);

    // create a row with action and selection buttons
    //makeButtonPanel();
}

void TUIFunctionEditorTab::changeTFEDimension(int newEditorType)
{
    if (functionEditor != NULL && newEditorType != this->editorType)
    {
        delete functionEditor;
        functionEditor = NULL;
    }

    //initialize the color chooser to a meaningful color
    currColor = QColor(Qt::green);
    chooser->setColor(currColor, 255);

    this->editorType = (TFEditorType)newEditorType;

    if (functionEditor == NULL)
    {
        if (editorType == TF_1D_EDITOR)
        {
            make1DEditor();
            btn1D->setChecked(true);
        }
        else
        {
            make2DEditor();
            btn2D->setChecked(true);
        }
    }
}

/*#define addPushButton(widget, text, tooltip, callback)
//QPushButton *widget = new QPushButton(text, this);
//connect(widget, SIGNAL (clicked()), this, SLOT (callback()));
//widget->setToolTip(tooltip);
//hb->addWidget(widget, 1); */

/*
//! make the lowest button box
*/
void TUIFunctionEditorTab::makeButtonPanel()
{
    QVBoxLayout *out = new QVBoxLayout();
    mainLayout->addLayout(out);

    // first line
    QHBoxLayout *hb = new QHBoxLayout();
    out->addLayout(hb);

    // generate a combo box with names of found colormaps
    //if(!port)
    //{
    QLabel *l = new QLabel("Predefined ColorMaps:", widget);
    hb->addWidget(l);
    namebox = new QComboBox(widget);
    hb->addWidget(namebox, 1);
    namebox->setToolTip("Currently available predefined colormaps");
    /*}

   else
   {
      nameboxM = new QComboBox();
      nameboxC = new QComboBox();
      nameboxM->hide();
      nameboxC->hide();
      nameboxM->setToolTip("Currently available predefined colormaps");
      nameboxC->setToolTip("Currently available predefined colormaps");
   }*/

    // fill combo box
    //if(port)
    //{
    //   connect(nameboxC, SIGNAL(activated(int)), this, SLOT(newMapC(int))) ;
    //   connect(nameboxM, SIGNAL(activated(int)), this, SLOT(newMapM(int))) ;
    //}
    //else
    connect(namebox, SIGNAL(activated(int)), this, SLOT(newMap(int)));

    for (int i = 0; i < mapNames.size(); i++)
    {
        //if(port)
        //{
        //   nameboxC->addItem(mapNames[i]);
        //   nameboxM->addItem(mapNames[i]);
        //}
        //else
        namebox->addItem(mapNames[i]);
    }

    // second line
    hb = new QHBoxLayout();
    out->addLayout(hb);
}

//!
//! make a toolbar for stand alone module
//!
void TUIFunctionEditorTab::makeToolbar()
{
    QWidget *w = new QWidget(widget);

    QHBoxLayout *box = new QHBoxLayout(w);
    box->setMargin(1);
    box->setSpacing(1);

    // create some buttons
    QCheckBox *tb = new QCheckBox("Execute on change", w);
    tb->setIcon(QPixmap(":/icons/exec.png"));
    tb->setToolTip("When selected changes are immediately executed after mouse release");
    tb->setCheckState(Qt::Checked);
    connect(tb, SIGNAL(stateChanged(int)), this, SLOT(stateCB(int)));
    box->addWidget(tb, 0);

    //colorSelector = new TUIColorSelector();
    //colorSelector->setToolTip("Use this color picker to select a color anywhere on the desktop");
    //connect(colorSelector, SIGNAL(clicked()), colorSelector, SLOT(selectedColorCB()));
    //connect(colorSelector, SIGNAL(pickedColor(const QColor &)), this, SLOT(showNewColor(const QColor &)));
    //box->addWidget(colorSelector);

    box->addStretch(1);

    mainLayout->addWidget(w);
}

//!
//! update minimum
//!
void TUIFunctionEditorTab::updateMin(float value)
{
    fmin = value;
    minval->setText(QString("Min: %1 ").arg(fmin, 0, 'f', 3));
}

//!
//! update maximun
//!
void TUIFunctionEditorTab::updateMax(float value)
{
    fmax = value;
    maxval->setText(QString("Max: %1 ").arg(fmax, 0, 'f', 3));
}

//!
//! update all widgets
//!
void TUIFunctionEditorTab::updateAll()
{
    functionEditor->update();
    chooser->update();
}

//
// Set a current marker, after changing a colormap or removing a marker.
// Always choose a color marker.
//
//void TUIFunctionEditorTab::setCurrentMarker()
//{
//   TUIColorPoint* colorPoint = colorPoints.at(colorPoints.size()/2);
//   currColor = colorPoint->getColor();
//   functionEditor->getSelectedMarker() = colorPoint;
//   functionEditor->getSelectedMarker()->setActivated(true);
//   currValue = functionEditor->getSelectedMarker()->getX();
//
//   chooser->setColor(currColor, 255);
//   float fval = currValue * (fmax - fmin) + fmin;
//   tfval->setText( QString("%1 ").arg(fval, 0, 'f', 3));
//   setFocus();
//}

//!
//! update all widgets if a marker was selected
//!
void TUIFunctionEditorTab::pointPicked(TUITFEWidget *wp)
{
    if (wp != NULL)
    {
        currValue = wp->getX();

        if (wp->hasOwnColor())
        {
            currColor = wp->getColor();
            chooser->setColor(currColor, 255);
            if (chkOwnColor != NULL)
                chkOwnColor->setChecked(true);
        }
        else
        {
            if (chkOwnColor != NULL)
                chkOwnColor->setChecked(false);
        }

        float fval = currValue * (fmax - fmin) + fmin;
        tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));
        setFocus();
    }
    else
        tfval->setText("");

    functionEditor->repaint();
}

//!
//! update all QWidgets if a marker was removed
//!
void TUIFunctionEditorTab::pointRemoved(TUITFEWidget *w)
{
    if (w == NULL)
        return;

    // set a new current point if the old one was deleted
    assert(w != functionEditor->getSelectedMarker());
    pointPicked(functionEditor->getSelectedMarker());

    setFocus();
}

//!
//! update all widgets when a marker has been moved
//!
void TUIFunctionEditorTab::valuesChanged(TUITFEWidget *wp)
{
    functionEditor->repaint();

    currValue = wp->getX();
    float fval = currValue * (fmax - fmin) + fmin;
    tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));
    setFocus();
}

//!
//! user has chosen a new  color
//!
void TUIFunctionEditorTab::showNewColor(QColor col)
{
    if (functionEditor == NULL || functionEditor->getSelectedMarker() == NULL)
        return;

    if (currColor != col)
    {
        currColor = col;
        functionEditor->getSelectedMarker()->setColor(currColor);
        chooser->setColor(currColor, 255);

        valueChanged();
    }

    functionEditor->repaint();
}

//!
//! user has created a new marker
//!
void TUIFunctionEditorTab::pointAdded(TUITFEWidget *cpnew)
{
    // show current value
    currValue = cpnew->getX();
    float fval = fmin + currValue * (fmax - fmin);
    tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));

    // set a current marker
    if (functionEditor->getSelectedMarker() != NULL)
        functionEditor->getSelectedMarker()->setActivated();

    // set the color in the color chooser
    chooser->setColor(cpnew->getColor(), 255);
    if (chkOwnColor != NULL)
        chkOwnColor->setChecked(functionEditor->getSelectedMarker()->hasOwnColor());

    setFocus();
}

//!
//! user has selected a new current value
//!
void TUIFunctionEditorTab::newWidgetValue()
{
    // get value
    float newval = tfval->text().toFloat();
    functionEditor->newWidgetValue(newval);
}

//!
//!
void TUIFunctionEditorTab::keyPressEvent(QKeyEvent *e)
{
    if (functionEditor->getSelectedMarker())
    {
        if (e->key() == Qt::Key_Delete || e->key() == Qt::Key_Backspace)
            pointRemoved(functionEditor->getSelectedMarker());
    }
}

//!
//! display a histogram
//!
void TUIFunctionEditorTab::updateHistogram(int num, const float &xmin, const float &xmax, int *values)
{
    if (editorType == TF_1D_EDITOR && histogram != NULL)
    {
        //we have the histogram widget right above the function editor
        histo->setVisible(!hidden);
        histogram->update(num, xmin, xmax, values);
        histogram->setMinimumHeight(96);
    }
}

/*!
   \class *
   \brief shows a histogram
*/

/*****************************************************************************
 *
 * Class TUIHistogram
 *
 *****************************************************************************/

TUIHistogram::TUIHistogram(TUIFunctionEditorTab *c, QWidget *p)
    : QWidget(p)
    , m_data(NULL)
    , colorMap(c)
{
    _logdata = NULL;
    histoPen.setColor(QColor(Qt::black));
    histoPen.setWidth(1);
    histoBrush.setStyle(Qt::SolidPattern);
    histoBrush.setColor(QColor(Qt::black));
    histoLogBrush.setStyle(Qt::SolidPattern);
    histoLogBrush.setColor(QColor(128, 128, 128));
}

TUIHistogram::~TUIHistogram()
{
    if (m_data)
        delete[] m_data;
    if (_logdata)
        delete[] _logdata;
}

//!
//! display a histogram
//!
void TUIHistogram::update(int num, const float &xmin, const float &xmax, int *values)
{
    m_fmin = xmin;
    m_fmax = xmax;
    m_num = num;
    if (m_data)
        delete[] m_data;
    m_data = values;

    if (_logdata)
        delete[] _logdata;
    computeLogData();

    repaint();
}

void TUIHistogram::computeLogData()
{
    _logdata = new float[m_num];
    for (int i = 0; i < m_num; ++i)
    {
        if (m_data[i] > 0)
            _logdata[i] = logf(m_data[i]);
        else
            _logdata[i] = 0.0f;
    }
}

//!
//! draw the color table
//!
void TUIHistogram::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    if (!colorMap || !colorMap->initFlag)
        return;

    QPainter p(this);
    p.setPen(histoPen);

    // get minmax of colormap
    float cmin = colorMap->getMin();
    float cmax = colorMap->getMax();

    // set a background color to that part containing the histogram data
    float fx = float(width() / (cmax - cmin));
    qreal start = (m_fmin - cmin) * fx;
    qreal end = (m_fmax - cmin) * fx;
    p.fillRect(QRectF(start, 0, end - start, height()), palette().brush(QPalette::Background));

    if (m_data)
    {
        // find histogram maximum
        int ymax = m_data[0];
        float lmax = _logdata[0];
        for (int i = 1; i < m_num; i++)
        {
            ymax = std::max(ymax, m_data[i]);
            lmax = std::max(lmax, _logdata[i]);
        }
        float fy = (float)height() / (float)ymax;
        float fl = (float)height() / lmax;

        // draw histogram
        if (cmin <= m_fmin && cmax >= m_fmax)
        {
            float dx = (float)(end - start) / (float)m_num;
            for (int i = 0; i < m_num; i++)
            {
                float y = m_data[i] * fy;
                float l = _logdata[i] * fl;
                p.fillRect(QRectF(start, height() - l, dx, l), histoLogBrush);
                p.fillRect(QRectF(start, height() - y, dx, y), histoBrush);
                start = start + dx;
            }
        }
        // histogram has to be spread
        else
        {
            // find left end
            float xx = m_fmin;
            float xl = xx;
            int sl = 0;
            float dx = (m_fmax - m_fmin) / (float)m_num;
            for (int i = 0; i < m_num; i++)
            {
                if (cmin > xx)
                {
                    xl = xx;
                    sl = i;
                    break;
                }
                else
                    xx = xx + dx;
            }

            // find right end
            xx = m_fmin;
            //float xr = xx;
            int sr = m_num - 1;
            for (int i = 0; i < m_num; i++)
            {
                if (cmax < xx)
                {
                    //xr = xx;
                    sr = i;
                    break;
                }
                else
                    xx = xx + dx;
            }
            dx = (float)width() / (float)(sr - sl);
            qreal start = (xl - cmin) * fx;
            for (int i = 0; i < sr - sl; i++)
            {
                qreal y = m_data[i] * fy;
                p.fillRect(QRectF(start, height() - y, dx, y), histoBrush);
                start = start + dx;
            }
        }
    }
}
