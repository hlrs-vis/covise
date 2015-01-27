/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QMenu>
#include <QLabel>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QComboBox>
#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QPainter>
#include <QHBoxLayout>
#include <QDebug>
#include <QGroupBox>
#include <QToolBar>
#include <QInputDialog>
#include <QDir>

#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

#include "MEColorMap.h"
#include "MEColorChooser.h"
#include "MEColorSelector.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"
#include "ports/MEColorMapPort.h"
#include "ports/MEColormapChoicePort.h"

#include "config/coConfig.h"

;

MEColorMap *colorMap = NULL;

static int nHeight = 96; // height of color table
static int mSize = 14; // size of color markers

static float s_normMap[] = {
    0.0, 0.000000, 0.000000, 1.000000, 1.000000,
    0.5, 1.000000, 0.000000, 0.000000, 1.000000,
    1.0, 1.000000, 1.000000, 0.000000, 1.000000
};

/*!
   \class MEColorMap
   \brief This class provides a widget for showing color maps
*/

/*****************************************************************************
 *
 * Class MEColorMap
 *
 *****************************************************************************/

MEColorMap::MEColorMap(MEParameterPort *p, QWidget *parent)
    : QFrame(parent)
    , initFlag(false)
    , oldSize(0)
    , oldValues(NULL)
    , fmin(0.0f)
    , fmax(1.0f)
    , icm("Module ColorMap")
    , port(p)
    , currPoint(NULL)

{
    setLineWidth(2);
    setFrameStyle(Panel | Plain);

#ifndef YAC
    // read colormaps from standard color XML file & from local user (.covise/colormaps)
    readConfigFile();
#endif

    // make main layout
    makeEditor();

#ifndef YAC
    // set default tables
    initFlag = true;
    setFocusPolicy(Qt::StrongFocus);
#endif
}

MEColorMap::~MEColorMap()
{
}

#define addPreview(type)                                        \
    preview[type] = new MEColorRGBTable(this);                  \
    preview[type]->setMinimumHeight(nHeight / 4);               \
    preview[type]->setToolTip("Preview for selected colormap"); \
    preview[type]->hide();

//!
//! make layout and widgets
//!
void MEColorMap::makeEditor()
{
    // set main layout
    mainLayout = new QVBoxLayout(this);
    mainLayout->setMargin(2);
    mainLayout->setSpacing(2);

    namebox = new QComboBox();
    namebox->setToolTip("Currently available predefined colormaps");
    connect(namebox, SIGNAL(activated(const QString &)), this, SLOT(loadMap(const QString &)));
    namebox->addItems(mapNames);

    if (port)
        makeToolbar();

    // add the color chooser
    chooser = new MEColorChooser(this);
    // signal is emitted  when color was changed, used inside the color map
    connect(chooser, SIGNAL(colorChanged(const QColor &)), this, SLOT(showNewColor(const QColor &)));
    // signal is emitted when mouse was released or a new value was typed in, execute the module
    connect(chooser, SIGNAL(colorReady()), this, SLOT(applyCB()));
    mainLayout->addWidget(chooser, 0);

    // add the color tables
    QGroupBox *gb = new QGroupBox("RGB-Alpha Table", this);
    QGridLayout *grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(0);

    // add the  RGBtables
    // preview shows the colormap within a parameter line (for port parameter only)
    preview[MODULE] = NULL;
    preview[CONTROL] = NULL;
    if (port)
    {
        addPreview(MODULE);
        addPreview(CONTROL);
    }

    table = new MEColorRGBTable(this, gb);
    table->setMinimumHeight(nHeight / 2);
    table->setToolTip("Current colormap");
    grid->addWidget(table, 0, 0);

    // add a widget that contains marker
    // manipulate marker
    container = new MEColorMarkerContainer(this, gb);
    container->setToolTip("Interpolation markers");
    // mouse press event
    connect(container, SIGNAL(newPoint(float)), this, SLOT(pointNew(float)));
    connect(container, SIGNAL(pickPoint(MEColorPoint *)), this, SLOT(pointPicked(MEColorPoint *)));
    // mouse release event
    connect(container, SIGNAL(sendColor()), this, SLOT(applyCB()));
    // mouse move event
    connect(container, SIGNAL(movePoint(MEColorPoint *)), this, SLOT(pointMoved(MEColorPoint *)));
    grid->addWidget(container, 1, 0);

    gb->setLayout(grid);
    mainLayout->addWidget(gb);

    // add widgets for value display
    QHBoxLayout *hb = new QHBoxLayout();

    // minimum
    minval = new QLabel(QString("Min: %1 ").arg(fmin, 0, 'f', 3), this);
    hb->addWidget(minval);
    hb->addStretch(5);

    // current value
    hb->addWidget(new QLabel("Current:", this));
    tfval = new QLineEdit(this);
    tfval->setText("NONE");
    tfval->setToolTip("Current value");
    connect(tfval, SIGNAL(returnPressed()), this, SLOT(newtfval()));
    hb->addWidget(tfval);
    hb->addStretch(5);

    // maximum
    maxval = new QLabel(QString("Max: %1 ").arg(fmax, 0, 'f', 3), this);
    hb->addWidget(maxval);
    mainLayout->addLayout(hb);

    // add a histogramm
    histo = new QGroupBox("Histogram", this);
    grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(0);

    // create a widget containing a data histogramm
    histogram = new MEHistogram(this, histo);
    histogram->setMinimumHeight(nHeight);
    grid->addWidget(histogram, 0, 0);
    histo->setLayout(grid);
    mainLayout->addWidget(histo, 4);
    histo->hide();

    // create a row with action and selection buttons
    makeButtonPanel();
}

#define addButton(tb, pixmap, tooltip, callback)            \
    QPushButton *tb = new QPushButton(w);                   \
    tb->setIcon(QPixmap(pixmap));                           \
    tb->setFlat(true);                                      \
    tb->setToolTip(tooltip);                                \
    connect(tb, SIGNAL(clicked()), this, SLOT(callback())); \
    box->addWidget(tb, 0);

//!
//! make a toolbar
//!
void MEColorMap::makeToolbar()
{

    QToolBar *tb = new QToolBar(this);

    // create some actions
    QAction *a = new QAction(this);
    a->setIcon(QPixmap(":/icons/filesave32.png"));
    a->setToolTip("Save the current colormap");
    connect(a, SIGNAL(triggered()), this, SLOT(saveXMLCB()));
    tb->addAction(a);

    a = new QAction(this);
    a->setIcon(QPixmap(":/icons/undo32.png"));
    a->setToolTip("Reset values of current colormap");
    connect(a, SIGNAL(triggered()), this, SLOT(resetCB()));
    tb->addAction(a);

    colorSelector = new MEColorSelector();
    colorSelector->setToolTip("Use this color picker to select a color anywhere on the desktop");
    connect(colorSelector, SIGNAL(clicked()), colorSelector, SLOT(selectedColorCB()));
    connect(colorSelector, SIGNAL(pickedColor(const QColor &)), this, SLOT(showNewColor(const QColor &)));
    tb->addWidget(colorSelector);

    tb->addWidget(namebox);

    tb->show();
    mainLayout->addWidget(tb, 0);
}

#define addPushButton(widget, text, tooltip, callback)          \
    QPushButton *widget = new QPushButton(text, this);          \
    connect(widget, SIGNAL(clicked()), this, SLOT(callback())); \
    widget->setToolTip(tooltip);                                \
    hb->addWidget(widget, 1);

//!
//! make comboboxes with colormap names
//!
void MEColorMap::makeButtonPanel()
{
    QVBoxLayout *out = new QVBoxLayout();
    mainLayout->addLayout(out);

    // first line
    QHBoxLayout *hb = new QHBoxLayout();
    out->addLayout(hb);

    // generate a combo box with names of found colormaps
    if (!port)
        out->addWidget(namebox, 1);

    // second line
    hb = new QHBoxLayout();
    out->addLayout(hb);
}

//!
//! user has selected a new colormap from combobox
//!
void MEColorMap::loadMap(const QString &name)
{
    setPredefinedMap(name);
    applyCB();
}

//!
//! show a predefined color map
//!
void MEColorMap::setPredefinedMap(const QString &name)
{
    // look in the list for available colormap name

    if (mapNames.contains(name))
    {
        currName = name;
        namebox->setCurrentIndex(namebox->findText(name));

        // remove old points
        if (points.size() != 0)
        {
            while (!points.isEmpty())
                delete points.takeFirst();
            points.clear();
        }

        // set a new color table
        int size = mapSize.value(name);
        float diff = 1.0 / (size - 1);

        // first point
        int j, k = 0;
        float *mval = mapValues.value(name);
        int red = int(mval[k + 1] * 255.);
        int green = int(mval[k + 2] * 255.);
        int blue = int(mval[k + 3] * 255.);
        int alpha = int(mval[k + 4] * 255.);
        float pos = 0.f;
        if (mval[k] == -1.)
            pos = 0.f;
        else
            pos = mval[k];

        MEColorPoint *cp = makePoint(pos, QColor(red, green, blue, alpha));
        points.append(cp);

        k = k + 5;
        int iend = (size - 1) * 5;

        for (j = k; j < iend; j = j + 5)
        {
            int red = int(mval[j + 1] * 255.);
            int green = int(mval[j + 2] * 255.);
            int blue = int(mval[j + 3] * 255.);
            int alpha = int(mval[j + 4] * 255.);
            if (mval[j] == -1.)
                pos = pos + diff;
            else
                pos = mval[j];

            MEColorPoint *cp = makePoint(pos, QColor(red, green, blue, alpha));
            points.append(cp);
        }

        // last  point
        k = j;
        red = int(mval[k + 1] * 255.);
        green = int(mval[k + 2] * 255.);
        blue = int(mval[k + 3] * 255.);
        alpha = int(mval[k + 4] * 255.);
        if (mval[k] == -1.)
            pos = 1.f;
        else
            pos = mval[k];

        cp = makePoint(pos, QColor(red, green, blue, alpha));
        points.append(cp);

        // update widget
        updateAll();

        // mark a point in the middle as active
        setCurrentMarker();

        // store values for a reset
        storeCurrentMap();
    }
}

//!
//! copy a colormap for resetting
//!
void MEColorMap::storeCurrentMap()
{
    if (oldValues)
        delete[] oldValues;
    oldSize = points.size();
    oldValues = new float[oldSize * 5];

    int i = 0;
    for (int k = 0; k < oldSize * 5; k = k + 5)
    {
        MEColorPoint *cp = points.at(i);
        QColor cc = cp->getColor();
        oldValues[k] = (float)(cp->getX());
        oldValues[k + 1] = (float)(cc.red()) / 255.;
        oldValues[k + 2] = (float)(cc.green()) / 255.;
        oldValues[k + 3] = (float)(cc.blue()) / 255.;
        oldValues[k + 4] = (float)(cc.alpha()) / 255.;
        i++;
    }
}

//!
//! set a colormap called from module (param message)
//!
void MEColorMap::updateColorMap(int numSteps, const float *rgbax)
{
    if (!mapNames.contains(icm))
    {
        mapNames.append(icm);
        namebox->insertItem(-1, icm);
        namebox->setCurrentIndex(0);
    }

    // change format rgbax to xrgba
    float *cval = new float[numSteps * 5];
    for (int i = 0; i < numSteps * 5; i = i + 5)
    {
        cval[i + 1] = rgbax[i];
        cval[i + 2] = rgbax[i + 1];
        cval[i + 3] = rgbax[i + 2];
        cval[i + 4] = rgbax[i + 3];
        cval[i] = rgbax[i + 4];
    }

    // store description
    mapSize.insert(icm, numSteps);
    mapValues.insert(icm, cval);

    // show map
    setPredefinedMap(icm);
}

//!
//! delete a colormap from local admin
//!
void MEColorMap::deleteMap(const QString &name)
{
    float *mval = mapValues.value(name);
    mapSize.remove(name);
    mapValues.remove(name);
    delete[] mval;
}

//!
//! update minimum
//!
void MEColorMap::updateMin(float value)
{
    fmin = value;
    minval->setText(QString("Min: %1 ").arg(fmin, 0, 'f', 3));
}

//!
//! update maximun
//!
void MEColorMap::updateMax(float value)
{
    fmax = value;
    maxval->setText(QString("Max: %1 ").arg(fmax, 0, 'f', 3));
}

//!
//! update all widgets
//!
void MEColorMap::updateAll()
{
    table->update();
    if (preview[MODULE])
        preview[MODULE]->update();
    if (preview[CONTROL])
        preview[CONTROL]->update();
    container->update();
}

//!
//! get the rgbax value for given index
//!
void MEColorMap::getStep(int step, float *r, float *g, float *b, float *a, float *x)
{
    MEColorPoint *point = points.at(step);
    if (point)
    {
        QColor c = point->getColor();
        if (r)
        {
            *r = c.red() / 255.0;
        }
        if (g)
        {
            *g = c.green() / 255.0;
        }
        if (b)
        {
            *b = c.blue() / 255.0;
        }
        if (a)
        {
            *a = c.alpha() / 255.0;
        }
        if (x)
        {
            *x = point->getX();
        }
    }
}

//!
//! set a current marker, after changing a colormap or removing a marker
//!
void MEColorMap::setCurrentMarker()
{
    currPoint = points.at(points.size() / 2);
    currPoint->setActivated(true);
    QColor c(currPoint->getColor());
    chooser->setColor(c);
    currValue = currPoint->getX();
    float fval = currValue * (fmax - fmin) + fmin;
    tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));
    setFocus();
}

//!
//! set a new point in the manipulation area
//!
MEColorPoint *MEColorMap::makePoint(float pos, QColor col)
{
    MEColorPoint *cp = new MEColorPoint(container);
    connect(cp, SIGNAL(removePoint(MEColorPoint *)), this, SLOT(pointRemoved(MEColorPoint *)));
    cp->addItems(pos, col);
    container->addMarker(cp);
    return cp;
}

//!
//! update all widgets if a marker was selected
//!
void MEColorMap::pointPicked(MEColorPoint *cp)
{
    if (currPoint != NULL && points.contains(currPoint))
        currPoint->setActivated(false);

    currPoint = cp;
    cp->setActivated(true);
    QColor c(cp->getColor());
    chooser->setColor(c);
    currValue = cp->getX();
    float fval = currValue * (fmax - fmin) + fmin;
    tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));
    setFocus();
}

//!
//! update all widgets if a marker was removed
//!
void MEColorMap::pointRemoved(MEColorPoint *cp)
{
    if (cp == NULL || currPoint == NULL)
        return;

    // check if the last or first marker was selected
    int index = points.indexOf(cp);
    if (index <= 0)
        return;

    if (index + 1 == points.size())
        return;

    // set a new current point if the old one will be deleted
    points.removeAt(index);
    cp->deleteLater();

    MEColorPoint *active = points.at(points.size() / 2);
    pointPicked(active);

    updateAll();
    setFocus();
    applyCB();
}

//!
//! update all widgets when a marker has been moved
//!
void MEColorMap::pointMoved(MEColorPoint *cp)
{
    int index = points.indexOf(cp);

    // first and last point are not allowed
    if (index <= 0)
        return;
    if (index + 1 == points.size())
        return;

    table->repaint();
    if (preview[MODULE])
        preview[MODULE]->repaint();
    if (preview[CONTROL])
        preview[CONTROL]->repaint();

    currValue = cp->getX();
    float fval = currValue * (fmax - fmin) + fmin;
    tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));
    setFocus();
}

//!
//! user has created a new marker
//!
void MEColorMap::pointNew(float newX)
{
    // load standard map if there are no points defined
    if (points.isEmpty())
        setPredefinedMap("Standard");

    MEColorPoint *cpnew = NULL;
    MEColorPoint *left = NULL;
    MEColorPoint *right = NULL;

    float xleft = 0.f;
    float xright = 0.f;

    if (points.first())
    {
        left = points.first();
        xleft = left->getX();
    }

    // find the neighbours for this new index
    int count = 0;
    for (int i = 1; i < points.size(); i++)
    {
        count++;
        right = points.at(i);
        xright = right->getX();
        if (newX > xleft && newX < xright)
        {
            // calculate color & alpha
            float diff = (xright - xleft);

            QColor rc = right->getColor();
            QColor lc = left->getColor();
            float rdiff = float(rc.red() - lc.red()) / diff;
            float gdiff = float(rc.green() - lc.green()) / diff;
            float bdiff = float(rc.blue() - lc.blue()) / diff;
            float adiff = float(rc.alpha() - lc.alpha()) / diff;

            int r = lc.red() + int(rdiff * (newX - xleft));
            int g = lc.green() + int(gdiff * (newX - xleft));
            int b = lc.blue() + int(bdiff * (newX - xleft));
            int a = lc.alpha() + int(adiff * (newX - xleft));

            QColor col(r, g, b, a);
            currColor = col;

            // make new marker
            cpnew = makePoint(newX, currColor);
            points.insert(count, cpnew);

            // show cuurent value
            currValue = cpnew->getX();
            float fval = fmin + currValue * (fmax - fmin);
            tfval->setText(QString("%1 ").arg(fval, 0, 'f', 3));

            break;
        }
        xleft = xright;
        left = right;
    }

    // set a current marker
    if (currPoint && points.contains(currPoint))
        currPoint->setActivated(false);

    currPoint = cpnew;
    if (currPoint != NULL)
        currPoint->setActivated(true);

    // set the color in the color chooser
    QColor c(currColor);
    chooser->setColor(c);
    setFocus();

    container->update();
}

//!
//! user has chosen a new  color (color picker or color choser)
//!
void MEColorMap::showNewColor(const QColor &col)
{
    if (currPoint == NULL)
        return;

    currColor = col;
    currPoint->setColor(currColor);

    // check  neighbours
    MEColorPoint *left = NULL;
    MEColorPoint *right = NULL;

    int index = points.indexOf(currPoint);
    if (index == -1)
        return;

    if (index == 0)
        left = currPoint;
    else
        left = points.at(index - 1);

    if (index + 1 == points.size())
        right = currPoint;
    else
        right = points.at(index + 1);

    if (left == right)
        return;

    table->repaint();
    if (preview[MODULE])
        preview[MODULE]->repaint();
    if (preview[CONTROL])
        preview[CONTROL]->repaint();
}

//!
//! user has selected a new current value
//!
void MEColorMap::newtfval()
{
    if (!currPoint)
        return;

    // get value
    float newval = tfval->text().toFloat();

    // check if value is possible
    // look for neighbours
    MEColorPoint *left;
    MEColorPoint *right;
    int index = points.indexOf(currPoint);
    if (index == -1)
        return;

    if (index == 0)
        left = currPoint;
    else
        left = points.at(index - 1);

    if (index + 1 == points.size())
        right = currPoint;
    else
        right = points.at(index + 1);

    // set new current value
    currValue = qMin(newval, right->getX());
    currValue = qMax(currValue, left->getX());

    // move point
    currPoint->setX(currValue);
    int x = int(currValue * container->width()) - mSize / 2;
    currPoint->move(x, 2);
    pointMoved(currPoint);
}

//!
//! user wants to reset the colormap
//!
void MEColorMap::resetCB()
{
    setPredefinedMap(currName);
    applyCB();
}

//!
//! user wants to hide the window
//!
void MEColorMap::hideCB()
{
    if (!port)
        hide();

    else
    {
        MEColorMapPort *dp = qobject_cast<MEColorMapPort *>(port);
        if (dp)
            dp->colorMapClosed();
    }
}

//!
//! remove a marker, callback from popup menu or keypress event
//!
void MEColorMap::removeMarker()
//!
{
    pointRemoved(currPoint);
}

//!
void MEColorMap::keyPressEvent(QKeyEvent *e)
//!
{
    if (currPoint)
    {
        if (e->key() == Qt::Key_Delete || e->key() == Qt::Key_Backspace)
            pointRemoved(currPoint);
    }
}

//!
//! user wants to store the current colormap in xml configuration
//!
void MEColorMap::saveXMLCB()
{

    bool ok = false;
    QString name;
    QString text = QInputDialog::getText(this,
                                         tr("COVISE - ColorMap Editor"),
                                         tr("Please enter the colormap name "),
                                         QLineEdit::Normal, QString::null, &ok);
    if (ok && !text.isEmpty())
        name = text;

    else
        return;

    // write scope entries
    covise::coConfigGroup *colorConfig = new covise::coConfigGroup("Colormap");
    QString place = covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";
    QDir directory(place);
    if (!directory.exists())
        directory.mkdir(place);

    place.append("/colormap_" + name + ".xml");
    colorConfig->addConfig(place, "local", true);
    colorConfig->setValue("Colormaps", name);
    QString cname = "Colormaps." + name;

    // store map and create new colormap values
    int k = 0;
    float *cval = new float[points.size() * 5];
    for (int i = 0; i < points.size() * 5; i = i + 5)
    {
        MEColorPoint *cp = points.at(k);
        QColor cc = cp->getColor();
        float x = (float)(cp->getX());
        float r = (float)(cc.red()) / 255.;
        float g = (float)(cc.green()) / 255.;
        float b = (float)(cc.blue()) / 255.;
        float a = (float)(cc.alpha()) / 255.;

        cval[i] = x;
        cval[i + 1] = r;
        cval[i + 2] = g;
        cval[i + 3] = b;
        cval[i + 4] = a;

        colorConfig->setValue("x", QString::number(x), cname + ".Point:" + QString::number(k));
        colorConfig->setValue("r", QString::number(r), cname + ".Point:" + QString::number(k));
        colorConfig->setValue("g", QString::number(g), cname + ".Point:" + QString::number(k));
        colorConfig->setValue("b", QString::number(b), cname + ".Point:" + QString::number(k));
        colorConfig->setValue("a", QString::number(a), cname + ".Point:" + QString::number(k));
        k++;
    }

    colorConfig->save();

    // add new created colormap to local administration
    mapNames.append(name);
    mapNames.sort();

    mapSize.insert(name, points.size());
    mapValues.insert(name, cval);

    // reset colormap names in combobox
    namebox->clear();
    namebox->addItems(mapNames);
    namebox->setCurrentIndex(namebox->findText(name));
}

//!
//! display a histogram
//!
void MEColorMap::updateHistogram(int num, const float &xmin, const float &xmax, int *values)
{
    if (histogram)
    {
        histo->show();
        histogram->update(num, xmin, xmax, values);
    }
}

//!
//! read colormaps from xml config file, read also local colormaps
//!
void MEColorMap::readConfigFile()
{

    config = covise::coConfig::getInstance();

    // read the name of all colormaps in file
    QStringList list;
    list = config->getVariableList("Colormaps");

    for (int i = 0; i < list.size(); i++)
        mapNames.append(list[i]);

    // read the values for each colormap
    for (int k = 0; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        QString cmapname = "Colormaps." + mapNames[k];
        QStringList variable = config->getVariableList(cmapname);

        mapSize.insert(mapNames[k], variable.size());
        float *cval = new float[variable.size() * 5];
        mapValues.insert(mapNames[k], cval);

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            QString tmp = cmapname + ".Point:" + QString::number(it);
            cval[l] = config->getFloat("x", tmp, -1.0);
            cval[l + 1] = config->getFloat("r", tmp, 1.0);
            cval[l + 2] = config->getFloat("g", tmp, 1.0);
            cval[l + 3] = config->getFloat("b", tmp, 1.0);
            cval[l + 4] = config->getFloat("a", tmp, 1.0);
            it++;
        }
    }

    // read values of local colormap files in .covise
    QString place = covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";

    QDir directory(place);
    if (directory.exists())
    {
        QStringList filters;
        filters << "colormap_*.xml";
        directory.setNameFilters(filters);
        directory.setFilter(QDir::Files);
        QStringList files = directory.entryList();

        // loop over all found colormap xml files
        for (int j = 0; j < files.size(); j++)
        {
            covise::coConfigGroup *colorConfig = new covise::coConfigGroup("Colormap");
            colorConfig->addConfig(place + "/" + files[j], "local", true);

            // read the name of the colormaps
            QStringList list;
            list = colorConfig->getVariableList("Colormaps");

            // loop over all colormaps in one file
            for (int i = 0; i < list.size(); i++)
            {

                // remove global colormap with same name
                int index = mapNames.indexOf(list[i]);
                if (index != -1)
                {
                    mapNames.removeAt(index);
                    deleteMap(list[i]);
                }
                mapNames.append(list[i]);

                // get all definition points for the colormap
                QString cmapname = "Colormaps." + mapNames.last();
                QStringList variable = colorConfig->getVariableList(cmapname);

                mapSize.insert(list[i], variable.size());
                float *cval = new float[variable.size() * 5];
                mapValues.insert(list[i], cval);

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    QString tmp = cmapname + ".Point:" + QString::number(it);
                    cval[l] = (colorConfig->getValue("x", tmp, " -1.0")).toFloat();
                    cval[l + 1] = (colorConfig->getValue("r", tmp, "1.0")).toFloat();
                    cval[l + 2] = (colorConfig->getValue("g", tmp, "1.0")).toFloat();
                    cval[l + 3] = (colorConfig->getValue("b", tmp, "1.0")).toFloat();
                    cval[l + 4] = (colorConfig->getValue("a", tmp, "1.0")).toFloat();
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j]);
        }
    }
    mapNames.sort();

    // set a default map
    mapNames.prepend("Standard");
    mapSize.insert("Standard", 3);
    mapValues.insert("Standard", s_normMap);
}

//!
QStringList MEColorMap::getColorMapNames()
//!
{
    QStringList names;

    for (int i = 0; i < mapNames[i].size(); i++)
        names << mapNames[i];

    return names;
}

//!
//! user wants to activate this colormap for using
//!
void MEColorMap::applyCB()
{
    MEColorMapPort *dp = qobject_cast<MEColorMapPort *>(port);
    if (dp)
    {
        dp->sendParamMessage();
        dp->getNode()->sendMessage("EXEC");
    }
}

//!
//! init or modify the values of a colormap (only used by YAC)
//!
void MEColorMap::updateColorMap(const QString &name, const QString &value)
{
    if (!mapNames.contains(name))
    {
        mapNames.append(name);
        namebox->insertItem(-1, name);
    }

    else
    {
        float *mval = mapValues.value(name);
        mapSize.remove(name);
        mapValues.remove(name);
        delete[] mval;
    }

    // parse value string from yac
    QStringList token = value.split(" ");

    // store length
    int it = 0;
    int numSteps = token[it].toInt();
    it++;
    mapSize.insert(name, numSteps);

    // unused token linear/sline
    it++;

    // store values (XRGBA -> RGBAX)
    float *cval = new float[numSteps * 5];
    for (int l = 0; l < numSteps * 5; l = l + 5)
    {
        cval[l] = token[it].toFloat();
        it++;
        cval[l + 1] = token[it].toFloat();
        it++;
        cval[l + 2] = token[it].toFloat();
        it++;
        cval[l + 3] = token[it].toFloat();
        it++;
        cval[l + 4] = token[it].toFloat();
        it++;
    }
    mapValues.insert(name, cval);

    initFlag = true;

    // repaint, if colormap was modified
    setPredefinedMap(name);

    // mark a point in the middle as active
    setCurrentMarker();
}

//!
//! read the currently defined values
//!
QString MEColorMap::getMapValues()
{

    QStringList buf;
    QString text;
    MEColorPoint *cp;

#ifdef YAC

    buf << QString::number(points.count()) << "linear";

    for (int i = 0; i < points.size(); i++)
    {
        cp = points.at(i);
        buf << QString::number(cp->getX());
        buf << QString::number((float)(cp->getColor().red()) / 255.);
        buf << QString::number((float)(cp->getColor().green()) / 255.);
        buf << QString::number((float)(cp->getColor().blue()) / 255);
        buf << QString::number((float)(cp->getColor().alpha()) / 255.);
    }

#else

    buf << QString::number(points.count()) << currType;

    for (int i = 0; i < points.size(); i++)
    {
        cp = points.at(i);
        buf << QString::number((float)(cp->getColor().red()) / 255);
        buf << QString::number((float)(cp->getColor().green()) / 255.);
        buf << QString::number((float)(cp->getColor().blue()) / 255.);
        buf << QString::number((float)(cp->getColor().alpha()) / 255.);
        buf << QString::number(cp->getX());
    }
#endif

    text = buf.join(" ");
    return text;
}

/*!
   \class *
   \brief his class provides a widget for diaplaying the actual colortable
*/

/*****************************************************************************
 *
 * Class MEColorRGBTable (Color table)
 *
 *****************************************************************************/

MEColorRGBTable::MEColorRGBTable(MEColorMap *c, QWidget *p)
    : QWidget(p)
    , colorMap(c)
{
    pm_checker = QPixmap(":/icons/checker.xpm");
}

MEColorRGBTable::~MEColorRGBTable()
{
}

//!
//! draw the color table
//!
void MEColorRGBTable::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    QPainter p(this);

    if (!colorMap || !colorMap->initFlag)
    {
        fprintf(stderr, "!init\n");
        p.fillRect(0, 0, width(), height(), QBrush(Qt::blue, Qt::SolidPattern));
        return;
    }

    // draw checker board
    p.fillRect(0, 0, width(), height(), QBrush(pm_checker));

    // loop over all interpolation markers
    for (int i = 0; i < colorMap->points.size() - 1; i++)
    {
        // set a gradientpoint
        MEColorPoint *left = colorMap->points.at(i);
        MEColorPoint *right = colorMap->points.at(i + 1);

        int xleft = int(left->getX() * width());
        int xright = int(right->getX() * width());

        QLinearGradient lgrad(xleft, 0., xright, 0.);

        QColor c1(left->getColor());
        lgrad.setColorAt(0., c1);

        QColor c2(right->getColor());
        lgrad.setColorAt(1., c2);

        p.fillRect(xleft, 0, (xright - xleft + 1), height(), QBrush(lgrad));
    }
}

/*!
   \class MEColorMarkerContainer
   \brief This class provides a widget for displaying the markers in a color map
*/

/*****************************************************************************
 *
 * Class MEColorMarkerContainer (Container for marker)
 *
 *****************************************************************************/

MEColorMarkerContainer::MEColorMarkerContainer(MEColorMap *c, QWidget *p)
    : QFrame(p)
    , colorMap(c)
    , clickPoint(NULL)
    , offset(0)
{
    setFrameStyle(Panel | Sunken);
    setFixedHeight(mSize + 4);
    ww = width();
}

MEColorMarkerContainer::~MEColorMarkerContainer()
{
    // nothing necessary
}

void MEColorMarkerContainer::addMarker(MEColorPoint *cp)
{
    int x = int(cp->getX() * width()) - mSize / 2;
    cp->move(x, 2);
    cp->show();
}

void MEColorMarkerContainer::mousePressEvent(QMouseEvent *e)

{
    if (e->button() != Qt::LeftButton)
        return;

    QVector<MEColorPoint *> list;

    // look if a marker was pressed
    for (int i = 0; i < colorMap->points.size(); i++)
    {
        MEColorPoint *cp = colorMap->points.at(i);
        QRect ra = cp->geometry();
        if (ra.contains(e->x(), e->y()))
            list.append(cp);
    }

    // no marker found --> must be a new point
    if (list.isEmpty())
    {
        clickPoint = NULL;
        float xpos = (float)(e->x()) / (float)width();
        emit newPoint(xpos);
        return;
    }

    // found exactly one point --> set this point active
    if (list.count() == 1)
        clickPoint = list.at(0);

    // found more points --> ignore first and last point
    else
    {
        if (list.contains(colorMap->points.first()))
            clickPoint = list.at(1);

        else if (list.contains(colorMap->points.last()))
            clickPoint = list.at(0);

        else
            clickPoint = list.last();
    }

    if (clickPoint)
    {
        QRect ra = clickPoint->geometry();
        offset = e->x() - ra.left();
        emit pickPoint(clickPoint);
    }
}

void MEColorMarkerContainer::mouseReleaseEvent(QMouseEvent *)

{
    if (clickPoint)
        emit sendColor();
}

void MEColorMarkerContainer::mouseMoveEvent(QMouseEvent *e)

{
    if (!clickPoint)
        return;

    int index = colorMap->points.indexOf(clickPoint);

    // first point cannot be moved
    if (index <= 0)
        return;

    // last point cannot be moved
    if (index + 1 == colorMap->points.size())
        return;

    // get neighbours
    float xx = float(e->x()) / float(width());
    float xmin = (colorMap->points.at(index - 1))->getX();
    float xmax = (colorMap->points.at(index + 1))->getX();

    // you cannot move points over the neighbour points
    if (xx <= xmin || xx >= xmax)
        return;

    clickPoint->move(e->x() - offset, 2);
    clickPoint->setX(xx);
    clickPoint->update();
    emit movePoint(clickPoint);
}

//!
//! draw the marker
//!
void MEColorMarkerContainer::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    if (ww != width())
    {
        for (int i = 0; i < colorMap->points.size(); i++)
        {
            MEColorPoint *cp = colorMap->points.at(i);
            int x = int(cp->getX() * width()) - mSize / 2;
            cp->move(x, 2);
        }
        ww = width();
    }
}

/*!
   \class MEColorPoint
   \brief This class creates provides the marker widget
*/

/*****************************************************************************
 *
 * Class MEColorPoint  (Marker)
 *
 *****************************************************************************/

MEColorPoint::MEColorPoint(QWidget *p)
    : QWidget(p)
    , state(false)
{
    setFixedSize(mSize, mSize);
    poly.setPoints(3, mSize / 2, 0, 0, mSize, mSize, mSize);

    // create a popup menu for right mouse
    menu = new QMenu(this);
    QAction *_delete = menu->addAction("&Delete");
    connect(_delete, SIGNAL(triggered()), this, SLOT(removeCB()));
}

MEColorPoint::~MEColorPoint()
{
    // nothing necessary
}

void MEColorPoint::addItems(float x, QColor col)
{
    pos = x;
    color = col;
}

void MEColorPoint::setActivated(bool flag)
{
    state = flag;
    if (state)
        raise();
    update();
}

void MEColorPoint::setColor(QColor col)
{
    color = col;
}

void MEColorPoint::contextMenuEvent(QContextMenuEvent *e)
{
    menu->popup(e->globalPos());
}

void MEColorPoint::removeCB()
{
    emit removePoint(this);
}

void MEColorPoint::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    p.setBackgroundMode(Qt::TransparentMode);

    QPen pen;
    if (state)
    {
        pen.setWidth(3);
        p.setPen(pen);
        p.drawPolyline(poly);
    }

    else
    {
        pen.setWidth(1);
    }
    p.setBrush(QBrush(color, Qt::SolidPattern));
    p.drawPolygon(poly);
}

/*!
   \class *
   \brief shows a histogram
*/

/*****************************************************************************
 *
 * Class MEHistogram 
 *
 *****************************************************************************/

MEHistogram::MEHistogram(MEColorMap *c, QWidget *p)
    : QWidget(p)
    , m_data(NULL)
    , colorMap(c)
{
    pen.setWidth(2);
}

MEHistogram::~MEHistogram()
{
    if (m_data)
        delete[] m_data;
}

//!
//! display a histogram
//!
void MEHistogram::update(int num, const float &xmin, const float &xmax, int *values)
{
    m_fmin = xmin;
    m_fmax = xmax;
    m_num = num;
    m_data = values;
    repaint();
}

//!
//! draw the color table
//!
void MEHistogram::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    if (!colorMap || !colorMap->initFlag)
        return;

    QPainter p(this);
    p.setPen(pen);

    // get minmax of colormap
    float cmin = colorMap->getMin();
    float cmax = colorMap->getMax();

    // set a background color to that part containing the histogram data
    float fx = float(width() / (cmax - cmin));
    qreal start = (m_fmin - cmin) * fx;
    qreal end = (m_fmax - cmin) * fx;
    p.fillRect(QRectF(start, 0, end - start, height()), Qt::lightGray);

    // find histogram maximum
    int ymax = m_data[0];
    for (int i = 1; i < m_num; i++)
        ymax = qMax(ymax, m_data[i]);
    float fy = (float)height() / (float)ymax;

    // draw histogram
    if (cmin <= m_fmin && cmax >= m_fmax)
    {
        qreal dx = (float)(end - start) / (float)m_num;
        for (int i = 0; i < m_num; i++)
        {
            qreal y = m_data[i] * fy;
            p.drawRect(QRectF(start, height() - y, dx, y));
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
            p.drawRect(QRectF(start, height() - y, dx, y));
            start = start + dx;
        }
    }
}
