/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#include "graphview.hpp"

#include "topviewgraph.hpp"
#include "graphscene.hpp"
#include "graphviewitems/graphviewshapeitem.hpp"
#include "src/cover/coverconnection.hpp"

#include <QtWidgets>
#include <QtNetwork>
#include <QUrl>

//MainWindow //
//
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/scenerysystem/scenerysystem.hpp"

// Items //
//
#include "items/view/ruler.hpp"
#include "items/scenerysystem/scenerysystemitem.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/toolmanager.hpp"
#include "src/gui/tools/zoomtool.hpp"
#include "src/gui/tools/selectiontool.hpp"
#include "src/gui/tools/maptool.hpp"
#include "src/gui/tools/junctioneditortool.hpp"
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/editors/osceditor.hpp"

// Qt //
//
#include <QWheelEvent>
#include <QMouseEvent>
#include <QFileDialog>
#include <QApplication>
#include <QUndoStack>
#include <QImage>
#include <QInputDialog>
#include <QXmlStreamReader>
#include <QFile>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>


// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

// DEFINES //
//
//#define USE_MIDMOUSE_PAN

GraphView::GraphView(GraphScene *graphScene, TopviewGraph *topviewGraph)
    : QGraphicsView(graphScene, topviewGraph)
    , topviewGraph_(topviewGraph)
    , graphScene_(graphScene)
    , doPan_(false)
    , doKeyPan_(false)
	, select_(true)
    , doBoxSelect_(BBOff)
    , doCircleSelect_(CircleOff)
    , doShapeEdit_(false)
    , radius_(0.0)
    , horizontalRuler_(NULL)
    , verticalRuler_(NULL)
    , rulersActive_(false)
    , rubberBand_(NULL)
    , circleItem_(NULL)
    , shapeItem_(NULL)
    , additionalSelection_(false)
{
    // ScenerySystem //
    //
    scenerySystemItem_ = new ScenerySystemItem(topviewGraph_, topviewGraph_->getProjectData()->getScenerySystem());
    scene()->addItem(scenerySystemItem_);

	// Zoom tool //
	//
	zoomTool_ = topviewGraph_->getProjectWidget()->getMainWindow()->getToolManager()->getZoomTool();

    // Zoom to mouse pos //
    //
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

	// Rubberband //
	//
	rubberBand_ = new QRubberBand(QRubberBand::Rectangle, this);

    // interactive background
    
    QPixmap pixmap("d:\\Pictures\\snapshot2.png");
    backgroundItem = new QGraphicsPixmapItem(pixmap);
    graphScene->addItem(backgroundItem);
	wgetInit();
}

GraphView::~GraphView()
{
    activateRulers(false);
}

/** Returns the scaling of the GraphicsView. Returns 0.0 if view is stretched.
*/
double
GraphView::getScale() const
{
    QMatrix vm = matrix();
    if (fabs(vm.m11()) != fabs(vm.m22()))
    {
        qDebug("View stretched! getScale() returns 0.0!");
        return 0.0;
    }
    else
    {
        return fabs(vm.m11());
    }
}

/*! \brief Resets the transformation of the view.
*
* \note The default view matrix is rotated 180 degrees around the x-Axis,
* because OpenDRIVE and Qt use different coordinate systems.
*/
void
GraphView::resetViewTransformation()
{
    QTransform trafo;
    trafo.rotate(180.0, Qt::XAxis);

    resetMatrix();
    setTransform(trafo);
}

void
GraphView::shapeEditing(bool edit)
{            
	if (edit)
	{
		doShapeEdit_ = true;
		shapeItem_ = new GraphViewShapeItem(this, x(), y(), width(), height());
		scene()->addItem(shapeItem_);
	}
	else if (doShapeEdit_)
	{
		doShapeEdit_ = false;
		//               splineControlPoints_.clear();
		if (scene())
		{
			scene()->removeItem(shapeItem_);
			delete shapeItem_;
		}
	}
}

//################//
// SLOTS          //
//################//

/*! \brief .
*
*/
void
GraphView::toolAction(ToolAction *toolAction)
{
	static QList<ODD::ToolId> selectionToolIds = QList<ODD::ToolId>() << ODD::TRL_SELECT << ODD::TRT_MOVE << ODD::TTE_ROAD_MOVE_ROTATE << ODD::TEL_SELECT << ODD::TSE_SELECT << ODD::TCF_SELECT << ODD::TLN_SELECT << ODD::TLE_SELECT << ODD::TJE_SELECT << ODD::TSG_SELECT << ODD::TOS_SELECT;

    // Zoom //
    //
    ZoomToolAction *zoomToolAction = dynamic_cast<ZoomToolAction *>(toolAction);
    if (zoomToolAction)
    {
        ZoomTool::ZoomToolId id = zoomToolAction->getZoomToolId();

        if (id == ZoomTool::TZM_ZOOMTO)
        {
            zoomTo(zoomToolAction->getZoomFactor());
        }
        else if (id == ZoomTool::TZM_ZOOMIN)
        {
            zoomIn();
        }
        else if (id == ZoomTool::TZM_ZOOMOUT)
        {
            zoomOut();
        }
        else if (id == ZoomTool::TZM_ZOOMBOX)
        {
            zoomBox();
        }
        else if (id == ZoomTool::TZM_VIEW_SELECTED)
        {
            viewSelected();
        }
        else if (id == ZoomTool::TZM_RULERS)
        {
            activateRulers(zoomToolAction->isToggled());
        }

    }

 
    // Circular Cutting Tool
    //
    JunctionEditorToolAction *junctionEditorAction = dynamic_cast<JunctionEditorToolAction *>(toolAction);
    if (junctionEditorAction)
    {
        ODD::ToolId id = junctionEditorAction->getToolId();

        if (id == ODD::TJE_CIRCLE)
        {
			if (doCircleSelect_ == CircleOff)
            {
                doCircleSelect_ = CircleActive;
                radius_ = junctionEditorAction->getThreshold();

                QPen pen(Qt::DashLine);
                pen.setColor(ODD::instance()->colors()->brightBlue());

                circleItem_ = new QGraphicsPathItem();
                circleItem_->setPen(pen);
                scene()->addItem(circleItem_);
            }
		}
		else
		{
			if (id == ODD::TJE_THRESHOLD)
			{
				radius_ = junctionEditorAction->getThreshold();
			}
			else if (circleItem_)
			{
				deleteCircle();
			}
		}

    }

    // Shape Editing Tool //
    //
    OpenScenarioEditorToolAction *oscEditorAction = dynamic_cast<OpenScenarioEditorToolAction *>(toolAction);
    if (oscEditorAction)
    {
        ODD::ToolId id = oscEditorAction->getToolId();

        if (id == ODD::TOS_GRAPHELEMENT)
        {
			shapeEditing(!doShapeEdit_);
        }
    }

    // Map //
    //
    MapToolAction *mapToolAction = dynamic_cast<MapToolAction *>(toolAction);
    if (mapToolAction)
    {
        MapTool::MapToolId id = mapToolAction->getMapToolId();

        if (id == MapTool::TMA_LOAD)
        {
            loadMap();
            lockMap(true);
        }
        else if (id == MapTool::TMA_GOOGLE)
        {
            loadGoogleMap();
            lockMap(false);
        }
        else if (id == MapTool::TMA_BING)
        {
            loadBingMap();
            lockMap(false);
        }
        else if (id == MapTool::TMA_DELETE)
        {
            deleteMap();
        }
        else if (id == MapTool::TMA_LOCK)
        {
            lockMap(mapToolAction->isToggled());
        }
        else if (id == MapTool::TMA_OPACITY)
        {
            setMapOpacity(mapToolAction->getOpacity());
        }
        //		else if(id == MapTool::TMA_X)
        //		{
        //			setMapX(mapToolAction->getX());
        //		}
        //		else if(id == MapTool::TMA_Y)
        //		{
        //			setMapY(mapToolAction->getY());
        //		}
        //		else if(id == MapTool::TMA_WIDTH)
        //		{
        //			setMapWidth(mapToolAction->getWidth(), mapToolAction->isKeepRatio());
        //		}
        //		else if(id == MapTool::TMA_HEIGHT)
        //		{
        //			setMapHeight(mapToolAction->getHeight(), mapToolAction->isKeepRatio());
        //		}
    }

	ODD::EditorId editorId = toolAction->getEditorId();
	if (editorId != ODD::ENO_EDITOR)
	{
		if (selectionToolIds.contains(toolAction->getToolId()))
		{
			select_ = true;
		}
		else
		{
			select_ = false;
		}
	}
}


void GraphView::setMap(float x,float y,float width,float height,int xRes,int yRes,const char *buf)
{
    
   // QPixmap pixmap(xRes,yRes);
    //pixmap.
    backgroundItem->setPixmap(QPixmap::fromImage(QImage((uchar *)buf,xRes,yRes,QImage::Format_RGBA8888)));
    backgroundItem->setPos(x,y);
    double widthScale = width / backgroundItem->pixmap().width();
    double heightScale = height / backgroundItem->pixmap().height();

    QTransform trafo;
    //trafo.rotate(180, Qt::XAxis);
    trafo.scale(widthScale, heightScale);

    backgroundItem->setTransform(trafo);
}

/**
*/
void
GraphView::rebuildRulers()
{
    QPointF pos = viewportTransform().inverted().map(QPointF(0.0, 0.0));
    double width = viewport()->size().width() / matrix().m11();
    double height = viewport()->size().height() / matrix().m22();

    COVERConnection::instance()->resizeMap(pos.x(),pos.y(),width,height);

    if (!rulersActive_)
    {
        return;
    }
    
    horizontalRuler_->updateRect(QRectF(pos.x(), pos.y(), width, height), matrix().m11(), matrix().m22());
    verticalRuler_->updateRect(QRectF(pos.x(), pos.y(), width, height), matrix().m11(), matrix().m22());
    update();
}

/**
*/
void
GraphView::activateRulers(bool activate)
{
    if (activate)
    {
        // Activate rulers //
        //
        if (!horizontalRuler_)
        {
            horizontalRuler_ = new Ruler(Qt::Horizontal);
            scene()->addItem(horizontalRuler_);
        }
        if (!verticalRuler_)
        {
            verticalRuler_ = new Ruler(Qt::Vertical);
            scene()->addItem(verticalRuler_);
        }
    }
    else
    {
        // Deactivate rulers //
        //
        if (horizontalRuler_)
        {
            scene()->removeItem(horizontalRuler_);
            delete horizontalRuler_;
            horizontalRuler_ = NULL;
        }
        if (verticalRuler_)
        {
            scene()->removeItem(verticalRuler_);
            delete verticalRuler_;
            verticalRuler_ = NULL;
        }
    }

    rulersActive_ = activate;
    rebuildRulers();
}

/**
*/
void
GraphView::zoomTo(const QString &zoomFactor)
{
    // Erase %-sign and parse to double
    double scaleFactor = zoomFactor.left(zoomFactor.indexOf(tr("%"))).toDouble() / 100.0;
    //	const QMatrix & vm = matrix();
    //	resetMatrix();
    //	translate(vm.dx(), vm.dy()); // this is 0.0 anyway!

    resetViewTransformation();
    scaleView(scaleFactor, scaleFactor);
}

/**
*/
void
GraphView::zoomIn()
{
    zoomIn(1.25);
}

/**
*/
void
GraphView::zoomIn(double zoom)
{
    if (zoom < 1.0)
        zoom = 1.0;

#define MAX_ZOOM_IN 50.0f
    if (getScale() * zoom >= MAX_ZOOM_IN)
    {
        zoom = MAX_ZOOM_IN / getScale();
    }
    scaleView(zoom, zoom);
    //		update();
    rebuildRulers();
#undef MAX_ZOOM_IN
}

/**
*/
void
GraphView::zoomOut()
{
    scaleView(0.8, 0.8);
    //	update();
    rebuildRulers();
}

void
GraphView::scaleView(qreal sx, qreal sy)
{
    scale(sx, sy);
    scaling_ = getScale();
}

/**
*/
void
GraphView::zoomBox()
{

    qDebug("GraphView::zoomBox() not yet implemented");
}

/**
*/
void
GraphView::viewSelected()
{
    QList<QGraphicsItem *> selectList = scene()->selectedItems();
    QRectF boundingRect = QRectF();

    foreach (QGraphicsItem *item, selectList)
    {
        boundingRect.operator|=(item->sceneBoundingRect());
    }

    fitInView(boundingRect, Qt::KeepAspectRatio);
}

/*! \brief .
*/
void
GraphView::loadMap()
{
    // File //
    //
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image File"));
    if (filename.isEmpty())
    {
        return;
    }

    scenerySystemItem_->loadMap(filename, mapToScene(10.0, 10.0)); // place pic near top left corner
}


/*! \brief .
*/
void
GraphView::loadGoogleMap()
{
    //May need this later, it's the formula Google uses to calculate the scale of the map
    //156543.03392 * Math.cos(latLng.lat() * Math.PI / 180) / Math.pow(2, zoom)
    //https://groups.google.com/forum/#!topic/google-maps-js-api-v3/hDRO4oHVSeM

    QString location;
    QString maptype;
    QString sizePair;
    QDir directoryOperator;
    bool mapRejected = false;

    //Sets up the UI

    QDialog dialog(this);
    dialog.setWindowTitle("Google Map Config");
    QFormLayout form(&dialog);
    form.addRow(new QLabel("Please enter location, map type, and size"));

    QList<QLineEdit *> fields;
    QLineEdit *lineEdit1 = new QLineEdit(&dialog);
    QString label = QString("Location (address or coordinates):");
    form.addRow(label, lineEdit1);
    fields << lineEdit1;

    QLineEdit *lineEdit2 = new QLineEdit(&dialog);
    QString label2 = QString("Map Type (satellite or roadmap):");
    form.addRow(label2, lineEdit2);
    lineEdit2->setText("satellite");
    fields << lineEdit2;

    QLineEdit *lineEdit3 = new QLineEdit(&dialog);
    QString label3 = QString("Size (XcommaY):");
    lineEdit3->setText("3,3");
    form.addRow(label3, lineEdit3);
    fields << lineEdit3;

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                               Qt::Horizontal, &dialog);
    form.addRow(&buttonBox);
    QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
    QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

    if (dialog.exec() == QDialog::Accepted) {
        location = lineEdit1->text();
        maptype  = lineEdit2->text();
        sizePair = lineEdit3->text();
        }
    else
        mapRejected = true;

    QStringList sizePairList = sizePair.split(",");
    QString sizeX = sizePairList.value(0);
    QString sizeY = sizePairList.value(1);

    //For debugging purposes
    //double offSet = QInputDialog::getDouble(this, tr("Offset?"),
    //                tr("Offset test"), 0.00, -1, 1, 10, &ok);




    if(!mapRejected){

        //This wget will download an XML file of the location entered by the user, including the latitude and longitude.
        //QString XMLlocationCommand = "wget -O location.xml 'https://maps.google.com/maps/api/geocode/xml?address=" + location + "&key=AIzaSyCvZVXlu-UfJdPUb6_66YHjyPj4qHKc_Wc'";

        //system(qPrintable(XMLlocationCommand));
		downloadFile("location.xml", "https://maps.google.com/maps/api/geocode/xml?address=" + location + "&key=AIzaSyCvZVXlu-UfJdPUb6_66YHjyPj4qHKc_Wc");

        QString lat;
        QString lon;

        QFile xmlFile("location.xml");
        if(!xmlFile.open(QFile::ReadOnly | QFile::Text))
            exit(0);

        QXmlStreamReader xmlReader(&xmlFile);


        //Finds latitude and longitude of the selected location by parsing downloaded XML document

        if (xmlReader.readNextStartElement())
        {
            if(xmlReader.name() == "GeocodeResponse")
            {
                while(xmlReader.readNextStartElement())
                {
                    if(xmlReader.name() == "status")
                    {
                        xmlReader.skipCurrentElement();
                    }
                    else if(xmlReader.name() == "result")
                    {
                        while(xmlReader.readNextStartElement())
                        {
                            if(xmlReader.name() != "geometry")
                            {
                                xmlReader.skipCurrentElement();
                            }
                            else if (xmlReader.name() == "geometry")
                                while(xmlReader.readNextStartElement())
                                {
                                    if(xmlReader.name() == "location"){
                                        while(xmlReader.readNextStartElement())
                                        {
                                            if(xmlReader.name() == "lat"){
                                                lat = xmlReader.readElementText();
                                            }
                                            if(xmlReader.name() == "lng"){
                                                lon = xmlReader.readElementText();
                                            }
                                        }
                                    }
                                }
                        }
                    }
                }
            }
        }

        //system(qPrintable("echo Y converted to " + QString::number(yPosition) + " X converted to " + QString::number(xPosition) + " z converted to " + QString::number(zPosition)));
        double dlat = lat.toDouble();
        double dlon = lon.toDouble();


        QString folderName = QString(QString::number(dlat) + QString::number(dlon) + maptype + sizeX + sizeY);
        directoryOperator.mkdir("OddlotMapImages");
        directoryOperator.setCurrent("OddlotMapImages");
        directoryOperator.mkdir(folderName);
        directoryOperator.setCurrent(folderName);
        //example format:
        //wget -O 'https://maps.googleapis.com/maps/api/staticmap?center=Stuttgart%20Vaihingen,Germany&zoom=16&size=1200x1200&scale=2'

        QString zoom = "19";
        QString style = "&style=feature:all|element:labels|visibility:off";
        //QString uploadPrefix = "wget -O ";
        QString uploadPrefix2 ="https://maps.googleapis.com/maps/api/staticmap?center=";
        QString uploadPostfix = "&zoom=" + zoom + "&maptype=" + maptype + style + "&size=1200x1200&scale=2&key=AIzaSyCvZVXlu-UfJdPUb6_66YHjyPj4qHKc_Wc";


        //this equation was calculated by calibrating the latitude offset to a variety of locations. this is the equation of the line of best fit.
        double xOffset = 0.00000000000509775733811385*dlat*dlat*dlat*dlat + 0.0000000000712116529947624*dlat*dlat*dlat
                - 0.000000249574727260668*dlat*dlat - 0.000000107541426772267*dlat + 0.0016557178;
        double yOffset = .00170;
        QString newLoc;
        double newLat;
        double newLon;
        QString filename;


        //doesn't work at all if one dimension is less than one, so, defaults to 3x3 if the user enters value less than 1
        double xSize = sizeX.toDouble();
        if(xSize < 2)
            xSize = 3;
        double ySize = sizeY.toDouble();
        if (ySize < 2)
            ySize = 3;
        int progress = xSize*ySize;

        //Grabs each image, and saves it to a file indicating its x,y coordinates (in the context of the map).
        //Uses the previously determined offsets to change the center of each image
        int i = 0;
        int j = 0;
        for (i = -xSize/2; i < xSize/2; i++)
        {
            for (j = -ySize/2; j < ySize/2; j++)
            {
                double latIterator = double(j);
                double lonIterator = double(i);
                newLat = dlat + -latIterator*xOffset;
                newLon = dlon + lonIterator*yOffset;
                newLoc = QString::number(newLat, 'f', 7)+ "," + QString::number(newLon, 'f', 7);
                //system(qPrintable(QString("echo Progress: " + QString::number(progress) + " images left.")));
                progress--;
                QString newFilename = QString(QDir().absolutePath() + "/image" + QString::number(i) + QString::number(j) + ".png");
                //QString command = uploadPrefix + newFilename + " " + uploadPrefix2 + newLoc + uploadPostfix;
                //system(qPrintable(command));

				downloadFile(newFilename, uploadPrefix2 + newLoc + uploadPostfix);

                double yPosition = (newLat) * DEG_TO_RAD;
                double xPosition = (newLon) * DEG_TO_RAD;
                double zPosition = 0.0;

                ProjectionSettings::instance()->transform(xPosition, yPosition, zPosition);

                scenerySystemItem_->loadGoogleMap(newFilename, xPosition - 63 + i*1.75, yPosition - 65 + -j*1.65);
            }
        }      

        xmlFile.remove();
        //system(qPrintable("echo Offset used: " + QString::number(xOffset, 'f', 7)));
        //system(qPrintable("echo Latitude used: " + QString::number(dlat, 'f', 7)));
        //system(qPrintable("echo Longitude used: " + QString::number(dlon, 'f', 7)));
        directoryOperator.setCurrent("..");
        directoryOperator.setCurrent("..");
    }
}
void
GraphView::loadBingMap()
{
    //Request XML with bounding box using:
    //http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/47.619048,-122.35384/15?mapSize=1500,1500&mapMetadata=1&o=xml&key=AlG2vgS1nf8uEEiq4ypPUu3Be-Mr1QOWiTj_lY55b8RAVNl7h3v1Bx0nTqavOJDm


    QString location;
    QString mapType;
    QString sizePair;
    QDir directoryOperator;
    bool mapRejected = false;

    //Sets up the UI

    QDialog dialog(this);
    dialog.setWindowTitle("Bing Map Config");
    QFormLayout form(&dialog);
    form.addRow(new QLabel("Please enter location, map type, and size"));

    QList<QLineEdit *> fields;
    QLineEdit *lineEdit1 = new QLineEdit(&dialog);
    QString label = QString("Location (address or coordinates):");
    form.addRow(label, lineEdit1);
    fields << lineEdit1;

    QLineEdit *lineEdit2 = new QLineEdit(&dialog);
    QString label2 = QString("Map Type (satellite or roadmap):");
    form.addRow(label2, lineEdit2);
    lineEdit2->setText("satellite");
    fields << lineEdit2;

    QLineEdit *lineEdit3 = new QLineEdit(&dialog);
    QString label3 = QString("Size (XcommaY):");
    lineEdit3->setText("3,3");
    form.addRow(label3, lineEdit3);
    fields << lineEdit3;

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                               Qt::Horizontal, &dialog);
    form.addRow(&buttonBox);
    QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
    QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

    if (dialog.exec() == QDialog::Accepted) {
        location = lineEdit1->text();
        mapType  = lineEdit2->text();
        sizePair = lineEdit3->text();
        }
    else
        mapRejected = true;

    QStringList sizePairList = sizePair.split(",");
    QString sizeX = sizePairList.value(0);
    QString sizeY = sizePairList.value(1);

    if(!mapRejected){

        //This wget will download an XML file of the location entered by the user, including the latitude and longitude.
        //Bing's API to turn a location into a set of coordinates isn't as flexible as Google's, so we'll keep using Google's system for this part.
       // QString XMLlocationCommand = "wget -O location.xml 'https://maps.google.com/maps/api/geocode/xml?address=" + location + "&key=AIzaSyCvZVXlu-UfJdPUb6_66YHjyPj4qHKc_Wc'";

       // system(qPrintable(XMLlocationCommand));

		downloadFile("location.xml", "https://maps.google.com/maps/api/geocode/xml?address=" + location + "&key=AIzaSyCvZVXlu-UfJdPUb6_66YHjyPj4qHKc_Wc");

        QString lat;
        QString lon;

        QFile xmlFile("location.xml");
        if(!xmlFile.open(QFile::ReadOnly | QFile::Text))
            exit(0);

        QXmlStreamReader xmlReader(&xmlFile);


        //Finds latitude and longitude of the selected location by parsing downloaded XML document

        if (xmlReader.readNextStartElement())
        {
            if(xmlReader.name() == "GeocodeResponse")
            {
                while(xmlReader.readNextStartElement())
                {
                    if(xmlReader.name() == "status")
                    {
                        xmlReader.skipCurrentElement();
                    }
                    else if(xmlReader.name() == "result")
                    {
                        while(xmlReader.readNextStartElement())
                        {
                            if(xmlReader.name() != "geometry")
                            {
                                xmlReader.skipCurrentElement();
                            }
                            else if (xmlReader.name() == "geometry")
                                while(xmlReader.readNextStartElement())
                                {
                                    if(xmlReader.name() == "location"){
                                        while(xmlReader.readNextStartElement())
                                        {
                                            if(xmlReader.name() == "lat"){
                                                lat = xmlReader.readElementText();
                                            }
                                            if(xmlReader.name() == "lng"){
                                                lon = xmlReader.readElementText();
                                            }
                                        }
                                    }
                                }
                        }
                    }
                }
            }
        }

        //system(qPrintable("echo Y converted to " + QString::number(yPosition) + " X converted to " + QString::number(xPosition) + " z converted to " + QString::number(zPosition)));

        QString folderName = lat + lon + mapType + sizeX + sizeY;
        directoryOperator.mkdir("OddlotMapImages");
        directoryOperator.setCurrent("OddlotMapImages");
        directoryOperator.mkdir(folderName);
        directoryOperator.setCurrent(folderName);
        //example format:
        //wget -O 'http://dev.virtualearth.net/REST/V1/Imagery/Map/Aerial?mapArea=48.737,9.097,48.742,9.098&ms=2500,2500&key=AlG2vgS1nf8uEEiq4ypPUu3Be-Mr1QOWiTj_lY55b8RAVNl7h3v1Bx0nTqavOJDm'

        //QString XMLlocationCommandBing = "wget -O locationBing.xml 'http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/" + lat + "," + lon + "/19?mapSize=1500,1500&mapMetadata=1&o=xml&key=AlG2vgS1nf8uEEiq4ypPUu3Be-Mr1QOWiTj_lY55b8RAVNl7h3v1Bx0nTqavOJDm'";

        //system(qPrintable(XMLlocationCommandBing));

		downloadFile("locationBing.xml", "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/" + lat + "," + lon + "/19?mapSize=1500,1500&mapMetadata=1&o=xml&key=AlG2vgS1nf8uEEiq4ypPUu3Be-Mr1QOWiTj_lY55b8RAVNl7h3v1Bx0nTqavOJDm");



        //Here, we parse the XML file Bing's API gives us to find the bounding coordinates of the tile, so we can get the centers of our other tiles.

        QFile xmlFileBing("locationBing.xml");
        if(!xmlFileBing.open(QFile::ReadOnly | QFile::Text))
            exit(0);

        QString boundingSouth;
        QString boundingWest;
        QString boundingNorth;
        QString boundingEast;

        QXmlStreamReader xmlReaderBing(&xmlFileBing);

        if (xmlReaderBing.readNextStartElement())
        {
            if(xmlReaderBing.name() == "Response")
            {
                while(xmlReaderBing.readNextStartElement())
                {
                    if(xmlReaderBing.name() != "ResourceSets")
                    {
                        xmlReaderBing.skipCurrentElement();
                    }
                    else
                    {
                        while(xmlReaderBing.readNextStartElement())
                        {
                            if(xmlReaderBing.name() != "ResourceSet")
                            {
                                xmlReaderBing.skipCurrentElement();
                            }
                            else
                            {
                                while(xmlReaderBing.readNextStartElement())
                                {
                                    if(xmlReaderBing.name() != "Resources")
                                    {
                                        xmlReaderBing.skipCurrentElement();
                                    }
                                    else
                                    {
                                        while(xmlReaderBing.readNextStartElement())
                                        {
                                            if(xmlReaderBing.name() == "StaticMapMetadata")
                                            {
                                                while(xmlReaderBing.readNextStartElement())
                                                {
                                                    if(xmlReaderBing.name() == "BoundingBox")
                                                    {
                                                        while(xmlReaderBing.readNextStartElement())
                                                        {
                                                            if(xmlReaderBing.name() == "SouthLatitude")
                                                            {
                                                                boundingSouth = xmlReaderBing.readElementText();
                                                            }
                                                            if(xmlReaderBing.name() == "WestLongitude")
                                                            {
                                                                boundingWest = xmlReaderBing.readElementText();
                                                            }
                                                            if(xmlReaderBing.name() == "NorthLatitude")
                                                            {
                                                                boundingNorth = xmlReaderBing.readElementText();
                                                            }
                                                            if(xmlReaderBing.name() == "EastLongitude")
                                                            {
                                                                boundingEast = xmlReaderBing.readElementText();
                                                            }
                                                        }
                                                    }
                                                }
                                            }


                                        }

                                    }
                                }
                            }
                    }
                }
            }
        }

        double boundingSouthNum = boundingSouth.toDouble();
        double boundingWestNum  = boundingWest.toDouble();
        double boundingNorthNum = boundingNorth.toDouble();
        double boundingEastNum  = boundingEast.toDouble();
        /*
         * TODO: calculate the absolute values of the numbers to figure out what quadrant of the world we're in. Important for size calculations.
        if(boundingNorthNum > boundingSouthNum)
        {

        }

        */


        double NorthSouthSize = boundingNorthNum - boundingSouthNum;
        double WestEastSize   = boundingEastNum  - boundingWestNum;

        QString uploadPrefix = "wget -O ";
        QString uploadPrefix2 ="http://dev.virtualearth.net/REST/V1/Imagery/Map/Aerial/";
        QString uploadPostfix = "/19?mapSize=1500,1500&key=AlG2vgS1nf8uEEiq4ypPUu3Be-Mr1QOWiTj_lY55b8RAVNl7h3v1Bx0nTqavOJDm";



        QString newLoc;
        double newLat;
        double newLon;
        QString filename;


        //doesn't work at all if one dimension is less than one, so, defaults to 3x3 if the user enters value less than 1
        //This should either be fixed later or some sort of notice should be given in the program that this is the minimum size.
        double xSize = sizeX.toDouble();
        if(xSize < 2)
            xSize = 3;
        double ySize = sizeY.toDouble();
        if (ySize < 2)
            ySize = 3;
       int progress = xSize*ySize;

        //Grabs each image, and saves it to a file indicating its x,y coordinates (in the context of the map).
        //Uses the previously determined offsets to change the center of each image
        int i = 0;
        int j = 0;
        for (i = -xSize/2; i < xSize/2; i++)
        {
            for (j = -ySize/2; j < ySize/2; j++)
            {
                double latIterator = double(j);
                double lonIterator = double(i);
                newLat = lat.toDouble() + -latIterator*NorthSouthSize;
                newLon = lon.toDouble() + lonIterator*WestEastSize;
                //system(qPrintable(QString("echo Progress: " + QString::number(progress) + " images left.")));
                newLoc = QString::number(newLat, 'f', 10)+ "," + QString::number(newLon, 'f', 10);
                progress--;
                QString newFilename = QString(QDir().absolutePath() + "/image" + QString::number(i) + QString::number(j) + ".jpg");
                QString command = uploadPrefix + newFilename + " " + uploadPrefix2 + newLoc + uploadPostfix;
                //system(qPrintable(command));
				downloadFile(newFilename, uploadPrefix2 + newLoc + uploadPostfix);


                double yPosition = (newLat-NorthSouthSize/2) * DEG_TO_RAD;
                double xPosition = (newLon-WestEastSize/2) * DEG_TO_RAD;
                double zPosition = 0.0;

                ProjectionSettings::instance()->transform(xPosition, yPosition, zPosition);

                scenerySystemItem_->loadBingMap(newFilename, xPosition, yPosition);
            }
        }
        xmlFile.remove();
        xmlFileBing.remove();

        system(qPrintable("echo Latitude used: " + lat));
        system(qPrintable("echo Longitude used: " + lon));
        directoryOperator.setCurrent("..");
        directoryOperator.setCurrent("..");
    }
}
}

/*! \brief .
*/
void
GraphView::deleteMap()
{
    scenerySystemItem_->deleteMap();
}

/*! \brief Locks all the MapItems (no selecting/moving).
*/
void
GraphView::lockMap(bool locked)
{
    scenerySystemItem_->lockMaps(locked);
}

/*! \brief Sets the opacity of the selected MapItems.
*/
void
GraphView::setMapOpacity(const QString &opacity)
{
    double opacityValue = opacity.left(opacity.indexOf(tr("%"))).toDouble() / 100.0;
    scenerySystemItem_->setMapOpacity(opacityValue);
}

/*! \brief Sets the x-coordinate of the selected MapItems.
*/
void
GraphView::setMapX(double x)
{
    scenerySystemItem_->setMapX(x);
}

/*! \brief Sets the y-coordinate of the selected MapItems.
*/
void
GraphView::setMapY(double y)
{
    scenerySystemItem_->setMapY(y);
}

/*! \brief Sets the width of the selected MapItems.
*/
void
GraphView::setMapWidth(double width, bool keepRatio)
{
    scenerySystemItem_->setMapWith(width, keepRatio);
}

/*! \brief Sets the height of the selected MapItems.
*/
void
GraphView::setMapHeight(double height, bool keepRatio)
{
    scenerySystemItem_->setMapHeight(height, keepRatio);
}

//################//
// EVENTS         //
//################//

/*! \brief Mouse events for panning, etc.
*
*/
void
GraphView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    rebuildRulers();
}

/*! \brief Mouse events for panning, etc.
*
*/
void
GraphView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    rebuildRulers();
}

void
GraphView::mousePressEvent(QMouseEvent *event)
{

    if (doKeyPan_)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
		QApplication::setOverrideCursor(Qt::OpenHandCursor);
        setInteractive(false); // this prevents the event from being passed to the scene
        QGraphicsView::mousePressEvent(event); // pass to baseclass
    }
#ifdef USE_MIDMOUSE_PAN
    else if (event->button() == Qt::MidButton)
    {
        doPan_ = true;

        setDragMode(QGraphicsView::ScrollHandDrag);
		QApplication::setOverrideCursor(Qt::OpenHandCursor);
        setInteractive(false); // this prevents the event from being passed to the scene

        // Harharhar Hack //
        //
        // Qt wants a LeftButton event for dragging, so feed Qt what it wants!
        QMouseEvent *newEvent = new QMouseEvent(QEvent::MouseMove, event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QGraphicsView::mousePressEvent(newEvent); // pass to baseclass
        delete newEvent;
        return;
    }
#endif
	else if (event->button() == Qt::LeftButton)
	{
		if (select_)
		{
			QGraphicsItem *item = NULL;
			if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) == 0)
			{
				item = scene()->itemAt(mapToScene(event->pos()), QGraphicsView::transform());
			}

			if (item)
			{
				QGraphicsView::mousePressEvent(event); // pass to baseclass
			}
			else
			{
				doBoxSelect_ = BBActive;

				if ((event->modifiers() & (Qt::ControlModifier | Qt::AltModifier)) != 0)
				{
					additionalSelection_ = true;
				}

				mp_ = event->pos();
			}
		}
		else if (doCircleSelect_ == CircleActive)
		{
			circleCenter_ = mapToScene(event->pos());
			QPainterPath circle = QPainterPath();
			circle.addEllipse(circleCenter_, radius_, radius_);
			circleItem_->setPath(circle);

			// Select roads intersecting with circle
			//
			scene()->setSelectionArea(circle);
		}
		else 
		{
			if (doShapeEdit_)
			{
				shapeItem_->mousePressEvent(event);
			}

			QGraphicsView::mousePressEvent(event); // pass to baseclass
		}
	}
}

void
GraphView::mouseMoveEvent(QMouseEvent *event)
{
    if (doBoxSelect_ == BBActive)
    {

        // Check for enough drag distance
        if ((mp_ - event->pos()).manhattanLength() < QApplication::startDragDistance())
        {
            return;
        }
		else
		{
			if (!rubberBand_->isVisible())
			{
				rubberBand_->show();
			}
		}

        QPoint ep = event->pos();

        rubberBand_->setGeometry(QRect(qMin(mp_.x(), ep.x()), qMin(mp_.y(), ep.y()),
                                       qAbs(mp_.x() - ep.x()) + 1, qAbs(mp_.y() - ep.y()) + 1));
    }
    else if (doKeyPan_)
    {
        QGraphicsView::mouseMoveEvent(event); // pass to baseclass
    }
#ifdef USE_MIDMOUSE_PAN
    else if (doPan_)
    {
        QMouseEvent *newEvent = new QMouseEvent(QEvent::MouseMove, event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QGraphicsView::mouseMoveEvent(newEvent); // pass to baseclass
        delete newEvent;
    }
#endif
	else if (doCircleSelect_ == CircleActive)
	{
		// Draw circle with radius and mouse pos center
		//
		circleCenter_ = mapToScene(event->pos());
		QPainterPath circle = QPainterPath();
		circle.addEllipse(circleCenter_, radius_, radius_);
		circleItem_->setPath(circle);

		QGraphicsView::mouseMoveEvent(event); // pass to baseclass
	}
	else
    {
		if (doShapeEdit_)
		{
			shapeItem_->mouseMoveEvent(event);
		}
        QGraphicsView::mouseMoveEvent(event); // pass to baseclass
    }
}

void
GraphView::mouseReleaseEvent(QMouseEvent *event)
{

	if (doKeyPan_)
	{
		setDragMode(QGraphicsView::NoDrag);
		setInteractive(true);
		if (doBoxSelect_ == BBPressed)
		{
			doBoxSelect_ = BBActive;
		}
		doKeyPan_ = false;
		QApplication::restoreOverrideCursor();
	}

#ifdef USE_MIDMOUSE_PAN
	else if (doPan_)
	{
		setDragMode(QGraphicsView::NoDrag);
		setInteractive(true);
		if (doBoxSelect_ == BBPressed)
		{
			doBoxSelect_ = BBActive;
		}
		doPan_ = false;
		QApplication::restoreOverrideCursor();
	}
#endif
	else if (doShapeEdit_)
	{
		shapeItem_->mouseReleaseEvent(event);

		QGraphicsView::mouseReleaseEvent(event);
	}
	//	setDragMode(QGraphicsView::RubberBandDrag);
	else if (!select_)
	{
		QGraphicsView::mouseReleaseEvent(event);
	}

	else
	{

		if (doBoxSelect_ == BBActive)
		{
			doBoxSelect_ = BBOff;

			if ((mp_ - event->pos()).manhattanLength() < QApplication::startDragDistance())
			{
				if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) != 0)
				{

					// Deselect element from the previous selection

					QList<QGraphicsItem *> oldSelection = scene()->selectedItems();

					QGraphicsView::mousePressEvent(event); // pass to baseclass

					QGraphicsItem *selectedItem = scene()->mouseGrabberItem();

					foreach (QGraphicsItem *item, oldSelection)
					{
						item->setSelected(true);
					}
					if (selectedItem)
					{
						if (((event->modifiers() & Qt::ControlModifier) != 0) && !oldSelection.contains(selectedItem))
						{
							selectedItem->setSelected(true);
						}
						else
						{
							selectedItem->setSelected(false);
						}
					}
				}
				else
				{
					QGraphicsView::mousePressEvent(event);
					QGraphicsView::mouseReleaseEvent(event);
				}
				return;
			}

			QList<QGraphicsItem *> oldSelection;

			if (additionalSelection_)
			{
				// Save old selection

				oldSelection = scene()->selectedItems();
			}

			// Set the new selection area

			QPainterPath selectionArea;

			selectionArea.addPolygon(mapToScene(QRect(rubberBand_->pos(), rubberBand_->rect().size())));
			selectionArea.closeSubpath();
			scene()->clearSelection();
			scene()->setSelectionArea(selectionArea, Qt::IntersectsItemShape, viewportTransform());

			// Compare old and new selection lists and invert the selection state of elements contained in both

			QList<QGraphicsItem *> selectList = scene()->selectedItems();
			foreach (QGraphicsItem *item, oldSelection)
			{
				if (selectList.contains(item))
				{
					item->setSelected(false);
					selectList.removeOne(item);
				}
				else
				{
					item->setSelected(true);
				}
			}

			// deselect elements which were not in the oldSelection

			if ((event->modifiers() & Qt::AltModifier) != 0)
			{
				foreach (QGraphicsItem *item, selectList)
				{
					item->setSelected(false);
				}
			}

			rubberBand_->hide();

		}

		if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) == 0)
		{
			QGraphicsView::mouseReleaseEvent(event);
		}

	}

	additionalSelection_ = false;
	setInteractive(true);


    //	if(doBoxSelect_)
    //	{
    //	}
    //	else if(doKeyPan_)
    //	{
    //	}
    //	else if(doPan_)
    //	{
    //	}
    //	else
    //	{
    ////	if(event->button() == Qt::MidButton) // end panning anyway
    ////	{
    //	}
}

void
GraphView::wheelEvent(QWheelEvent *event)
{
    if (event->delta() > 0)
    {
		zoomTool_->zoomIn();
    }
    else
    {
		zoomTool_->zoomOut();
    }


 //   QGraphicsView::wheelEvent(event);
}

void
GraphView::keyPressEvent(QKeyEvent *event)
{
    // TODO: This will not notice a key pressed, when the view is not active
    switch (event->key())
    {

    case Qt::Key_Delete:
    {
        // Macro Command //
        //
        int numberSelectedItems = scene()->selectedItems().size();
        if (numberSelectedItems > 1)
        {
            topviewGraph_->getProjectData()->getUndoStack()->beginMacro(QObject::tr("Delete Elements"));
        }
        bool deletedSomething = false;
        do
        {
            deletedSomething = false;
            QList<QGraphicsItem *> selectList = scene()->selectedItems();

            foreach (QGraphicsItem *item, selectList)
            {
                GraphElement *graphElement = dynamic_cast<GraphElement *>(item);
                if (graphElement)
                {
                    if (graphElement->deleteRequest())
                    {
                        deletedSomething = true;
                        break;
                    }
                }
            }
        } while (deletedSomething);

        // Macro Command //
        //
        if (numberSelectedItems > 1)
        {
            topviewGraph_->getProjectData()->getUndoStack()->endMacro();
        }
        break;
    }

    default:
        QGraphicsView::keyPressEvent(event);
    }
}

void
GraphView::keyReleaseEvent(QKeyEvent *event)
{
    /*switch (event->key())
    {

    default:
        QGraphicsView::keyReleaseEvent(event);
    }
    default:*/
        QGraphicsView::keyReleaseEvent(event);
    //}
}

void
GraphView::contextMenuEvent(QContextMenuEvent *event)
{
    if (doShapeEdit_)
    {
        shapeItem_->contextMenu(event);
//        GraphViewShapeItem::contextMenuEvent(event);
    }
	else
	{
		QGraphicsView::contextMenuEvent(event);
	}
}

void
	GraphView::deleteCircle()
{
	if (circleItem_)
	{
		doCircleSelect_ = CircleOff;
	    scene()->removeItem(circleItem_);
        delete circleItem_;
        circleItem_ = NULL;
	}
}


void GraphView::wgetInit()
{
	connect(&qnam, &QNetworkAccessManager::authenticationRequired,
		this, &GraphView::slotAuthenticationRequired);
#ifndef QT_NO_SSL
	connect(&qnam, &QNetworkAccessManager::sslErrors,
		this, &GraphView::sslErrors);
#endif
}

void GraphView::startRequest(const QUrl &requestedUrl)
{
	url = requestedUrl;
	httpRequestAborted = false;

	reply = qnam.get(QNetworkRequest(url));
	connect(reply, &QNetworkReply::finished, this, &GraphView::httpFinished);
	connect(reply, &QIODevice::readyRead, this, &GraphView::httpReadyRead);
}

void GraphView::downloadFile(const QString &fn, const QString &url)
{
	QUrl requestedUrl(url);
	QString fileName=fn;
	/*QString downloadDirectory = QDir::cleanPath("c:/tmp");
	bool useDirectory = !downloadDirectory.isEmpty() && QFileInfo(downloadDirectory).isDir();
	if (useDirectory)
		fileName.prepend(downloadDirectory + '/');*/
	if (QFile::exists(fileName))
	{
		QFile::remove(fileName);
	}

	file = openFileForWrite(fileName);
	if (!file)
		return;

	// schedule the request
	startRequest(requestedUrl);

	QTimer timeoutTimer;
	QEventLoop loop;
	//connect(reply, &QNetworkReply::finished, &loop, SLOT(quit()));
	connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
	connect(&timeoutTimer, SIGNAL(timeout()), &loop, SLOT(quit()));

	// wait for file to be downloaded
	timeoutTimer.start(100000);
	loop.exec(); //blocks untill either theSignalToWaitFor or timeout was fired
}

QFile *GraphView::openFileForWrite(const QString &fileName)
{
	QScopedPointer<QFile> file(new QFile(fileName));
	if (!file->open(QIODevice::WriteOnly)) {
		QMessageBox::information(this, tr("Error"),
			tr("Unable to save the file %1: %2.")
			.arg(QDir::toNativeSeparators(fileName),
				file->errorString()));
		return nullptr;
	}
	return file.take();
}

void GraphView::cancelDownload()
{
	httpRequestAborted = true;
	reply->abort();
}

void GraphView::httpFinished()
{
	QFileInfo fi;
	if (file) {
		fi.setFile(file->fileName());
		file->close();
		delete file;
		file = nullptr;
	}

	if (httpRequestAborted) {
		reply->deleteLater();
		reply = nullptr;
		return;
	}

	if (reply->error()) {
		QFile::remove(fi.absoluteFilePath());
		reply->deleteLater();
		reply = nullptr;
		return;
	}

	const QVariant redirectionTarget = reply->attribute(QNetworkRequest::RedirectionTargetAttribute);

	reply->deleteLater();
	reply = nullptr;

	if (!redirectionTarget.isNull()) {
		const QUrl redirectedUrl = url.resolved(redirectionTarget.toUrl());
		if (QMessageBox::question(this, tr("Redirect"),
			tr("Redirect to %1 ?").arg(redirectedUrl.toString()),
			QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
			QFile::remove(fi.absoluteFilePath());
			return;
		}
		file = openFileForWrite(fi.absoluteFilePath());
		if (!file) {
			return;
		}
		startRequest(redirectedUrl);
		return;
	}

}

void GraphView::httpReadyRead()
{
	// this slot gets called every time the QNetworkReply has new data.
	// We read all of its new data and write it into the file.
	// That way we use less RAM than when reading it at the finished()
	// signal of the QNetworkReply
	if (file)
		file->write(reply->readAll());
}

void GraphView::slotAuthenticationRequired(QNetworkReply *, QAuthenticator *authenticator)
{
	/*QDialog authenticationDialog;
	Ui::Dialog ui;
	ui.setupUi(&authenticationDialog);
	authenticationDialog.adjustSize();
	ui.siteDescription->setText(tr("%1 at %2").arg(authenticator->realm(), url.host()));*/

	// Did the URL have information? Fill the UI
	// This is only relevant if the URL-supplied credentials were wrong
	/*ui.userEdit->setText(url.userName());
	ui.passwordEdit->setText(url.password());

	if (authenticationDialog.exec() == QDialog::Accepted) {
		authenticator->setUser(ui.userEdit->text());
		authenticator->setPassword(ui.passwordEdit->text());
	}*/
}

#ifndef QT_NO_SSL
void GraphView::sslErrors(QNetworkReply *, const QList<QSslError> &errors)
{
	QString errorString;
	foreach(const QSslError &error, errors) {
		if (!errorString.isEmpty())
			errorString += '\n';
		errorString += error.errorString();
	}

	if (QMessageBox::warning(this, tr("SSL Errors"),
		tr("One or more SSL errors has occurred:\n%1").arg(errorString),
		QMessageBox::Ignore | QMessageBox::Abort) == QMessageBox::Ignore) {
		reply->ignoreSslErrors();
	}
}
#endif