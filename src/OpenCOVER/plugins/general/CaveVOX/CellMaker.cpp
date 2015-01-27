/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <math.h>
#include <time.h>

// VTK:
#include <vtk/vtkPolyDataMapper.h>
#include <vtk/vtkSphereSource.h>
#include <vtk/vtkProperty.h>

// Virvo:
#include <vvtoolshed.h>

// Local:
#include "CellMaker.H"
#include "vtkActorToOSG.H"

using namespace osg;
using namespace cui;

CellMaker::CellMaker(osgDrawObj *osgObj, Interaction *interaction)
    : VolumePickBox(interaction, osgObj, Vec3(-1, -1, -1), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN)
{
    // Add cellMaker to scene graph:
    _osgObj = osgObj;
    _osgObj->addWorldChild(_node.get());

    _node->setNodeMask(~1); // ~1=1110; ~0=1111; 0=0000; ~1=sichtbar, kein isect; ~0=sichtbar+isect

    _fp = NULL;
    _isVirvo = false;

    // Set the initial seed:
    srand((unsigned)time(NULL));
}

CellMaker::~CellMaker()
{
}

void CellMaker::init()
{
    std::vector<SphereData *>::iterator iter;
    for (iter = _mSpheres.begin(); iter != _mSpheres.end(); ++iter)
    {
        _scale->removeChild((*iter)->_node);
    }
    _mSpheres.clear();

    for (int i = 0; i < _numCells; i++)
    {
        //    cerr << endl << "creating cell " << i+1 << endl;
        SphereData *newSphere = makeRandomCell(
            -0.5f, 0.5f,
            -0.5f, 0.5f,
            -0.5f, 0.5f);
        displaySphere(newSphere);
    }
}

void CellMaker::displaySphere(SphereData *sphere)
{
    sphere->_node = makeOSGSphere(sphere);
    _scale->addChild(sphere->_node);

    _mSpheres.push_back(sphere);
    //  sphere->print();

    //  Node* tmpNode = vtkActorToOSG(makeCell(sphere), true);
    //  _scale->addChild(tmpNode);
}

/**
 * Creates a random cell between the parameters.
 * @param minRadius
 * @param maxRadius
 * @param minX
 * @param maxX
 * @param minY
 * @param maxY
 * @param minZ
 * @param maxZ
 * @param red
 * @param green
 * @param blue
 * @return
 */
CellMaker::SphereData *CellMaker::makeRandomCell(
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ)
{
    SphereData *sphere = new SphereData();

    sphere->radius = (float)getRandom(_minRadius, _maxRadius);
    Vec3 position = getCollisionsFreeRandomPosition(sphere->radius, minX, maxX, minY, maxY, minZ, maxZ);
    sphere->x = position[0];
    sphere->y = position[1];
    sphere->z = position[2];
    return sphere;
}

/**
 * Returns a random position that is between the parameters.
 * @param minX
 * @param maxX
 * @param minY
 * @param maxY
 * @param minZ
 * @param maxZ
 * @return
 */
Vec3 CellMaker::getRandomPosition(float radius,
                                  float minX, float maxX,
                                  float minY, float maxY,
                                  float minZ, float maxZ)
{
    double x = getRandom(minX + radius, maxX - radius);
    double y = getRandom(minY + radius, maxY - radius);
    double z = getRandom(minZ + radius, maxZ - radius);

    Vec3 position;
    position[0] = x;
    position[1] = y;
    position[2] = z;

    return position;
}

/**
 * Returns a position of a sphere that doesn't intersect with any other spheres we have.
 * @param radius
 * @param minX
 * @param maxX
 * @param minY
 * @param maxY
 * @param minZ
 * @param maxZ
 * @return
 */
Vec3 CellMaker::getCollisionsFreeRandomPosition(
    float radius,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ)
{
    Vec3 position1 = getRandomPosition(radius, minX, maxX, minY, maxY, minZ, maxZ);

    // look through all the spheres to see if there is a collision
    for (int i = 0; i < _mSpheres.size(); i++)
    {
        Vec3 position2(_mSpheres[i]->x, _mSpheres[i]->y, _mSpheres[i]->z);

        // if there's a collision, then start over again
        if (isCollision(position1, radius, position2, _mSpheres[i]->radius))
        {
            position1 = getRandomPosition(radius, minX, maxX, minY, maxY, minZ, maxZ);
            i = 0;
        }
    }
    return position1;
}

/**
 * Determines if there is a collision between two points and their radii.
 * @param position1
 * @param radius1
 * @param position2
 * @param radius2
 * @return
 */
bool CellMaker::isCollision(Vec3 position1, float radius1, Vec3 position2, float radius2)
{
    bool isCollision = false;

    if (getDistance(position1, position2) < fabs(radius1 + radius2))
    {
        isCollision = true;
    }

    return isCollision;
}

/**
 * Returns the distance between two points.
 * @param position1
 * @param position2
 * @return
 */
float CellMaker::getDistance(Vec3 position1, Vec3 position2)
{
    double xDifference = position1[0] - position2[0];
    double yDifference = position1[1] - position2[1];
    double zDifference = position1[2] - position2[2];

    return (float)sqrt(xDifference * xDifference + yDifference * yDifference + zDifference * zDifference);
}

/**
 * Gets a random number between min and max.
 * @param min
 * @param max
 * @return
 */
double CellMaker::getRandom(float min, float max)
{
    // number between 0.0 and 1.0
    double randNum = float(rand()) / float(RAND_MAX);
    float difference = max - min;
    return (difference * randNum) + min;
}

/**
 * Creates a sphere from the given parameters.  This also adds the sphere to our
 * interal vector of spheres.
 * 
 * @param x
 * @param y
 * @param z
 * @param radius
 * @param red
 * @param green
 * @param blue
 * @return The actor that is the sphere.
 */
vtkActor *CellMaker::makeCell(SphereData *sphere)
{
    vtkSphereSource *sphereSource = vtkSphereSource::New();
    sphereSource->SetThetaResolution(6); // 8 is default
    sphereSource->SetPhiResolution(6); // 8 is default
    sphereSource->SetRadius(sphere->radius);

    vtkPolyDataMapper *sphereMapper = vtkPolyDataMapper::New();
    sphereMapper->SetInput(sphereSource->GetOutput());

    vtkActor *sphereActor = vtkActor::New();
    sphereActor->SetMapper(sphereMapper);
    sphereActor->SetPosition(sphere->x, sphere->y, sphere->z);
    sphereActor->GetProperty()->SetColor(_r, _g, _b);

    return sphereActor;
}

Node *CellMaker::makeOSGSphere(SphereData *sphere)
{
    const float SCALE_FACTOR = 4.0f;
    float biggerRadius; // use bigger radius than necessary for better normals

    Vec4 color(_r, _g, _b, 1.0f);

    // Set tessellation hints:
    // setDetailRatio is a factor to multiply the default values for
    // numSegments (40) and numRows (20).
    // They won't go below the minimum values of MIN_NUM_SEGMENTS = 5, MIN_NUM_ROWS = 3
    TessellationHints *hints = new TessellationHints();
    hints->setDetailRatio(0.2f);

    Geode *sphereGeode = new Geode();
    biggerRadius = sphere->radius * SCALE_FACTOR;
    Sphere *sphereGeometry = new Sphere(Vec3(sphere->x * SCALE_FACTOR, sphere->y * SCALE_FACTOR, sphere->z * SCALE_FACTOR), biggerRadius);
    ShapeDrawable *sphereDrawable = new ShapeDrawable(sphereGeometry);
    sphereDrawable->setColor(color);
    sphereDrawable->setUseDisplayList(false); // allow changes to color and shape
    sphereDrawable->setTessellationHints(hints);
    sphereGeode->addDrawable(sphereDrawable);

    // Set state set so as to avoid lighting errors when scaling:
    StateSet *stateSet = sphereGeode->getOrCreateStateSet();
    stateSet->setMode(GL_RESCALE_NORMAL, StateAttribute::ON);

    // Use scale node to correct for bigger radius:
    MatrixTransform *scaleNode = new MatrixTransform();
    Matrix mat;
    mat.makeScale(1.0f / SCALE_FACTOR, 1.0f / SCALE_FACTOR, 1.0f / SCALE_FACTOR);
    scaleNode->setMatrix(mat);
    scaleNode->addChild(sphereGeode);

    return scaleNode;
}

/**
 * Writes and OBJ Wavefront file and a .sphere file that stores the radius
 * followed by the x, y, and z position on each line.
 * @param dataSetName Name the file will be saved as.
 *
void CellMaker::export(std::string dataSetName) 
{
	// write the OBJ
	writeOBJ(dataSetName);
		
	// write a file with the center of the spheres and the radii
	std::string fileName = dataSetName + ".sphere";
	writePositionAndRadii(fileName);
}
*/
/**
 * On each line, writes the radii followed by the position:
 *      [radius] [x] [y] [y]
 * @param fileName Name the file will be saved as.
 *
void CellMaker::writePositionAndRadii(std::string fileName) 
{
	std::string buffer = "";
		
	for(int i=0; i<_mCurrSphere; i++) 
	{
		buffer += _mSpheres[i].radius + " " + 
				  _mSpheres[i].x + " " +
				  _mSpheres[i].y + " " +
				  _mSpheres[i].z + "\n";
	}
		
	try {
		Writer out = new OutputStreamWriter(new FileOutputStream(fileName));
		out.write(buffer);
		out.close();
	} catch (FileNotFoundException e) {
		e.printStackTrace();
	} catch (IOException e) {
		e.printStackTrace();
	}
}
*/
/**
 * @param dataSetName
 *
void CellMaker::writeOBJ(std::string dataSetName) 
{
	vtkOBJExporter exporter = new vtkOBJExporter();
	exporter.SetFilePrefix(dataSetName);
				
	exporter.SetInput(_mRenderWindow);
	exporter.Write();
}
*/

void CellMaker::showCells(bool vis)
{
    vis ? (_node->setNodeMask(~1))
        : (_node->setNodeMask(0));
}

void CellMaker::setNumCells(int num)
{
    _numCells = num;
}

void CellMaker::setExtremes(float min, float max)
{
    _minRadius = min;
    _maxRadius = max;
}

void CellMaker::setColorRGB(Vec4 color)
{
    _r = color[0];
    _g = color[1];
    _b = color[2];
}

/** converts HLS color value to the class members _r, _g, _b
*/
void CellMaker::setColorHueToRGB(float hue)
{
    vvToolshed::HSBtoRGB(hue, 1, 1, &_r, &_g, &_b);
}

void CellMaker::trackballRotation(float x, float y, Matrix &m2w)
{
    VolumePickBox::trackballRotation(x, y, m2w);
    //  updateSpheresColor();
}

void CellMaker::cursorUpdate(InputDevice *dev)
{
    if (_isNavigating && _moveThresholdReached)
    {
        processMoveInput(dev);
        //	  updateSpheresColor();
    }

    VolumePickBox::cursorUpdate(dev);
}

void CellMaker::updateSpheresColor()
{
    cerr << "updateSpheresColor" << endl;

    DistanceVisitor dv;
    _scale->accept(dv);

    SpheresColorVisitor scv;
    scv.setDistances(dv.getDistances());
    _scale->accept(scv);

    /*
  float min_z = -1000.0f;
  float max_z = 1000.0f;
  vector<float> color_scale(1000);  //_mSpheres.size());

  osg::NodeVisitor nv(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN);
  _scale->accept(nv);
  vector<Node*> childs = nv.getNodePath();
  cerr << "NodePath length: " << childs.size() << " validNodeMask: " << nv.validNodeMask(*_scale.get()) << endl << "visitor type: " << nv.getVisitorType() << endl;
  vector<Node*>::iterator iter;
  for (iter=childs.begin(); iter!=childs.begin(); ++iter)
  {
    MatrixTransform* transform = dynamic_cast<MatrixTransform*>(*iter);
    if (transform)
    { 
      Matrix tmp = transform->getMatrix() * _scale->getMatrix();
      min_z = min(min_z, tmp.getTrans().z());
      max_z = max(max_z, tmp.getTrans().z());
    }
  }
  for (iter=childs.begin(); iter!=childs.begin(); ++iter)
  {
    Geode* geode = dynamic_cast<Geode*>(*iter);
    if (geode) 
    {
      cerr << geode->getNumDrawables() << " Drawables attached to Geode" << endl;
      ShapeDrawable* sphereDrawable = dynamic_cast<ShapeDrawable*>(geode->getDrawable(0));
      sphereDrawable->setColor(Widget::COL_BLUE);
    }
  }
*/
}

/// @return true if written ok, else false
bool CellMaker::writeSpheresFile(const char *filename)
{
    _fp = fopen(filename, "wb");
    if (_fp == NULL)
    {
        cerr << "Failed creating spheres file " << filename << endl;
        return false;
    }
    std::vector<SphereData *>::iterator iter;
    int i = 0;
    for (iter = _mSpheres.begin(); iter != _mSpheres.end(); ++iter)
    {
        fprintf(_fp, "SPHERE\t%f\t%f\t%f\t%f\n", (*iter)->x, (*iter)->y, (*iter)->z, (*iter)->radius);
        i++;
    }
    fclose(_fp);
    cerr << "Spheres file created: " << filename << endl;
    return true;
}

bool CellMaker::readSpheresFile(const char *filename)
{
    float x, y, z, radius;
    int i, err;

    // Remove previously existing spheres:
    std::vector<SphereData *>::iterator iter;
    for (iter = _mSpheres.begin(); iter != _mSpheres.end(); ++iter)
    {
        _scale->removeChild((*iter)->_node);
    }
    _mSpheres.clear();

    // Open file:
    _fp = fopen(filename, "rb");
    if (_fp == NULL)
    {
        cerr << "Failed opening spheres file " << filename << endl;
        return false;
    }

    // Read spheres:
    do
    {
        err = fscanf(_fp, "SPHERE\t%f\t%f\t%f\t%f\n", &x, &y, &z, &radius);
        if (err != EOF)
        {
            SphereData *newSphere = new SphereData();
            newSphere->x = x;
            newSphere->y = y;
            newSphere->z = z;
            newSphere->radius = radius;
            displaySphere(newSphere);
        }
    } while (err != EOF);
    fclose(_fp);
    setNumCells(_mSpheres.size());

    cerr << "Spheres file read: " << filename << endl;
    return true;
}

void CellMaker::addMarkerByHand(Marker *m)
{
    /*  if (collisionDetection(m)!=NULL)
  {
    std::vector<Marker*>::iterator iter;
    for (iter=_markers.begin(); iter!=_markers.end(), ++iter)
    {
      if (collisionDetection(*iter)!=NULL) break;
    }
    //Vec4 color(1.0f, 0.0f, 0.0f, 1.0f);
    //m->setColor(color);
  }
  else
*/
    if (collisionDetection(m) == NULL)
    {
        //	cerr << "bad marker placement" << endl;
    }
    VolumePickBox::addMarkerByHand(m);
}

/** checks the present markers for correct placing and colors them accordingly
*/
void CellMaker::checkMarkerCollisions()
{
    std::vector<Marker *>::iterator mIter;
    int i = 0;
    for (mIter = _markers.begin(); mIter != _markers.end(); i++, ++mIter)
    {
        cerr << "Marker " << i << endl;
        SphereData *collSphere = collisionDetection(*mIter);
        if (collSphere == NULL) // no sphere intersection
        {
            cerr << "bad marker placement" << endl;
            (*mIter)->setColor(Widget::COL_RED);
        }
        else
        {
            std::vector<SphereData *>::iterator sIter;
            for (sIter = _mSpheres.begin(); sIter != _mSpheres.end(); ++sIter)
            {
                if ((*sIter) == collSphere)
                {
                    (*sIter)->markersInside++;
                    if ((*sIter)->markersInside > 1) // multiple markers intersected with sphere
                    {
                        cerr << "multiple marker placement at sphere" << endl;
                        (*mIter)->setColor(Widget::COL_LIGHT_GRAY);
                    }
                }
            }
        }
    }
}

CellMaker::SphereData *CellMaker::collisionDetection(Marker *m)
{
    Vec3 coneTip = (m->getMatrix()).getTrans();
    Vec3 sphereCenter;
    bool collision;
    float minDistance = 1000.0f;

    std::vector<SphereData *>::iterator iter;
    for (iter = _mSpheres.begin(); iter != _mSpheres.end(); ++iter)
    {
        sphereCenter.set((*iter)->x, (*iter)->y, (*iter)->z);
        //    sphereCenter=sphereCenter*_scale;
        Vec3 vec = coneTip - sphereCenter;
        collision = (vec.length() < (*iter)->radius);
        if (collision)
            return (*iter);
        else
        {
            minDistance = min(minDistance, vec.length() - (*iter)->radius);
        }
    }
    cerr << "nearest distance to any sphere: " << minDistance << endl;

    return NULL;
}

/*********************************************************************/

/** obsolete: Creates an ellipsoid of given size along the axes.
*
vtkActor* CellMaker::makeCell(Vec3& pos)
{
  vtkSphereSource* sphereSource = vtkSphereSource::New();
  sphereSource->SetThetaResolution(100);
  sphereSource->SetPhiResolution(50);

  vtkPolyDataMapper* sphereMapper = vtkPolyDataMapper::New();
  sphereMapper->SetInput(sphereSource->GetOutput());

  _sphere = vtkActor::New();
  _sphere->SetMapper(sphereMapper);
  _sphere->GetProperty()->SetColor(1,1,1);
  _sphere->GetProperty()->SetAmbient(0.3);
  _sphere->GetProperty()->SetDiffuse(0.0);
  _sphere->GetProperty()->SetSpecular(1.0);
  _sphere->GetProperty()->SetSpecularPower(5.0);
  _sphere->AddPosition(pos[0], pos[1], pos[2]);
  return _sphere;
}
*/
