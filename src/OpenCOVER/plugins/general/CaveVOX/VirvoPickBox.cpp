/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Virvo:
#include <vvvecmath.h>
#include <vvrenderer.h>
#include <vvtexrend.h>

// OSG:
#include <osg/Geode>
#include <osg/LineSegment>
#include <osg/Switch>

// CUI:
#include <CUI.H>
#include <Measure.H>

// Local:
#include "VirvoPickBox.H"
#include "vtkActorToOSG.H"
#include "vtkVolumeGeode.H"

// VTK:
#include <vtk/vtkStructuredPoints.h>
#include <vtk/vtkContourFilter.h>

using namespace cui;
using namespace osg;

VirvoPickBox::VirvoPickBox(Interaction *interaction, osgDrawObj *osgObj,
                           const Vec3 &min, const Vec3 &max, const Vec4 &c1, const Vec4 &c2,
                           const Vec4 &c3, Virvo::AlgorithmType alg)
    :

    VolumePickBox(interaction, osgObj, min, max, c1, c2, c3)
{
    _virvoDrawable = new Virvo(alg);
    _geode = new Geode();
    _geode->addDrawable(_virvoDrawable.get());
    _geode->setNodeMask(~1);
    _scale->addChild(_geode.get());
    _vtkNode = new MatrixTransform();
    _vtkSwitch = new Switch();
    Matrix rot;
    rot.makeRotate(M_PI, 1.0, 0.0, 0.0);
    _vtkNode->setMatrix(rot);
    _vtkNode->addChild(_vtkSwitch.get());
    _vtkNode->setNodeMask(~1);
    _vtkThickness = new MatrixTransform();
    Matrix xf;
    xf.makeIdentity(); // XXX -shouldn't be identity .. need to get
    // default value and use it here...  is it 0.8?
    _vtkThickness->setMatrix(xf);
    //  _scale->addChild(_vtkThickness.get());
    //  _vtkThickness->addChild(_vtkNode.get());
    _isoValue = 0.5f;
    _isoColor = 0.5f;
    _isVirvo = true;
}

VirvoPickBox::~VirvoPickBox()
{
    _scale->removeChild(_geode.get());

    /*
    _vtkNode->removeChild(_vtkSwitch.get());
    while(_vtkVolumes.num())
    {
      delete _vtkVolumes.pop();
    }
  */
}

Virvo *VirvoPickBox::getDrawable()
{
    return _virvoDrawable.get();
}

vvRenderer *VirvoPickBox::getRenderer()
{
    return _virvoDrawable->getRenderer();
}

vvVolDesc *VirvoPickBox::getVD()
{
    return _virvoDrawable->getVD();
}

bool VirvoPickBox::loadVolumeFile(const char *filename)
{
    bool ret;

    ret = _virvoDrawable->loadVolumeFile(filename);
    updateBoundingBox();

    /* 
    if (_vtkNode->getNodeMask() != 0)   // was previously an isosurface displayed?
    {
      recreateVTKIsosurfaces();
      updateIsoGeometry();
    }
  */

    return ret;
}

void VirvoPickBox::updateBoundingBox()
{
    BoundingBox box;
    _virvoDrawable->updateBoundingBox();
    box = _virvoDrawable->getBound();
    Vec3 size = box._max - box._min;
    setBoxSize(size);
    _measure->setBBox(_bbox);

    computeLocalToVoxel();
}

void VirvoPickBox::cursorUpdate(InputDevice *dev)
{
    bool roiInteraction = false;
    bool clipInteraction = false;
    vvVector3 probeSize, vProbeSize, vProbeSize2;
    int i;

    _virvoDrawable->getRenderer()->getProbeSize(&probeSize);

    if (_isNavigating && _moveThresholdReached)
    {
        // Compute pointer end points in world space:
        Vec3 wPointerStart = _lastWand2w.getTrans();
        double *mat = _lastWand2w.ptr();
        Vec3 wPointerDir(mat[8], mat[9], mat[10]);
        wPointerDir.normalize();
        Vec3 wPointerEnd(wPointerStart + wPointerDir * 100.0f);

        // Compute pointer end points in volume space:
        Matrix w2v = Matrix::inverse(getB2W());
        Vec3 vPointerStart = wPointerStart * w2v;
        Vec3 vPointerEnd = wPointerEnd * w2v;

        // Compute difference matrix between last and current wand:
        Matrix invLastWand2w = Matrix::inverse(_lastWand2w);
        Matrix wDiff = invLastWand2w * dev->getI2W();
        Matrix vDiff = getB2W() * wDiff * w2v;

        // button pressed in ROI mode?
        if (_virvoDrawable->getRenderer()->isROIEnabled())
        {
            // Compute ROI bounding box in volume space:
            vvVector3 size = _virvoDrawable->getVD()->getSize();
            for (i = 0; i < 3; ++i)
            {
                vProbeSize[i] = probeSize[i] * size.e[i];
            }
            vProbeSize2.copy(&vProbeSize);
            vProbeSize2.scale(0.5f);
            vvVector3 vv_vPos;
            _virvoDrawable->getRenderer()->getProbePosition(&vv_vPos);
            Vec3 vPos(vv_vPos.e[0], vv_vPos.e[1], vv_vPos.e[2]);
            Vec3 min, max;
            min.set(vPos[0] - vProbeSize2[0], vPos[1] - vProbeSize2[1], vPos[2] - vProbeSize2[2]);
            max.set(vPos[0] + vProbeSize2[0], vPos[1] + vProbeSize2[1], vPos[2] + vProbeSize2[2]);
            BoundingBox vROI(min, max);

            // Create line segment:
            ref_ptr<LineSegment> vLine = new LineSegment();
            vLine->set(vPointerStart, vPointerEnd);

            // Test intersection of ROI box and pointer line:
            if (vLine->intersect(vROI))
            {
                // ROI follows wand movement:
                vPos = vPos * vDiff;
                vv_vPos.set(vPos[0], vPos[1], vPos[2]);
                _virvoDrawable->getRenderer()->setProbePosition(&vv_vPos);
                roiInteraction = true;

                // Change ROI size with wand twist:
                Matrix i2w = dev->getI2W();
                float angle = angleDiffZ(_lastWand2w, i2w);
                probeSize.scale((1.0f + angle / 180.0f));
                if (probeSize[0] < 0.01f) // limit to minimum size
                    probeSize[0] = 0.01f;
                else if (probeSize[1] < 0.01f)
                    probeSize[1] = 0.01f;
                else if (probeSize[2] < 0.01f)
                    probeSize[2] = 0.01f;

                _virvoDrawable->getRenderer()->setProbeSize(&probeSize);
                _virvoDrawable->getRenderer()->setProbeColor(1.0f, 0.0f, 0.0f);
            }
            else
            {
                _virvoDrawable->getRenderer()->setProbeColor(1.0f, 1.0f, 1.0f);
            }
        }

        // Clipping plane interaction:
        if (!roiInteraction && _virvoDrawable->getRenderer()->_renderState._clipMode)
        {
            clipInteraction = true;

            // Find out if pointer intersects clipping plane:
            vvVector3 normal, point, start, end, isect;
            _virvoDrawable->getRenderer()->_renderState.getClippingPlane(&point, &normal);
            start.set(vPointerStart[0], vPointerStart[1], vPointerStart[2]);
            end.set(vPointerEnd[0], vPointerEnd[1], vPointerEnd[2]);

            // Find intersection of clipping plane and pointer line and let clipping plane follow
            // wand movement:
            if (isect.isectPlaneLine(&normal, &point, &start, &end))
            {
                if (_virvoDrawable->getRenderer()->isInVolume(&isect))
                {
                    // Convert to OSG vectors:
                    Vec3 vIsect(isect.e[0], isect.e[1], isect.e[2]);
                    Vec3 vNormal(normal.e[0], normal.e[1], normal.e[2]);

                    // Compute second point along normal:
                    Vec3 vPoint2 = vNormal + vIsect;

                    // Transform both points:
                    vIsect = vIsect * vDiff;
                    vPoint2 = vPoint2 * vDiff;

                    // Compute transformed normal:
                    vNormal = vPoint2 - vIsect;

                    // Set new clipping plane parameters:
                    point.set(vIsect[0], vIsect[1], vIsect[2]);
                    normal.set(vNormal[0], vNormal[1], vNormal[2]);
                    // is intersection point still in volume?
                    if (_virvoDrawable->getRenderer()->isInVolume(&point))
                    {
                        _virvoDrawable->getRenderer()->_renderState.setClippingPlane(&point, &normal);
                    }
                }
            }
        }
        if (!clipInteraction && !roiInteraction)
        {
            processMoveInput(dev);
        }
    }
    VolumePickBox::cursorUpdate(dev);
}

void VirvoPickBox::setThickness(float thick)
{
    float oldThick = getThickness();

    // Update volume:
    getVD()->dist[2] = thick / (getVD()->_scale * getVD()->vox[2]);
    vvTexRend *texrend = dynamic_cast<vvTexRend *>(getRenderer());
    if (texrend)
        texrend->updateBrickGeom();

    // Update box, markers, isosurface:
    updateBoundingBox();
    scaleMarkers(thick / oldThick);
    Matrix xf;
    xf.makeScale(1, 1, thick);
    _vtkThickness->setMatrix(xf);
    updateIsoGeometry();
}

float VirvoPickBox::getThickness()
{
    vvVector3 size = getVD()->getSize();
    return size.e[2];
}

void VirvoPickBox::setVisible(DataType dt, bool visible)
{
    switch (dt)
    {
    case VIRVO:
        _virvoDrawable->setVisible(visible);
        // TODO: fix      _virvoDrawable->setRenderer(visible ? Virvo::VV_TEXREND : Virvo::VV_MEMORY);
        break;
    case VTK:
        if (visible)
        {
            if (_vtkSwitch->getNumChildren() == 0)
            {
                recreateVTKIsosurfaces();
                updateIsoGeometry();
            }
            //        _vtkNode->setNodeMask(~1);
        }
        //      else _vtkNode->setNodeMask(0);
        break;
    default:
        break;
    }
}

void VirvoPickBox::setIsoValue(float iso)
{
    cerr << "setIsoValue" << endl;
    _isoValue = iso;
    recreateVTKIsosurfaces();
    updateIsoGeometry();
    cerr << "setIsoValue done" << endl;
}

float VirvoPickBox::getIsoValue()
{
    return _isoValue;
}

void VirvoPickBox::setIsoColor(float col)
{
    _isoColor = col;
    recreateVTKIsosurfaces();
    updateIsoGeometry();
}

float VirvoPickBox::getIsoColor()
{
    return _isoColor;
}

void VirvoPickBox::updateIsoGeometry()
{
    /*
    for(int i=0;i<_vtkVolumes.num();i++)
    {
      // Recompute isosurface for current volume size:
      vvVector3 vvSize = getVD()->getSize();
      Vec3 osgSize(vvSize[0], vvSize[1], vvSize[2]);
      _vtkVolumes[i]->setSize(osgSize);
    }
  */
}

int VirvoPickBox::nextTimeStep()
{
    int frame = getRenderer()->getCurrentFrame() + 1;
    if (frame >= getVD()->frames)
        frame = 0;
    setCurrentFrame(frame);
    return frame;
}

int VirvoPickBox::prevTimeStep()
{
    int frame = getRenderer()->getCurrentFrame() - 1;
    if (frame < 0)
        frame = getVD()->frames - 1;
    setCurrentFrame(frame);
    return frame;
}

void VirvoPickBox::recreateVTKIsosurfaces()
{
    /*
    cerr << "recreateVTKIsosurfaces" << endl;

    // remove any previous isosurfaces
    // Problem: removeChild has a bug, see:
    // http://openscenegraph.net/pipermail/osg-users/2003-October/035743.html
    // This is a workaround that should work:
    int i;
    int numChildren = _vtkSwitch->getNumChildren();
    for (i=0; i<numChildren; ++i)
      _vtkSwitch->removeChild(_vtkVolumes[i]->getNode());

  cerr << "trace1" << endl;

  while(_vtkVolumes.num())
  delete _vtkVolumes.pop();

  // create a VTK isosurface for each volume
  for(int i=0;i<getVD()->frames;i++)
  {
  cerr << "Making isosurface for frame #" << i << endl;
  vtkVolumeGeode *vg = new vtkVolumeGeode(getVD(), _isoValue, _isoColor, i);
  vvVector3 vvSize = getVD()->getSize();
  Vec3 osgSize(vvSize[0], vvSize[1], vvSize[2]);
  vg->setSize(osgSize);
  _vtkVolumes += vg;
  _vtkSwitch->addChild(vg->getNode());
  cerr << "done" << endl;
  }
  _vtkSwitch->setSingleChildOn(getRenderer()->getCurrentFrame());
  cerr << "recreateVTKIsosurfaces done" << endl;
  */
}

// XXX - this should probably just become 'setTimeStep(..)' above, but
// it came from the use in $G/src/vox-cave/CaveVOX.C
void VirvoPickBox::setCurrentFrame(int frame)
{
    // update volume renderering
    getRenderer()->setCurrentFrame(frame);

    // update isosurface display
    if (_vtkSwitch->getNumChildren() > frame)
        _vtkSwitch->setSingleChildOn(frame);
}

void VirvoPickBox::computeHistogram()
{
    int twidth, theight;
    unsigned char *tex;
    vvArray<float *> data;
    int i, size;

    getLineData(data);

    twidth = 256;
    theight = 256;

    tex = new unsigned char[twidth * theight * 4];

    getVD()->makeLineTexture(vvVolDesc::HISTOGRAM, getCheckedChannels(), twidth, theight, true, data, tex);

    data.deleteElementsArray();

    _histoImage->setImage(256, 256, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, tex, Image::USE_NEW_DELETE);
    _histoImage->dirty();
}

void VirvoPickBox::computeRectIntensityDiagram()
{
    unsigned char *tmp;
    int width, height;

    tmp = getRectData(width, height);

    _heightField->createHeightField(width, height, 1, tmp);
    _heightField->setVisible(true);
}

void VirvoPickBox::computeIntensityDiagram()
{
    int twidth, theight;
    unsigned char *tex;
    vvArray<float *> data;

    getLineData(data);

    // set height and width of texture
    twidth = 256;
    theight = 256;

    tex = new unsigned char[twidth * theight * 4];

    getVD()->makeLineTexture(vvVolDesc::INTENSDIAG, getCheckedChannels(), twidth, theight, true, data, tex);

    _intensImage->setImage(twidth, theight, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, tex, Image::USE_NEW_DELETE);
    _intensImage->dirty();
}

void VirvoPickBox::computeRectHistogram()
{
    int twidth, theight;
    unsigned char *tex;
    vvArray<float *> data;
    int i, size;

    getRectHistogramData(data);

    twidth = 256;
    theight = 256;

    tex = new unsigned char[twidth * theight * 4];

    getVD()->makeLineTexture(vvVolDesc::HISTOGRAM, getCheckedChannels(), twidth, theight, true, data, tex);

    std::cerr << "data count: " << data.count() << endl;
    //  data.deleteElementsArray();

    _histoImage->setImage(256, 256, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, tex, Image::USE_NEW_DELETE);
    _histoImage->dirty();
}

void VirvoPickBox::getLineData(vvArray<float *> &data)
{
    int x0, x1, y0, y1, z0, z1;
    Vec3 start, end;

    start = _measure->getLeftEnd() * _localToVoxel;
    end = _measure->getRightEnd() * _localToVoxel;

    x0 = (int)floor(start[0]);
    y0 = (int)floor(start[1]);
    z0 = (int)floor(start[2]);
    x1 = (int)floor(end[0]);
    y1 = (int)floor(end[1]);
    z1 = (int)floor(end[2]);

    // get data of all voxel on the line
    getVD()->getLineHistData(x0, y0, z0, x1, y1, z1, data);
}

void VirvoPickBox::getRectHistogramData(vvArray<float *> &data)
{
    float value;
    int width, height;
    int x, y, index;
    unsigned char *tmp;

    // get points of rectangle
    tmp = getRectData(width, height);

    for (y = 0; y < height; y++)
        for (x = 0; x < width; x++)
        {
            index = y * width + x;
            value = (float)(tmp[index]);
            data.append(&value);
        }

    delete[] tmp;
}

unsigned char *VirvoPickBox::getRectData(int &width, int &height)
{
    vvTexRend *texrend;
    float points[4][3];
    unsigned char *tmp;

    _rectangle->getCorners(points);

    std::cerr << "Point1: " << points[0][0] << " " << points[0][1] << " " << points[0][2] << endl;
    std::cerr << "Point2: " << points[1][0] << " " << points[1][1] << " " << points[1][2] << endl;
    std::cerr << "Point3: " << points[2][0] << " " << points[2][1] << " " << points[2][2] << endl;
    std::cerr << "Point4: " << points[3][0] << " " << points[3][1] << " " << points[3][2] << endl;

    texrend = dynamic_cast<vvTexRend *>(getRenderer());

    if (texrend)
        tmp = texrend->getHeightFieldData(points, width, height);
    else
        assert(0);

    std::cerr << "Width: " << width << " Height: " << height << endl;

    return tmp;
}

void VirvoPickBox::computeLocalToVoxel()
{
    float dx, dy, dz;
    Matrix rot, trans, scale;

    // voxel size
    dx = getBoxSize()[0] / getVD()->vox[0];
    dy = getBoxSize()[1] / getVD()->vox[1];
    dz = getBoxSize()[2] / getVD()->vox[2];

    rot.makeRotate(M_PI, Vec3(1, 0, 0));
    trans.makeTranslate(-_bbox._min);
    scale.makeScale(Vec3(1 / dx, 1 / dy, 1 / dz));

    _localToVoxel = rot * trans * scale;
}

/** Creates a volume with zero data values to be updated later with
updateDynamicVolume().
*/
void VirvoPickBox::makeDynamicVolume(const char *name, int w, int h, int s)
{
    _virvoDrawable->makeDynamicVolume(name, w, h, s);
    updateBoundingBox();
}

/** Update data array of dynamic volume. Size of array newData must be the same
as in makeDynamicVolume. The data at newData must be deleted by the _caller_.
  @param newData pointer to updated data array. If data format is double, it must be cast to uchar
*/
void VirvoPickBox::updateDynamicVolume(double *newData)
{
    _virvoDrawable->updateDynamicVolume(newData);
}
