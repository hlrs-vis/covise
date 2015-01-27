/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DrawObj.H"
#include "DrawMgr.H"
#include <GL/glut.h>

// Virvo:
#include <vvvecmath.h>
#include <vvrenderer.h>
#include <vvtokenizer.h>
#include <vvtexrend.h>

// OSG:
#include <osg/Geode>
#include <osg/LineSegment>

// CUI:
#include <CUI.H>

// Local:
#include "VolumePickBox.H"
#include "VirvoPickBox.H"
#include "vtkActorToOSG.H"
#include "vtkVolumeGeode.H"

// VTK:
#include <vtk/vtkStructuredPoints.h>
#include <vtk/vtkContourFilter.h>

using namespace cui;
using namespace osg;

class MyDebugObj : public DrawObj
{
protected:
    double _x, _y, _z;
    Vec3 _col;

public:
    MyDebugObj(Vec3 color, double x, double y, double z)
        : DrawObj("MyDebugObj")
        , _x(x)
        , _y(y)
        , _z(z)
        , _col(color)
    {
    }
    void draw()
    {
        glColor3f(_col[0], _col[1], _col[2]);
        glPushMatrix();
        glTranslated(_x, _y, _z);
        glutSolidCube(0.1); //,7,7);
        glPopMatrix();
    }
};
void
addDebugPoint(Vec3 col, double x, double y, double z)
{
    DRAWMGR::registerObj(new MyDebugObj(col, x, y, z));
}

VolumePickBox::VolumePickBox(Interaction *interaction, osgDrawObj *osgObj,
                             const Vec3 &min, const Vec3 &max, const Vec4 &c1, const Vec4 &c2,
                             const Vec4 &c3)
    : PickBox(interaction, osgObj, min, max, c1, c2, c3)
    , CardListener()
    , RadioGroupListener()
{
    _interaction = interaction;
    _gazeSupport = false;
    _markupMode = false;
    _paintMode = false;
    _prevType = InputDevice::LASER;
    _settingPointerLength = false;
    _isNavigating = false;
    _moveThresholdReached = true;
    _logFile = NULL;
    _markerType = Marker::CONE;
    _paintType = Paintbrush::SPHERE;
    _isPainting = false;
    _lineSize = 1;

    _measure = new Measure(interaction, this);
    _measure->setVisible(false);
    _measure->addMeasureListener(this);
    _scale->addChild(_measure->getNode());

    _rectangle = new Rectangle(interaction);
    _rectangle->setVisible(false);
    _scale->addChild(_rectangle->getNode());

    _paintLine = new Bar(interaction, NULL);
    _paintLine->setVisible(false);
    _scale->addChild(_paintLine->getNode());

    createDiagramPanel();

    // Set the initial seed:
    srand((unsigned)time(NULL));
}

VolumePickBox::~VolumePickBox()
{
    _histoImage->unref();
    _intensImage->unref();
    _osgObj->removeWorldChild(_diagramPanel->getNode());
}

/** @return true if matrices are different enough that one can assume
  the wand was moved on purpose.
*/
bool VolumePickBox::moveThresholdTest(Matrix &m1, Matrix &m2)
{
    // Compare translational part:
    Vec3 m1trans = m1.getTrans();
    Vec3 m2trans = m2.getTrans();
    Vec3 diff = m1trans - m2trans;
    float len = diff.length();
    if (len > 0.03f) // this is an empirical value!
    {
        return true;
    }
    else
    {
        // Compare rotational part:
        return false;
    }
}

void VolumePickBox::processMoveInput(InputDevice *dev)
{
    Matrix i2w = dev->getI2W();
    if (_isMovable)
    {
        move(_lastWand2w, i2w);

        // Notify event listeners:
        std::list<PickBoxListener *>::iterator iter;
        for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
        {
            (*iter)->pickBoxMoveEvent(this, dev);
        }
    }
}

void VolumePickBox::cursorUpdate(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;

    if (_isNavigating)
    {
        if (!_moveThresholdReached)
        {
            Matrix i2w = dev->getI2W();
            Matrix buttonPressedI2W = dev->getPressedI2W(0);
            _moveThresholdReached = moveThresholdTest(buttonPressedI2W, i2w);
        }
    }
    else if (_isPainting)
    {
        if (_pt == VolumePickBox::BOX || _pt == VolumePickBox::SPHERE)
        {
            paint(0);
        }
        else
        {
            Vec3 vPos = getPointInVolume();
            _paintLine->setVertices(_lastPoint, vPos);
        }
    }

    PickBox::cursorUpdate(dev);
}

void VolumePickBox::cursorEnter(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;

    // Set cursor according to marker mode:
    if (_markupMode && dev->getCursorType() != dev->NONE)
    {
        _prevType = dev->getCursorType();
        if (_markerType == Marker::CONE)
            dev->setCursorType(dev->CONE_MARKER);
        else if (_markerType == Marker::BOX)
        {
            dev->setCursorType(dev->BOX_MARKER);
        }
        else
            dev->setCursorType(dev->SPHERE_MARKER);
    }
    else if (_paintMode && dev->getCursorType() != dev->NONE)
    {
        _prevType = dev->getCursorType();
        if (_pt == VolumePickBox::BOX)
            dev->setCursorType(dev->BOX_BRUSH);
        else
            dev->setCursorType(dev->SPHERE_BRUSH);
    }

    PickBox::cursorEnter(dev);
}

void VolumePickBox::cursorLeave(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;

    // Set cursor according to marker mode:
    if (_markupMode && dev->getCursorType() != dev->NONE)
    {
        dev->setCursorType(_prevType);
    }
    else if (_paintMode && dev->getCursorType() != dev->NONE)
    {
        dev->setCursorType(_prevType);
    }

    PickBox::cursorLeave(dev);
}

/** Rotate box with trackball. Box rotates around its center, in the direction the trackball
  is pushed.
  Coordinate systems:
  w = world
  o = volume object
  m = mouse
  @param x,y trackball forces [-1..1]
  @param m2w mouse (=trackball) to world matrix
*/
void VolumePickBox::trackballRotation(float x, float y, Matrix &m2w)
{
    const float SENSITIVITY = 4.0f; // this is an empirical value! higher values make the object spin faster

    if (_isMovable)
    {
        // rotation in x direction, around z axis of wand
        float angle1 = x * SENSITIVITY * M_PI / 180.0f;
        // rotation in y direction, around x axis of wand
        float angle2 = y * SENSITIVITY * M_PI / 180.0f;
        Vec3 wAxisX, wAxisY; // trackball's x and y axes in world coordinates
        Matrix m2w_rot = m2w;
        m2w_rot.setTrans(0, 0, 0);
        wAxisX = Vec3(0, 0, 1) * m2w_rot;
        if (CUI::_display == CUI::CAVE)
        {
            wAxisY = Vec3(1, 0, 0) * m2w_rot;
        }
        else
        {
            wAxisY = Vec3(0, 1, 0) * m2w_rot;
        }
        Matrix wRotate;
        wAxisX.normalize();
        wAxisY.normalize();
        wRotate.makeRotate(angle1, wAxisX, angle2, wAxisY, 0, Vec3(1, 0, 0));

        // This rotates around the object center, matching the orientation of the mouse:
        Matrix o2w = _node->getMatrix(); // volume object to world
        Matrix o2w_rot = o2w;
        o2w_rot.setTrans(0, 0, 0);
        Vec3 trans;
        Matrix o2w_trans;
        trans = o2w.getTrans();
        o2w_trans.makeTranslate(trans);
        _node->setMatrix(o2w_rot * wRotate * o2w_trans);
    }
}

void VolumePickBox::setMarkupMode(bool markupMode)
{
    _markupMode = markupMode;
    vector<Marker *>::const_iterator iter;
    for (iter = _markers.begin(); iter != _markers.end(); ++iter)
    {
        (*iter)->setVisible(_markupMode);
    }
}

void VolumePickBox::setPaintMode(bool paintMode)
{
    _paintMode = paintMode;
}

bool VolumePickBox::getPaintMode()
{
    return _paintMode;
}

void VolumePickBox::setPaintType(PaintType pt)
{
    _pt = pt;
}

VolumePickBox::PaintType VolumePickBox::getPaintType()
{
    return _pt;
}

bool VolumePickBox::getMarkupMode()
{
    return _markupMode;
}

void VolumePickBox::placeMarker(Marker *m)
{
    // Place marker at end of pointer ray:
    Matrix i2w = _interaction->_wandR->getI2W();
    Matrix w2v = Matrix::inverse(getB2W());
    Matrix rot, trans;
    if (CUI::_display == CUI::FISHTANK)
    {
        rot.makeRotate(M_PI / 2.0f, 0.0f, 1.0f, 0.0f);
    }
    else
    {
        rot.makeRotate(M_PI / 2.0f, 1.0f, 0.0f, 0.0f);
    }
    trans.makeTranslate(0.0f, 0.0f, _interaction->_wandR->getPointerLength());
    Matrix marker2v = rot * trans * i2w * w2v;
    m->setMatrix(marker2v);
}

Vec3 VolumePickBox::getPointInVolume()
{
    // Get the position of the wand
    Vec3 wPointerStart = _interaction->_wandR->getCursorPos();

    //  addDebugPoint(Vec3(1,1,0), wPointerStart[0], wPointerStart[1], wPointerStart[2]);

    // Translate to the end of the pointer
    Vec3 dir = _interaction->_wandR->getCursorDir() * _interaction->_wandR->getPointerLength();
    Matrix trans;
    trans.makeTranslate(dir[0], dir[1], dir[2]);
    Vec3 wPointerEnd = wPointerStart * trans;

    //  addDebugPoint(Vec3(1,0,1), wPointerEnd[0], wPointerEnd[1], wPointerEnd[2]);

    Matrix w2v = Matrix::inverse(getB2W());
    Vec3 vPos = wPointerEnd * w2v;

    return vPos;
}

Vec3 VolumePickBox::getVolume2Voxel(Vec3 pt)
{
    VirvoPickBox *virvoPickBox;
    if ((virvoPickBox = dynamic_cast<VirvoPickBox *>(this)) != NULL)
    {
        vvVolDesc *vd = virvoPickBox->getVD();

        // Rotate 180 around X-axis
        Matrix rot;
        rot.makeRotate(M_PI, 1, 0, 0);

        Vec3 vPos = pt * rot;

        Vec3 p;
        p[0] = vPos[0] + getBoxSize()[0] / 2;
        p[1] = vPos[1] + getBoxSize()[1] / 2;
        p[2] = vPos[2] + getBoxSize()[2] / 2;

        float dx = getBoxSize()[0] / vd->vox[0];
        float dy = getBoxSize()[1] / vd->vox[1];
        float dz = getBoxSize()[2] / vd->vox[2];

        Vec3 pos(p[0] / dx, p[1] / dy, p[2] / dz);

        return pos;
    }
}

void VolumePickBox::paint(int channel)
{
    VirvoPickBox *virvoPickBox;
    if ((virvoPickBox = dynamic_cast<VirvoPickBox *>(this)) != NULL)
    {
        vvVolDesc *vd = virvoPickBox->getVD();

        uchar uc[vd->bpc];

        for (int i = 0; i < vd->bpc; i++)
            uc[i] = 255; // (uchar) (_color[i] * 255);

        Vec3 pos = getVolume2Voxel(getPointInVolume());

        if (_pt == VolumePickBox::BOX)
        {
            float dx = getBoxSize()[0] / vd->vox[0];
            float dy = getBoxSize()[1] / vd->vox[1];
            float dz = getBoxSize()[2] / vd->vox[2];

            //Actual painting
            float brushSize = _interaction->_wandR->getBoxBrush()->getSize() / 10;
            float xSize = (brushSize / dx) / 2;
            float ySize = (brushSize / dy) / 2;
            float zSize = (brushSize / dz) / 2;

            vd->drawBox((int)(pos.x() - xSize), (int)(pos.y() - ySize), (int)(pos.z() - zSize),
                        (int)(pos.x() + xSize), (int)(pos.y() + ySize), (int)(pos.z() + zSize), 3, uc);

            vvTexRend *tex = dynamic_cast<vvTexRend *>(virvoPickBox->getRenderer());
            if (tex != NULL)
                tex->updateVolumeData((int)(pos.x() - xSize), (int)(pos.y() - ySize),
                                      (int)(pos.z() - zSize), (int)(pos.x() + xSize), (int)(pos.y() + ySize), (int)(pos.z() + zSize));
        }
        else if (_pt == VolumePickBox::SPHERE)
        {
            float dr = getBoxSize()[0] / vd->vox[0];
            float radius = _interaction->_wandR->getSphereBrush()->getSize() / 2 / dr / 5;

            vd->drawSphere((int)(pos[0]), (int)(pos[1]), (int)(pos[2]), int(radius), 3, uc);

            vvTexRend *tex = dynamic_cast<vvTexRend *>(virvoPickBox->getRenderer());
            if (tex != NULL)
                tex->updateVolumeData((int)(pos[0] - radius), (int)(pos[1] - radius), (int)(pos[2] - radius),
                                      (int)(pos[0] + radius), (int)(pos[1] + radius), (int)(pos[2] + radius));
        }
        else if (_pt == VolumePickBox::LINE)
        {
            _lastPoint = getPointInVolume();
            _paintLine->setVisible(true);
        }
        //    virvoPickBox->getRenderer()->updateVolumeData();
        //    virvoPickBox->getRenderer()->updateVolumeData(
    }
}

void VolumePickBox::paintLine()
{
    VirvoPickBox *virvoPickBox;
    if ((virvoPickBox = dynamic_cast<VirvoPickBox *>(this)) != NULL)
    {
        uchar uc[3] = { 255, 255, 255 };
        vvVolDesc *vd = virvoPickBox->getVD();

        Vec3 pos = getVolume2Voxel(getPointInVolume());
        Vec3 pos2 = getVolume2Voxel(_lastPoint);

        if (_lineSize == 1)
            vd->drawLine((int)pos[0], (int)pos[1], (int)pos[2], (int)pos2[0], (int)pos2[1], (int)pos2[2], uc);
        else
        {
            for (int i = -_lineSize; i < _lineSize; i++)
            {
                vd->drawLine((int)pos[0] + i, (int)pos[1], (int)pos[2], (int)pos2[0] + i, (int)pos2[1], (int)pos2[2], uc);
                //        vd->drawLine((int)pos[0], (int)pos[1]+i, (int)pos[2], (int)pos2[0], (int)pos2[1]+i, (int)pos2[2], uc);
                //        vd->drawLine((int)pos[0], (int)pos[1], (int)pos[2]+i, (int)pos2[0], (int)pos2[1], (int)pos2[2]+i, uc);
            }
        }
    }
}

void VolumePickBox::clear()
{
    int i;

    VirvoPickBox *virvoPickBox;
    if ((virvoPickBox = dynamic_cast<VirvoPickBox *>(this)) != NULL)
    {
        vvVolDesc *vd = virvoPickBox->getVD();

        uchar uc[vd->bpc];
        for (i = 0; i < vd->bpc; ++i)
            uc[i] = 0;
        for (i = 0; i < vd->chan; ++i)
        {
            vd->drawBox(0, 0, 0, vd->vox[0] - 1, vd->vox[1] - 1, vd->vox[2] - 1, i, uc);
        }
        virvoPickBox->getRenderer()->updateVolumeData();
    }
}

void VolumePickBox::addMarkerByHand(Marker *m)
{
    _markers.push_back(m);
    _scale->addChild(m->getNode());
    m->setVisible(_markupMode);
    m->addMarkerListener(this);

    // add to general Log File
    if (_logFile)
        _logFile->addMarkerLog(_markers.size(), m);
    cerr << "Number of markers: " << _markers.size() << endl;
}

void VolumePickBox::addMarkerFromFile(Marker *m)
{
    _markers.push_back(m);
    _scale->addChild(m->getNode());
    m->setVisible(_markupMode);
    m->addMarkerListener(this);
}

bool VolumePickBox::writeMarkerFile(const char *filename)
{
    float fMat[16];
    int j;

    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        cerr << "Failed creating marker file " << filename << endl;
        return false;
    }
    else
        cerr << "Marker file created: " << filename << endl;

    Vec3 pos;
    Matrix marker2v;

    Vec4 col;
    float size;

    // shuffle marker vector
    random_shuffle(_markers.begin(), _markers.end());

    std::vector<Marker *>::iterator iter;
    for (iter = _markers.begin(); iter != _markers.end(); ++iter)
    {
        col = (*iter)->getColor();
        size = (*iter)->getSize();
        marker2v = (*iter)->getMatrix();
        double *mat = marker2v.ptr();
        for (j = 0; j < 16; ++j)
        {
            fMat[j] = float(mat[j]);
        }
        fprintf(fp, "MARKER\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                size,
                col[0], col[1], col[2], col[3],
                fMat[0], fMat[1], fMat[2], fMat[3],
                fMat[4], fMat[5], fMat[6], fMat[7],
                fMat[8], fMat[9], fMat[10], fMat[11],
                fMat[12], fMat[13], fMat[14], fMat[15]);
    }

    fclose(fp);
    return true;
}

/// @return true if successful
bool VolumePickBox::readMarkerFile(const char *filename)
{
    removeAllMarkers();

    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        cerr << "Failed opening marker file " << filename << endl;
        return false;
    }
    else
        cerr << "Marker file opened: " << filename << endl;

    Marker *newMarker;
    int i = 0, err, j;
    double mat[16];
    float fMat[16];
    Matrix marker2v;
    Vec4 col;
    float size;
    do
    {
        err = fscanf(fp, "MARKER\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                     &size,
                     &col[0], &col[1], &col[2], &col[3],
                     &fMat[0], &fMat[1], &fMat[2], &fMat[3],
                     &fMat[4], &fMat[5], &fMat[6], &fMat[7],
                     &fMat[8], &fMat[9], &fMat[10], &fMat[11],
                     &fMat[12], &fMat[13], &fMat[14], &fMat[15]);
        if (err != EOF)
        {
            cerr << "Reading Marker " << ++i << endl;
            for (j = 0; j < 16; ++j)
            {
                mat[j] = fMat[j];
            }
            newMarker = new Marker(_markerType);
            marker2v.set(mat);
            newMarker->setMatrix(marker2v);
            newMarker->setSize(size);
            newMarker->setColor(col);
            addMarkerFromFile(newMarker);
        }
    } while (err != EOF);

    fclose(fp);
    return true;
}

void VolumePickBox::removeMarker(Marker *m)
{
    _scale->removeChild(m->getNode());
    _markers.erase(find(_markers.begin(), _markers.end(), m));
    cerr << "Number of markers left: " << _markers.size() << endl;
}

/**
 * removes all the markers inside the Volume
 * @return the amount of removed markers
 */
int VolumePickBox::removeAllMarkers()
{
    Marker *m;
    int i = 0;
    while (_markers.size() > 0)
    {
        m = _markers.front();
        removeMarker(m);
        i++;
    }

    return i;
}

/**
 * removes the last marker of the vector inside the Volume until the specified number of markers in Volume is reached
 * @return amount of removed markers
 */
int VolumePickBox::removeRandMarkersUntil(int num)
{
    int i = 0;
    while (_markers.size() > num)
    {
        /*               // number between 0.0 and 1.0
              randNum = (float)rand()/(float)RAND_MAX;
        // number between 0 and _markers.size()-1
        scaledRand = (int)(randNum * (_markers.size()-1));  */

        removeMarker(_markers.back()); // or: _markers.erase(_markers.end()-1);//
        i++;
    }

    return i;
}

void VolumePickBox::scaleMarkers(float scaleZ)
{
    Matrix mat;
    Vec3 pos;

    vector<Marker *>::const_iterator iter;
    for (iter = _markers.begin(); iter != _markers.end(); ++iter)
    {
        pos = (*iter)->getPosition();
        pos[2] *= scaleZ;
        (*iter)->setPosition(pos);
    }
}

void VolumePickBox::setColor(Vec4 col)
{
    _color = col;
}

void VolumePickBox::buttonEvent(InputDevice *dev, int button)
{
    if (dev == dev->_interaction->_wandR)
    {
        if (button == 0)
        {
            if (dev->getButtonState(button) == 1) // button pressed?
            {
                _isNavigating = true;
                _moveThresholdReached = true;
            }
            else if (dev->getButtonState(button) == 0) // button released?
            {
                _isNavigating = false;
                _settingPointerLength = true;
            }
        }
        if (button == 1)
        {
            if (dev->getButtonState(button) == 1)
            {
                // Add a marker:
                if (_markupMode)
                {
                    float size;
                    Vec4 color;
                    if (_markerType == Marker::CONE)
                    {
                        size = _interaction->_wandR->getConeMarker()->getSize();
                        color = _interaction->_wandR->getConeMarker()->getColor();
                    }
                    else if (_markerType == Marker::BOX)
                    {
                        size = _interaction->_wandR->getBoxMarker()->getSize();
                        color = _interaction->_wandR->getBoxMarker()->getColor();
                    }
                    else
                    {
                        size = _interaction->_wandR->getSphereMarker()->getSize();
                        color = _interaction->_wandR->getSphereMarker()->getColor();
                    }
                    Marker *m = new Marker(_markerType, _interaction, size, color);
                    placeMarker(m);
                    addMarkerByHand(m);
                }
                else if (_paintMode)
                {
                    paint(0);
                    _isPainting = true;
                }
            }
            else if (dev->getButtonState(button) == 0)
            {
                if (_pt == VolumePickBox::LINE)
                    paintLine();
                _paintLine->setVisible(false);
                _isPainting = false;
            }
        }
    }
    PickBox::buttonEvent(dev, button); // call parent
}

void VolumePickBox::setAllMarkerSizes(float size)
{
    vector<Marker *>::const_iterator iter;
    for (iter = _markers.begin(); iter != _markers.end(); ++iter)
    {
        cerr << "resetting marker size to " << size << endl;
        (*iter)->setSize(size);
    }
}

void VolumePickBox::markerEvent(Marker *m, int button, int state)
{
    if (button == 2 && state == 1)
        removeMarker(m);
}

void VolumePickBox::setGazeSupport(bool gaze)
{
    _gazeSupport = gaze;
}

void VolumePickBox::setLogFile(LogFile *logFile)
{
    _logFile = logFile;
}

void VolumePickBox::getMarkersFromLog(const char *logfile)
{
    removeAllMarkers();

    vvTokenizer *tokenizer; // ASCII file tokenizer
    vvTokenizer::TokenType ttype; // currently processed token type
    FILE *fp; // volume file pointer
    bool done;

    if ((fp = fopen(logfile, "rb")) == NULL)
    {
        cout << "Error: Cannot open logfile." << endl;
        return;
    }

    // Read file data:
    tokenizer = new vvTokenizer(fp);
    //  tokenizer->setCommentCharacter('#');
    tokenizer->setEOLisSignificant(true);
    tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
    tokenizer->setParseNumbers(true);
    //  tokenizer->setWhitespaceCharacter(' ');
    done = false;
    while (!done)
    {
        // Read a token:
        ttype = tokenizer->nextToken();
        if (ttype == vvTokenizer::VV_EOF)
        {
            done = true;
            continue;
        }

        ttype = tokenizer->nextToken();
        switch (ttype)
        {
        case vvTokenizer::VV_WORD:
        {
            if (strcmp(tokenizer->sval, "MARKER") == 0)
            {
                Marker *m = new Marker(_markerType);
                Vec3 pos;
                Matrix mat;

                ttype = tokenizer->nextToken();
                int num = (int)tokenizer->nval;

                for (int i = 0; i < 3; ++i)
                {
                    ttype = tokenizer->nextToken();
                    pos[i] = tokenizer->nval;
                }
                mat.setTrans(pos);
                m->setMatrix(mat);
                m->setSize(m->getSize() / 5.0f);
                addMarkerFromFile(m);
                tokenizer->nextLine();
            }
            else
                tokenizer->nextLine();
            break;
        }
            /*      case vvTokenizer::VV_NUMBER:
            case vvTokenizer::VV_EOL:
              cout << "error!" << endl;
              break;
            default: done = true;
              break;*/
        }
    }

    // Clean up:
    delete tokenizer;
    fclose(fp);
}

void VolumePickBox::setMarkerType(Marker::GeometryType markerType)
{
    _markerType = markerType;
}

Marker::GeometryType VolumePickBox::getMarkerType()
{
    return _markerType;
}

void VolumePickBox::createDiagramPanel()
{
    _diagramPanel = new Panel(_interaction, _osgObj, Panel::STATIC, Panel::FREE_MOVABLE);

    _intensityButton = new RadioButton(_interaction);
    _intensityButton->setText("Intens");
    _diagramPanel->addCard(_intensityButton, 3, 0);
    _radioGroup1.add(_intensityButton);

    _histogramButton = new RadioButton(_interaction);
    _histogramButton->setText("Histo");
    _diagramPanel->addCard(_histogramButton, 3, 1);
    _radioGroup1.add(_histogramButton);

    _lineButton = new RadioButton(_interaction);
    _lineButton->setText("Line");
    _diagramPanel->addCard(_lineButton, 4, 0);
    _radioGroup2.add(_lineButton);

    _rectangleButton = new RadioButton(_interaction);
    _rectangleButton->setText("Rectangle");
    _diagramPanel->addCard(_rectangleButton, 4, 1);
    _radioGroup2.add(_rectangleButton);

    _redChannel = new CheckBox(_interaction);
    _redChannel->addCardListener(this);
    _redChannel->setText("Red");
    _redChannel->setChecked(true);
    _diagramPanel->addCard(_redChannel, 0, 3);

    _greenChannel = new CheckBox(_interaction);
    _greenChannel->addCardListener(this);
    _greenChannel->setText("Green");
    _greenChannel->setChecked(true);
    _diagramPanel->addCard(_greenChannel, 1, 3);

    _blueChannel = new CheckBox(_interaction);
    _blueChannel->addCardListener(this);
    _blueChannel->setText("Blue");
    _blueChannel->setChecked(true);
    _diagramPanel->addCard(_blueChannel, 2, 3);

    _alphaChannel = new CheckBox(_interaction);
    _alphaChannel->addCardListener(this);
    _alphaChannel->setText("Alpha");
    _alphaChannel->setChecked(true);
    _diagramPanel->addCard(_alphaChannel, 3, 3);

    _selChannel = 0x0f;

    _diagTexture = new TextureWidget(_interaction, 3.0, 3.0);

    Vec4 color;

    Cone *blueCone = new osg::Cone(Vec3(0, 0, 0), TextureWidget::DEFAULT_LABEL_HEIGHT / 2.0, TextureWidget::DEFAULT_LABEL_HEIGHT / 2.0);
    ShapeDrawable *drawableBlue = new osg::ShapeDrawable(blueCone);
    color.set(0.0, 0.0, 1.0, 1.0);
    drawableBlue->setColor(color);
    drawableBlue->setUseDisplayList(false);

    Cone *redCone = new osg::Cone(Vec3(0, 0, 0), TextureWidget::DEFAULT_LABEL_HEIGHT / 2.0, TextureWidget::DEFAULT_LABEL_HEIGHT / 2.0);
    ShapeDrawable *drawableRed = new osg::ShapeDrawable(redCone);
    color.set(1.0, 0.0, 0.0, 1.0);
    drawableRed->setColor(color);
    drawableRed->setUseDisplayList(false);

    _diagTexture->addGeomToLeft(drawableBlue);
    _diagTexture->addGeomToRight(drawableRed);

    _intensImage = new Image();
    _histoImage = new Image();

    _diagTexture->setImage(0, _intensImage);
    _diagTexture->setLabelText(0, "Intensity");
    _diagTexture->setImage(1, _histoImage);
    _diagTexture->setLabelText(1, "Histogram");
    _diagramPanel->addTexture(_diagTexture, 0, 0);

    _diagramPanel->setVisible(false);

    Matrix tmp, scale;
    tmp = _diagramPanel->getNode()->getMatrix();
    tmp.makeRotate(Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, -2.0));
    tmp.setTrans(Vec3(-0.65 * _diagramPanel->getWidth() / 2.0, 3, 4));
    scale.makeScale(0.65, 0.65, 0.65);
    _diagramPanel->getNode()->setMatrix(scale * tmp);

    _osgObj->addWorldChild(_diagramPanel->getNode());

    _heightField = new HeightFieldPickBox(_interaction, _osgObj, Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN);
    _heightField->setVisible(false);
    _osgObj->addWorldChild(_heightField->getNode());

    _radioGroup1.addRadioGroupListener(this);
    _radioGroup1.setSelected(_intensityButton);

    _radioGroup2.addRadioGroupListener(this);
    _radioGroup2.setSelected(_lineButton);
}

void VolumePickBox::setChannelBoxesVisible()
{
    if (isVirvo())
    {
        switch (dynamic_cast<VirvoPickBox *>(this)->getVD()->chan)
        {
        case 0:
            _redChannel->setVisible(false);
            _greenChannel->setVisible(false);
            _blueChannel->setVisible(false);
            _alphaChannel->setVisible(false);
            break;
        case 1:
            _redChannel->setVisible(true);
            _greenChannel->setVisible(false);
            _blueChannel->setVisible(false);
            _alphaChannel->setVisible(false);
            break;
        case 2:
            _redChannel->setVisible(true);
            _greenChannel->setVisible(true);
            _blueChannel->setVisible(false);
            _alphaChannel->setVisible(false);
            break;
        case 3:
            _redChannel->setVisible(true);
            _greenChannel->setVisible(true);
            _blueChannel->setVisible(true);
            _alphaChannel->setVisible(false);
            break;
        case 4:
            _redChannel->setVisible(true);
            _greenChannel->setVisible(true);
            _blueChannel->setVisible(true);
            _alphaChannel->setVisible(true);
            break;
        default:
            break;
        }
    }
    else
    {
        _redChannel->setVisible(false);
        _greenChannel->setVisible(false);
        _blueChannel->setVisible(false);
        _alphaChannel->setVisible(false);
    }
}

void VolumePickBox::setDiagramPanelVisible(bool flag)
{
    if (_lineButton->isChecked())
        setMeasureVisible(true);
    else if (_rectangleButton->isChecked())
        setRectangleVisible(true);

    _diagramPanel->setVisible(flag);
    setChannelBoxesVisible();
    showDiagram();
}

void VolumePickBox::setMeasureVisible(bool flag)
{
    _measure->setVisible(flag);
}

bool VolumePickBox::getMeasureVisible()
{
    return _measure->isVisible();
}

void VolumePickBox::setRectangleVisible(bool flag)
{
    _rectangle->setVisible(flag);
}

bool VolumePickBox::getRectangleVisible()
{
    return _rectangle->isVisible();
}

void VolumePickBox::setCheckedChannels(vvVolDesc::Channel c, bool flag)
{
    switch (c)
    {
    case vvVolDesc::CHANNEL_R:
        if (flag)
            _selChannel = _selChannel | vvVolDesc::CHANNEL_R;
        else
            _selChannel = _selChannel & (~vvVolDesc::CHANNEL_R);
        break;
    case vvVolDesc::CHANNEL_G:
        if (flag)
            _selChannel = _selChannel | vvVolDesc::CHANNEL_G;
        else
            _selChannel = _selChannel & (~vvVolDesc::CHANNEL_G);
        break;
    case vvVolDesc::CHANNEL_B:
        if (flag)
            _selChannel = _selChannel | vvVolDesc::CHANNEL_B;
        else
            _selChannel = _selChannel & (~vvVolDesc::CHANNEL_B);
        break;
    case vvVolDesc::CHANNEL_A:
        if (flag)
            _selChannel = _selChannel | vvVolDesc::CHANNEL_A;
        else
            _selChannel = _selChannel & (~vvVolDesc::CHANNEL_A);
        break;
    default:
        break;
    }
}

unsigned char VolumePickBox::getCheckedChannels()
{
    return _selChannel;
}

bool VolumePickBox::cardButtonEvent(Card *card, int button, int newState)
{
    if (button == 0 && newState == 0)
    {
        if (card == _redChannel)
        {
            setCheckedChannels(vvVolDesc::CHANNEL_R, _redChannel->isChecked());
            showDiagram();
        }
        else if (card == _greenChannel)
        {
            setCheckedChannels(vvVolDesc::CHANNEL_G, _greenChannel->isChecked());
            showDiagram();
        }
        else if (card == _blueChannel)
        {
            setCheckedChannels(vvVolDesc::CHANNEL_B, _blueChannel->isChecked());
            showDiagram();
        }
        else if (card == _alphaChannel)
        {
            setCheckedChannels(vvVolDesc::CHANNEL_A, _alphaChannel->isChecked());
            showDiagram();
        }
    }
    return true;
}

bool VolumePickBox::cardCursorUpdate(Card *, InputDevice *)
{
    return false;
}

bool VolumePickBox::radioGroupStatusChanged(RadioGroup *group)
{
    if (group == &_radioGroup2)
    {
        setMeasureVisible(_lineButton->isChecked());
        setRectangleVisible(_rectangleButton->isChecked());
    }

    showDiagram();
    return true;
}

void VolumePickBox::showDiagram()
{
    if (isVirvo())
    {
        if (_lineButton->isChecked())
        {
            if (_histogramButton->isChecked())
                dynamic_cast<VirvoPickBox *>(this)->computeHistogram();
            else if (_intensityButton->isChecked())
                dynamic_cast<VirvoPickBox *>(this)->computeIntensityDiagram();
        }
        else if (_rectangleButton->isChecked())
        {
            if (_histogramButton->isChecked())
                dynamic_cast<VirvoPickBox *>(this)->computeRectHistogram();
            else if (_intensityButton->isChecked())
                dynamic_cast<VirvoPickBox *>(this)->computeRectIntensityDiagram();
        }
    }

    if (_histogramButton->isChecked())
        _diagTexture->showTexture(1);
    else if (_intensityButton->isChecked())
        _diagTexture->showTexture(0);
}

void VolumePickBox::measureUpdate()
{
    showDiagram();
}

void VolumePickBox::rectangleUpdate()
{
    showDiagram();
}

void VolumePickBox::setLineSize(int size)
{
    _lineSize = size;
}
