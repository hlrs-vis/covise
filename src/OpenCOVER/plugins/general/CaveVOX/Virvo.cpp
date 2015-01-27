/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Virvo:
#ifdef USE_STINGRAY
#include <vvstingray.h>
#endif
#include <vvrenderer.h>
#include <vvvoldesc.h>
#include <vvfileio.h>
#include <vvtexrend.h>
#include <vvdebugmsg.h>

// Local:
#include "Virvo.H"

// Inspace:
#include <WorldTranslate.H>

using namespace osg;

Virvo::Virvo(AlgorithmType algorithm)
{
    setSupportsDisplayList(false); // turn off display lists right now, just incase we want to modify the projection matrix along the way.
    _renderer = NULL;
    _vd = NULL;
    _algorithm = algorithm;
    _visible = true;

    initStateset();
}

Virvo::Virvo(const Virvo &virvo, const CopyOp &copyop)
    : Drawable(virvo, copyop)
{
    _renderer = virvo._renderer;
    _vd = virvo._vd;
    _algorithm = virvo._algorithm;
}

Virvo::~Virvo()
{
    delete _renderer;
    delete _vd;
}

void Virvo::initStateset()
{
    StateSet *stateset = new StateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setMode(GL_BLEND, StateAttribute::ON);
    stateset->setRenderBinDetails(100, "RenderBin"); // draw after everything else
    setStateSet(stateset);
}

bool Virvo::loadVolumeFile(const char *filename)
{
    vvFileIO *fio;
    vvRenderState renderState;

    if (_renderer)
        renderState = _renderer->_renderState;
    else
    {
        renderState._texMemorySize = CONFIGval("TEX_MEMORY_SIZE", 128, 0);
        std::cerr << "texMemorySize with CONFIGval: " << renderState._texMemorySize << endl;
    }

    delete _renderer;
    _renderer = NULL;
    delete _vd;
    _vd = NULL;

    _vd = new vvVolDesc(filename);
    fio = new vvFileIO();
    if (fio->loadVolumeData(_vd) != vvFileIO::OK)
    {
        cerr << "Error loading volume file: " << _vd->getFilename() << endl;
        delete fio;
        delete _vd;
        _vd = NULL;
        return false;
    }
    else
    {
        _vd->printInfoLine(_vd->getFilename());
        delete fio;
    }

    // Set default color scheme if no TF present:
    if (_vd->tf.isEmpty())
    {
        _vd->tf.setDefaultAlpha(0);
        _vd->tf.setDefaultColors((_vd->chan == 1) ? 0 : 3);
    }
    if (_vd->bpc == 4 && _vd->real[0] == 0.0f && _vd->real[1] == 1.0f)
        _vd->setDefaultRealMinMax();
    _vd->resizeEdgeMax(1.0f);

    setRenderer(_algorithm, renderState);
    return true;
}

void Virvo::setRenderer(AlgorithmType alg, vvRenderState state)
{
    vvVector3 pos;
    vvRenderState renderState;

    //if (_renderer) renderState = _renderer->_renderState;

    delete _renderer;
    _renderer = NULL;

    _algorithm = alg;

    switch (_algorithm)
    {
    case VV_MEMORY:
        _renderer = new vvRenderer(_vd, state);
        break;
        break;
    case VV_TEXREND:
        _renderer = new vvTexRend(_vd, state, vvTexRend::VV_AUTO, vvTexRend::VV_BEST);
        break;
    case VV_BRICKS:
        _renderer = new vvTexRend(_vd, state, vvTexRend::VV_BRICKS, vvTexRend::VV_BEST);
        break;
#ifdef USE_STINGRAY
    case VV_STINGRAY:
        _renderer = new vvStingray(_vd, state);
        break;
#endif
    default:
        assert(0);
        break;
    }

    pos.zero();
    _renderer->setPosition(&pos);
    _renderer->setBoundariesColor(1.0f, 1.0f, 1.0f);
    _renderer->setCurrentFrame(0);

    updateBoundingBox();
}

vvVolDesc *Virvo::getVD()
{
    return _vd;
}

vvRenderer *Virvo::getRenderer()
{
    return _renderer;
}

BoundingBox Virvo::computeBound() const
{
    vvVector3 size2;

    _boundingBox.init();
    if (_renderer)
    {
        size2 = _vd->getSize();
        size2.scale(0.5f);
        _boundingBox.set(-size2[0], -size2[1], -size2[2], size2[0], size2[1], size2[2]);
    }
    return _boundingBox;
}

void Virvo::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}

void Virvo::drawImplementation(State &state) const
{
    if (_renderer)
    {
        /* TODO: find out from here where view point is, so that call to
       setObjectDirection does not need to come from application.

        Matrix pm = state.getProjectionMatrix();
        Matrix mv = state.getModelViewMatrix();
        Matrix mvInv = Matrix::inverse(mv);
        Matrix pmInv = Matrix::inverse(pm);
        Vec3 eyePos(0,0,-1);
        eyePos = pmInv * eyePos;
        eyePos = mvInv * eyePos;
        cerr << "eyePos=" << eyePos << endl;
    vvVector3 eye(eyePos[0], eyePos[1], eyePos[2]);

    vvVector3 pos;
    renderer->getPosition(&pos);

    vvVector3 objDir;
    objDir.copy(&pos);
    objDir.sub(&eye);

    renderer->setObjectDirection(&objDir);
    */
        if (_visible)
            _renderer->renderVolumeGL();
    }
}

void Virvo::setVisible(bool visible)
{
    _visible = visible;
}

bool Virvo::getVisible()
{
    return _visible;
}

/** Creates a volume with zero data values to be updated later with
updateDynamicVolume().
*/
void Virvo::makeDynamicVolume(const char *name, int w, int h, int s)
{
    vvFileIO *fio;
    vvRenderState renderState;
    float *data;

    if (_renderer)
        renderState = _renderer->_renderState;

    delete _renderer;
    _renderer = NULL;
    delete _vd;
    _vd = NULL;

    // Create empty array:
    data = new float[w * h * s];
    memset(data, 0, w * h * s * sizeof(float)); // initialize with 0's
    _vd = new vvVolDesc(name);
    _vd->vox[0] = w;
    _vd->vox[1] = h;
    _vd->vox[2] = s;
    _vd->bpc = 4;
    _vd->chan = 1;
    _vd->frames = 1;
    _vd->real[0] = 0.0f;
    _vd->real[1] = 1.0f;
    _vd->addFrame((uchar *)data, vvVolDesc::ARRAY_DELETE);

    // Set default color scheme if no TF present:
    _vd->tf.setDefaultAlpha(0);
    _vd->tf.setDefaultColors(0);
    _vd->resizeEdgeMax(1.0f);

    setRenderer(_algorithm, renderState);
}

/** Update data array of dynamic volume. Size of array newData must be the same
as in makeDynamicVolume. The data at newData must be deleted by the _caller_.
  @param newData pointer to updated data array
*/
void Virvo::updateDynamicVolume(double *newData)
{
    double *src;
    float *dst;
    int i;

    // Convert double to float:
    dst = (float *)_vd->getRaw(0);
    src = newData;
    for (i = 0; i < _vd->getFrameVoxels(); ++i)
    {
        *dst = *src;
        ++dst;
        ++src;
    }
    _renderer->updateVolumeData();
}
