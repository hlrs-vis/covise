/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include <config/CoviseConfig.h>

#include "co3dtkDrawable.h"

#ifdef _MSC_VER
#ifdef OPENMP
#define _OPENMP
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "slam6d/point_type.h"
#include "show/display.h"

using namespace osg;
using namespace std;

#undef VERBOSE

co3dtkDrawable::ContextState::ContextState()
{
}

co3dtkDrawable::ContextState::~ContextState()
{
}

co3dtkDrawable::co3dtkDrawable()
{
    init();
}

co3dtkDrawable::co3dtkDrawable(const co3dtkDrawable &drawable,
                               const osg::CopyOp &copyop)
    : Drawable(drawable, copyop)
{
    init();
}

void co3dtkDrawable::init()
{
    pointmode = 0;
    voxelSize = 100;
    // TODO start lower and increase LevelOfDetail = 0.0001;
    LevelOfDetail = 1.0;
    currentFrame = 0;
    frameNr = 0;
    X = 0.0;
    Y = 0.0;
    Z = 0.0;
}

co3dtkDrawable::~co3dtkDrawable()
{
    contextState.clear();
}

void co3dtkDrawable::drawImplementation(RenderInfo &renderInfo) const
{

    const unsigned ctx = renderInfo.getState()->getContextID();
    if (ctx >= contextState.size())
    {
        contextState.resize(ctx + 1);
    }
    // vvRenderer *&renderer = contextState[ctx].renderer;

    DrawPoints();
}

void co3dtkDrawable::calcPointSequence(vector<int> &sequence, int frameNr) const
{
    sequence.clear();
    vector<pair<double, int> > dists;
    double x, y, z;

    for (unsigned int i = 0; i < octpts.size(); i++)
    {
        x = MetaMatrix[i][frameNr][12];
        y = MetaMatrix[i][frameNr][13];
        z = MetaMatrix[i][frameNr][14];
        dists.push_back(pair<double, int>(sqr(X + x) + sqr(Y + y) + sqr(Z + z), i));
    }

    sort(dists.begin(), dists.end());

    for (unsigned int i = 0; i < dists.size(); i++)
    {
        sequence.push_back(dists[i].second);
    }
}

/**
* Displays all data (i.e., points) that are to be displayed
*/
void co3dtkDrawable::DrawPoints() const
{

    // In case of animation
    if (frameNr != 0)
    {
        cm->setMode(ScanColorManager::MODE_ANIMATION);

#ifdef USE_GL_POINTS
        for (int iterator = (int)octpts.size() - 1; iterator >= 0; iterator--)
        {
#else
        for (int iterator = (int)Scan::allScans.size() - 1; iterator >= 0; iterator--)
        {
#endif
            if (MetaAlgoType[iterator][frameNr] == Scan::INVALID)
                continue;
            cm->selectColors(MetaAlgoType[iterator][frameNr]);
            glPushMatrix();
            glMultMatrixd(MetaMatrix[iterator][frameNr]);

            glPointSize(pointSize);
#ifdef USE_GL_POINTS
            ExtractFrustum(pointSize);
            cm->selectColors(MetaAlgoType[iterator][frameNr]);
            if (pointmode == 1)
            {
                octpts[iterator]->display();
            }
            else
            {
                octpts[iterator]->displayLOD(LevelOfDetail);
            }
#else
            for (unsigned int jterator = 0; jterator < vvertexArrayList[iterator].size(); jterator++)
            {

                if ((jterator == 0) && vvertexArrayList[iterator][jterator]->numPointsToRender > 0)
                {
                    cm->selectColors(MetaAlgoType[iterator][frameNr]);
                }

                if (vvertexArrayList[iterator][jterator]->numPointsToRender > 0)
                {
                    glCallList(vvertexArrayList[iterator][jterator]->name);
                }
            }
#endif
            glPopMatrix();
        }
    }
    else
    { // no animation

        // draw point is normal mode
        // -------------------------

        glPointSize(pointSize);

        vector<int> sequence;
        calcPointSequence(sequence, currentFrame);
#ifdef USE_GL_POINTS
        //for(int iterator = (int)octpts.size()-1; iterator >= 0; iterator--) {
        for (unsigned int i = 0; i < sequence.size(); i++)
        {
            int iterator = sequence[i];
#else
        for (int iterator = (int)Scan::allScans.size() - 1; iterator >= 0; iterator--)
        {
#endif
            if (MetaAlgoType[iterator][currentFrame] == Scan::INVALID)
                continue;
            glPushMatrix();
            //if (invert)                               // default: white points on black background
            glColor4d(1.0, 1.0, 1.0, 0.0);
            //else                                      // black points on white background
            //	glColor4d(0.0, 0.0, 0.0, 0.0);

            //  glMultMatrixd(MetaMatrix[iterator].back());
            if (currentFrame != (int)MetaMatrix.back().size() - 1)
            {
                cm->setMode(ScanColorManager::MODE_ANIMATION);
                cm->selectColors(MetaAlgoType[iterator][currentFrame]);
            }
            glMultMatrixd(MetaMatrix[iterator][currentFrame]);

#ifdef USE_GL_POINTS
            //cout << endl << endl;  calcRay(570, 266, 1.0, 40000.0);
            /* // for height mapped color in the vertex shader
			GLfloat v[16];
			for (unsigned int l = 0; l < 16; l++)
			v[l] = MetaMatrix[iterator].back()[l];
			glUniformMatrix4fvARB(glGetUniformLocationARB(p, "MYMAT"), 1, 0, v);
			*/
            ExtractFrustum(pointSize);
            if (pointmode == 1)
            {
                octpts[iterator]->display();
                //octpts[iterator]->displayOctTree(pointSize * pointSize * 5);
            }
            else
            {
                octpts[iterator]->displayLOD(LevelOfDetail);
            }
/*		if (!selected_points[iterator].empty()) {
				glColor4f(1.0, 0.0, 0.0, 1.0);
				glPointSize(pointSize + 2.0);
				glBegin(GL_POINTS);
				for ( set<sfloat*>::iterator it = selected_points[iterator].begin();
					it != selected_points[iterator].end(); it++) {
						glVertex3d((*it)[0], (*it)[1], (*it)[2]);
				}
				glEnd();
				glPointSize(pointSize);
			}*/

#else
            for (unsigned int jterator = 0; jterator < vvertexArrayList[iterator].size(); jterator++)
            {
                if (vvertexArrayList[iterator][jterator]->numPointsToRender > 0)
                {
                    glCallList(vvertexArrayList[iterator][jterator]->name);
                }
            }
#endif
            glPopMatrix();
        }
    }
}

//--------------------------------------------------------------------------------
/*
void selectPoints(int x, int y) {

GLuint selectBuf[BUFSIZE];
GLint hits;
GLint viewport[4];
if (selectOrunselect) {
// set the matrix mode
glMatrixMode(GL_MODELVIEW);
// init modelview matrix
glLoadIdentity();

// do the model-transformation
if (cameraNavMouseMode == 1) {
glRotated( mouseRotX, 1, 0, 0);
glRotated( mouseRotY, 0, 1, 0);
glRotated( mouseRotZ, 0, 0, 1);
} else {
double t[3] = {0,0,0};
double mat[16];
QuatToMatrix4(quat, t, mat);
glMultMatrixd(mat);

glGetFloatv(GL_MODELVIEW_MATRIX, view_rotate_button);
double rPT[3];
Matrix4ToEuler(mat, rPT);
mouseRotX = deg(rPT[0]);
mouseRotY = deg(rPT[1]);
mouseRotZ = deg(rPT[2]);
}
updateControls();
glTranslated(X, Y, Z);       // move camera	

static sfloat *sp2 = 0;

for(int iterator = (int)octpts.size()-1; iterator >= 0; iterator--) {
glPushMatrix();
glMultMatrixd(MetaMatrix[iterator].back());
calcRay(x, y, 1.0, 40000.0);
if (select_voxels) {
octpts[iterator]->selectRay(selected_points[iterator], selection_depth);
} else if (brush_size == 0) {
sfloat *sp = 0;
octpts[iterator]->selectRay(sp);
if (sp != 0) {
cout << "Selected point: " << sp[0] << " " << sp[1] << " " << sp[2] << endl;

if (sp2 != 0) {
cout << "Distance to last point: " << sqrt( sqr(sp2[0] - sp[0]) + sqr(sp2[1] - sp[1]) +sqr(sp2[2] - sp[2])  ) << endl; 
}
sp2 = sp;

selected_points[iterator].insert(sp);
}
} else { // select multiple points with a given brushsize
octpts[iterator]->selectRayBrushSize(selected_points[iterator], brush_size);
}

glPopMatrix();
}

} else {
// unselect points
glGetIntegerv(GL_VIEWPORT, viewport);

glSelectBuffer(BUFSIZE, selectBuf);
(void) glRenderMode(GL_SELECT);

glInitNames();
glPushName(0);

glMatrixMode(GL_PROJECTION);
glPushMatrix();
glLoadIdentity();

//    gluPickMatrix((GLdouble)x, (GLdouble)(viewport[3]-y), 10.0, 10.0, viewport);
gluPickMatrix((GLdouble)x, (GLdouble)(viewport[3]-y), brush_size*2, brush_size*2, viewport);
gluPerspective(cangle, aspect, neardistance, fardistance); 
glMatrixMode(GL_MODELVIEW);
DisplayItFunc(GL_SELECT);

glMatrixMode(GL_PROJECTION);
glPopMatrix();
glMatrixMode(GL_MODELVIEW);

hits = glRenderMode(GL_RENDER);                       // get hits
ProcessHitsFunc(hits, selectBuf);
}
glPopMatrix();
glutPostRedisplay();
/////////////////////////////////////
}
*/

//---------------------------------------------------------------------------------------

void co3dtkDrawable::mapColorToValue(int listboxColorVal)
{
    switch (listboxColorVal)
    {
    case 0:
        cm->setCurrentType(PointType::USE_HEIGHT);
        break;
    case 1:
        cm->setCurrentType(PointType::USE_REFLECTANCE);
        break;
    case 2:
        cm->setCurrentType(PointType::USE_AMPLITUDE);
        break;
    case 3:
        cm->setCurrentType(PointType::USE_DEVIATION);
        break;
    case 4:
        cm->setCurrentType(PointType::USE_TYPE);
        break;
    case 5:
        cm->setCurrentType(PointType::USE_COLOR);
        break;
    default:
        break;
    };
    resetMinMax();
}

void co3dtkDrawable::changeColorMap(int listboxColorMapVal)
{
    ColorMap c;
    GreyMap gm;
    HSVMap hsv;
    SHSVMap shsv;
    JetMap jm;
    HotMap hot;
    DiffMap diff;

    switch (listboxColorMapVal)
    {
    case 0:
        // TODO implement no color map
        cm->setColorMap(c);
        break;
    case 1:
        cm->setColorMap(gm);
        break;
    case 2:
        cm->setColorMap(hsv);
        break;
    case 3:
        cm->setColorMap(jm);
        break;
    case 4:
        cm->setColorMap(hot);
        break;
    case 5:
        cm->setColorMap(diff);
        break;
    case 6:
        cm->setColorMap(shsv);
        break;
    default:
        break;
    }
}

void co3dtkDrawable::resetMinMax()
{
    cm->setMinMax(cm->getMin(), cm->getMax());
}

void co3dtkDrawable::setScansColored(int colorScanVal)
{
    switch (colorScanVal)
    {
    case 0:
        cm->setMode(ScanColorManager::MODE_STATIC);
        break;
    case 1:
        cm->setMode(ScanColorManager::MODE_COLOR_SCAN);
        break;
    case 2:
        cm->setMode(ScanColorManager::MODE_POINT_COLOR);
        break;
    default:
        break;
    }
}
