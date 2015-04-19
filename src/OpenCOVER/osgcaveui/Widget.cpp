/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>
#include <assert.h>
#include <cstdlib>

// Covise:
#include <config/CoviseConfig.h>

// Local:
#include "Widget.h"

using namespace osg;
using namespace cui;
using namespace std;

const float Widget::EPSILON_Z = 0.02f;
const Vec4 Widget::COL_RED(1.0, 0.0, 0.0, 1.0);
const Vec4 Widget::COL_GREEN(0.0, 1.0, 0.0, 1.0);
const Vec4 Widget::COL_BLUE(0.0, 0.0, 1.0, 1.0);
const Vec4 Widget::COL_BLACK(0.0, 0.0, 0.0, 1.0);
const Vec4 Widget::COL_WHITE(1.0, 1.0, 1.0, 1.0);
const Vec4 Widget::COL_LIGHT_GRAY(0.7, 0.7, 0.7, 1.0);
const Vec4 Widget::COL_DARK_GRAY(0.3, 0.3, 0.3, 1.0);
const Vec4 Widget::COL_YELLOW(1.0, 1.0, 0.0, 1.0);
std::string Widget::_resourcePath;

int Widget::ID_COUNTER = 0;

int Widget::getNewGroupID()
{
    ID_COUNTER++;
    return ID_COUNTER;
}

Widget::Widget()
{
    _font = NULL;
    _focus = false;
    _highlighted = false;
    _node = new MatrixTransform();
    _visible = true;
    _inside = false;
    _groupID = 0;
    _logFile = 0;
    initResourcePath();
}

Widget::~Widget()
{
    // just for having it virtual
}

void Widget::initResourcePath()
{
    std::string dollarG;
#ifdef WIN32
    char delim = '\\';
#else
    char delim = '/';
#endif
    static bool resourcePathSet = false;
    if (!resourcePathSet)
    {
        resourcePathSet = true;
        if (const char *covisedir = getenv("COVISEDIR"))
        {
            std::string resPath(covisedir);
            resPath += "/share/cui/";
            _resourcePath = resPath;
        }
        else
        {
            std::string resPath = covise::coCoviseConfig::getEntry("CUI.ResourcesDirectory");
            if (!resPath.empty())
            {
                std::string tmpString(resPath);
                _resourcePath = tmpString + delim;
                cerr << "CUI.ResourcesDirectory =" << _resourcePath << endl;
            }
            else
            {
                cerr << "Error: CUI.ResourcesDirectory must be set in config file." << endl;
            }
        }
    }
}

void Widget::setMatrix(const Matrix &matrix)
{
    _node->setMatrix(matrix);
}

Matrix Widget::getMatrix()
{
    return _node->getMatrix();
}

MatrixTransform *Widget::getNode()
{
    return _node.get();
}

bool Widget::isHighlighted()
{
    return _highlighted;
}

void Widget::setHighlighted(bool highlighted)
{
    _highlighted = highlighted;
}

bool Widget::hasFocus()
{
    return _focus;
}

void Widget::setFocus(bool focus)
{
    _focus = focus;
}

void Widget::setFont(osgText::Font *font)
{
    _font = font;
}

void Widget::setVisible(bool isVisible)
{
    //cerr << "Visible Called" << endl;
    _visible = isVisible;
    _node->setNodeMask((_visible) ? (~0) : 0);
}

bool Widget::isVisible()
{
    return _visible;
}

/** @return wand angle around positive x/y/z axis [degrees]
 */
float Widget::angle(Matrix &xf, AngleType angle)
{
    float x, y, z; // angles [radians]
    computeEulerAngles(xf, x, y, z);
    switch (angle)
    {
    case X:
        return x * 180.0f / M_PI;
        break;
    case Y:
        return y * 180.0f / M_PI;
        break;
    case Z:
        return z * 180.0f / M_PI;
        break;
    default:
        assert(0);
        break;
    }
    return 0.0f; // just to make compilers happy
}

/** Compute Euler angles for a matrix. The angles are returned in Radians.
  Source: http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q37
*/
void Widget::computeEulerAngles(Matrix &matrix, float &angleX, float &angleY, float &angleZ)
{
    float tx, ty, c;

    double *mat = matrix.ptr();

    angleY = asinf(mat[2]); // Calculate Y-axis angle
    c = cosf(angleY);
    if (fabs(c) > 0.005f) // Gimball lock?
    {
        tx = mat[10] / c; // No, so get X-axis angle
        ty = -mat[6] / c;
        angleX = atan2f(ty, tx);
        tx = mat[0] / c; // Get Z-axis angle
        ty = -mat[1] / c;
        angleZ = atan2f(ty, tx);
    }
    else // Gimball lock has occurred
    {
        angleX = 0.0f; // Set X-axis angle to zero
        tx = mat[5]; // And calculate Z-axis angle
        ty = mat[4];
        angleZ = atan2f(ty, tx);
    }

    // Return only positive angles in [0, 2*VV_PI]:
    if (angleX < 0.0f)
        angleX += 2.0f * M_PI;
    if (angleY < 0.0f)
        angleY += 2.0f * M_PI;
    if (angleZ < 0.0f)
        angleZ += 2.0f * M_PI;
}

/** @return difference in wand angles around positive x/y/z axis [degrees]
            positive value = clockwise rotation
*/
float Widget::angleDiff(Matrix &lastWand2w, Matrix &newWand2w, AngleType at)
{
    Matrix newW2Wand = Matrix::inverse(newWand2w);
    Matrix diffWandXF = lastWand2w * newW2Wand;
    float diffAngle = angle(diffWandXF, at);
    if (diffAngle < -180.0f)
        diffAngle += 360.0f;
    if (diffAngle > 180.0f)
        diffAngle -= 360.0f;
    return diffAngle;
}

/** Calculates the angle between a vector and a plane. vec and normal
  don't need to be normalized.
  @param vec vector to determine angle for
  @param normal normal of a plane through origin
  @return a value between -pi and +pi.
*/
float Widget::vectorAnglePlane(const Vec3 &vec, const Vec3 &normal)
{
    Vec3 norm1(vec);
    Vec3 norm2(normal);
    norm1.normalize();
    norm2.normalize();
    float absAngle = M_PI / 2.0 - acosf(norm1 * norm2);
    return absAngle;
}

/** Calculates the angle between two vectors in a plane. vec1 and vec2
  don't need to be normalized.
  @return a value between -pi and +pi.
*/
float Widget::vectorAngle(const Vec3 &vec1, const Vec3 &vec2, int axis)
{
    Vec3 norm1(vec1);
    Vec3 norm2(vec2);
    norm1[axis] = 0.0f;
    norm2[axis] = 0.0f;
    norm1.normalize();
    norm2.normalize();
    float absAngle = acosf(norm1 * norm2);
    Vec3 cross = norm1 ^ norm2;
    float sign = (cross[axis] >= 0.0f) ? 1.0f : -1.0f;
    float angle = absAngle * sign;
    return angle;
}

bool Widget::isInside()
{
    return _inside;
}
