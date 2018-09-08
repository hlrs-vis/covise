#pragma once


#include "math/math.h"


namespace virvo
{


/*! project from object coordinates to window coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
template < typename T >
void project(vector< 3, T >* win, vector< 3, T > const& obj, matrix< 4, 4, T > const& modelview,
    matrix< 4, 4, T > const& projection, recti const& viewport);


/*! unproject from window coordinates to project coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
template < typename T >
void unproject(vector< 3, T >* obj, vector< 3, T > const& win, matrix< 4, 4, T > const& modelview,
    matrix< 4, 4, T > const& projection, recti const& viewport);


/*! calc bounding rect of box in screen space coordinates
 */
template < typename T >
recti bounds(aabb const& box, matrix< 4, 4, T > const& modelview,
    matrix< 4, 4, T > const& projection, recti const& viewport);


} // virvo


#include "project.impl.h"


