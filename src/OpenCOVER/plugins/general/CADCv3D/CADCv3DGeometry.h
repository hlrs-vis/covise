/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DGEOMETRY_H
#define _CADCV3DGEOMETRY_H
/****************************************************************************\
 **                 (C)2007 Titus Miloi, ZAIK/RRZK, University of Cologne  **
 **                                                                        **
 ** Description: 3d container classes for geometry transfer                **
 **                                                                        **
 **                                                                        **
 ** Author: Titus Miloi                                                    **
 **                                                                        **
 ** History:                                                               **
 ** 2007-08-02 v0.1                                                        **
 **                                                                        **
 ** $LastChangedDate: 2009-03-25 17:16:38 +0100 (Mi, 25 Mrz 2009) $
 ** $Revision: 770 $
 ** $LastChangedBy: miloit $
 **                                                                        **
\****************************************************************************/

#include <vector>
#include <string>

#include "CADCv3DSerializable.h"
#include "CADCv3DColor.h"
#include "CADCv3DVertex.h"
#include "CADCv3DPrimitive.h"

//! Stores a geometry (a buch of primitives).
/*! With the functions which end with a <code>P</code>, you can also
 *  directly insert pointers to colors, vertices, normals, etc. into the
 *  geometry (the default insert functions copy the object to be inserted).
 *  However, keep in mind, that all pointers in the geometry will be deleted
 *  when the geometry is cleared or destroyed. Also, the pointers will not be
 *  copied when geometries are copied, but the objects where the pointers point
 *  to will be replicated.
 *  Note: This class is serializable for network transmission.
 */
class CADCv3DGeometry : public CADCv3DSerializable
{
public:
    //! Constant for polygon/color binding: per vertex
    static const int BIND_PERVERTEX = 1;
    //! Constant for polygon/color binding: per polygon
    static const int BIND_PERPRIMITIVE = 2;
    //! Constant for polygon/color binding: per geometry
    static const int BIND_PERGEOMETRY = 3;

    //! The constructor.
    CADCv3DGeometry()
        : CADCv3DSerializable()
    {
    }
    //! The copy constructor.
    /*! @param src The source instance.
    */
    CADCv3DGeometry(const CADCv3DGeometry &src)
        : CADCv3DSerializable()
    {
        copy(src);
    }
    //! The destructor.
    virtual ~CADCv3DGeometry()
    {
        clear();
    }

    //! Adds a new color at the end of the geometry's color list.
    /*! @param col The new color.
    */
    void pushColor(const CADCv3DColor &col)
    {
        colors.push_back(new CADCv3DColor(col));
    }
    //! Adds a new color at the end of the geometry's color list.
    /*! @param col A pointer to the color to add to the list.
    */
    void pushColorP(CADCv3DColor *col)
    {
        colors.push_back(col);
    }
    //! Returns the color at the specified position in the color list.
    /*! @param pos The position of the color to return.
    *  @param col The object where to store the color in.
    *  @return <code>true</code> on success and <code>false</code> if
    *  the specified index exceeds the bounds of the color list.
    */
    bool getColor(int pos, CADCv3DColor &col) const;
    //! Returns a pointer to the color at the specified position in the color list.
    /*! @param pos The position of the color to return.
    *  @return A pointer to the specified color or <code>0</code> if
    *  the specified index exceeds the bounds of the color list.
    */
    CADCv3DColor *getColorP(int pos);
    //! Returns the total number of colors in the geometry.
    /*! @return See above.
    */
    int countColors() const
    {
        return (int)colors.size();
    }
    //! Adds a new vertex at the end of the geometry's vertex list.
    /*! @param v The new vertex.
    */
    void pushVertex(const CADCv3DVertex &v)
    {
        vertices.push_back(new CADCv3DVertex(v));
    }
    //! Adds a new vertex at the end of the geometry's vertex list.
    /*! @param v A pointer to the vertex to add to the list.
    */
    void pushVertexP(CADCv3DVertex *v)
    {
        vertices.push_back(v);
    }
    //! Returns the vertex at the specified position in the vertex list.
    /*! @param pos The position of the vertex to return.
    *  @param v The object where to store the vertex in.
    *  @return <code>true</code> on success and <code>false</code> if
    *  the specified index exceeds the bounds of the vertex list.
    */
    bool getVertex(int pos, CADCv3DVertex &v) const;
    //! Returns a pointer to the vertex at the specified position in the vertex list.
    /*! @param pos The position of the vertex to return.
    *  @return A pointer to the specified vertex or <code>0</code> if
    *  the specified index exceeds the bounds of the vertex list.
    */
    CADCv3DVertex *getVertexP(int pos);
    //! Returns the total number of vertices in the geometry.
    /*! @return See above.
    */
    int countVertices() const
    {
        return (int)vertices.size();
    }
    //! Adds a new normal at the end of the geometry's normal list.
    /*! @param n The new normal.
    */
    void pushNormal(const CADCv3DVertex &n)
    {
        normals.push_back(new CADCv3DVertex(n));
    }
    //! Adds a new normal at the end of the geometry's normal list.
    /*! @param n A pointer to the normal to add to the list.
    */
    void pushNormalP(CADCv3DVertex *n)
    {
        normals.push_back(n);
    }
    //! Returns the normal at the specified position in the normal list.
    /*! @param pos The position of the normal to return.
    *  @param n The object where to store the normal in.
    *  @return <code>true</code> on success and <code>false</code> if
    *  the specified index exceeds the bounds of the normal list.
    */
    bool getNormal(int pos, CADCv3DVertex &n) const;
    //! Returns a pointer to the normal at the specified position in the normal list.
    /*! @param pos The position of the normal to return.
    *  @return A pointer to the specified normal or <code>0</code> if
    *  the specified index exceeds the bounds of the normal list.
    */
    CADCv3DVertex *getNormalP(int pos);
    //! Returns the total number of normals in the geometry.
    /*! @return See above.
    */
    int countNormals() const
    {
        return (int)normals.size();
    }
    //! Adds a new primitive at the end of the geometry's primitive list.
    /*! @param p The new primitive.
    */
    void pushPrimitive(const CADCv3DPrimitive &p)
    {
        primitives.push_back(new CADCv3DPrimitive(p));
    }
    //! Adds a new primitive at the end of the geometry's primitive list.
    /*! @param p A pointer to the primitive to add to the list.
    */
    void pushPrimitiveP(CADCv3DPrimitive *p)
    {
        primitives.push_back(p);
    }
    //! Returns the primitive at the specified position in the primitive list.
    /*! @param pos The position of the primitive to return.
    *  @param p The object where to store the primitive in.
    *  @return <code>true</code> on success and <code>false</code> if
    *  the specified index exceeds the bounds of the primitive list.
    */
    bool getPrimitive(int pos, CADCv3DPrimitive &p) const;
    //! Returns a pointer to the primitive at the specified position in the primitive list.
    /*! @param pos The position of the primitive to return.
    *  @return A pointer to the specified primitive or <code>0</code> if
    *  the specified index exceeds the bounds of the primitive list.
    */
    CADCv3DPrimitive *getPrimitiveP(int pos);
    //! Returns the total number of primitives in the geometry.
    /*! @return See above.
    */
    int countPrimitives() const
    {
        return (int)primitives.size();
    }
    //! Returns the color binding type.
    /*! @return The color binding type.
    */
    int getColorBinding() const
    {
        return colorBinding;
    }
    //! Sets the color binding type.
    /*! @param v The color binding type.
    */
    void setColorBinding(int v)
    {
        colorBinding = v;
    }
    //! Returns the normal binding type.
    /*! @return The normal binding type.
    */
    int getNormalBinding() const
    {
        return normalBinding;
    }
    //! Sets the normal binding type.
    /*! @param v The normal binding type.
    */
    void setNormalBinding(int v)
    {
        normalBinding = v;
    }
    //! Returns the name of the geometry.
    /*! @return The name of the geometry.
    */
    const std::string &getName() const
    {
        return name;
    }
    //! Sets the name of the geometry.
    /*! @param v The name of the geometry.
    */
    void setName(const std::string &v)
    {
        name = v;
    }
    //! Clears the object.
    void clear();

    //! Calculates the size of the serialized object.
    /*! Implements <code>CADCv3DSerializable::calcSize()</code>.
    *  @return The size in bytes of the serialized object.
    */
    virtual unsigned int calcSize() const;
    //! Reads the object from a memory location.
    /*! Implements <code>CADCv3DSerializable::read()</code>.
    *  @param data A memory location where to read the object from.
    *  @param size The maximum amount of bytes to read. This will not
    *  be exceeded. Set this to 0 to not set a maximum size.
    *  @return <code>true</code> if the data was copied completely or
    *  <code>false</code> if errors occured and no object was read in.
    */
    virtual bool read(const void *data, unsigned int size = 0);
    //! Writes the object to a memory location.
    /*! Implements <code>CADCv3DSerializable::write()</code>.
    *  @param data A memory location where to read the object from.
    *  The memory must have been allocated previously. You need to allocate at
    *  least <code>calcSize()</code> bytes for this method.
    */
    virtual void write(void *data) const;
    //! Copies the content of another geometry to this one.
    /*! @param src The source geometry.
    *  @return A reference to this instance.
    */
    const CADCv3DGeometry &copy(const CADCv3DGeometry &src);
    //! The = operator (copies the source to this instance).
    /*! @param src The source instance.
    *  @return A reference to this instance.
    */
    const CADCv3DGeometry &operator=(const CADCv3DGeometry &src)
    {
        return copy(src);
    }

private:
    //! The color list for the geometry.
    std::vector<CADCv3DColor *> colors;
    //! The vertex list for the geometry.
    std::vector<CADCv3DVertex *> vertices;
    //! The normals list for the geometry.
    std::vector<CADCv3DVertex *> normals;
    //! The polygon list for the geometry.
    std::vector<CADCv3DPrimitive *> primitives;
    //! The color binding type.
    int colorBinding;
    //! The normal binding type.
    int normalBinding;
    //! The name of the geometry.
    std::string name;
};

#endif

// END OF FILE
