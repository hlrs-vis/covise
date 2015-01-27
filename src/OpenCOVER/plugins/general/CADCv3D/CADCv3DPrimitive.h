/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DPRIMITIVE_H
#define _CADCV3DPRIMITIVE_H
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

#include "CADCv3DSerializable.h"

//! Stores a primitive.
/*! Note: This class is serializable for network transmission.
 */
class CADCv3DPrimitive : public CADCv3DSerializable
{
public:
    //! Primitive type constant: loose points.
    static const int TYPE_POINTS = 0;
    //! Primitive type constant: loose lines.
    static const int TYPE_LINES = 1;
    //! Primitive type constant: a line strip.
    static const int TYPE_LINE_STRIP = 2;
    //! Primitive type constant: a line loop.
    static const int TYPE_LINE_LOOP = 3;
    //! Primitive type constant: loose triangles.
    static const int TYPE_TRIANGLES = 4;
    //! Primitive type constant: a triangle strip.
    static const int TYPE_TRIANGLE_STRIP = 5;
    //! Primitive type constant: a triangle fan.
    static const int TYPE_TRIANGLE_FAN = 6;
    //! Primitive type constant: loose quads.
    static const int TYPE_QUADS = 7;
    //! Primitive type constant: a strip of quads.
    static const int TYPE_QUAD_STRIP = 8;
    //! Primitive type constant: a polygon.
    static const int TYPE_POLYGON = 9;

    //! The constructor.
    CADCv3DPrimitive()
        : CADCv3DSerializable()
    {
        type = TYPE_POINTS;
    }
    //! The copy constructor.
    /*! @param src The source instance.
    */
    CADCv3DPrimitive(const CADCv3DPrimitive &src)
        : CADCv3DSerializable()
    {
        copy(src);
    }
    //! The initializing contructor.
    /*! @param t The primitive type. See <code>TYPE_xxx</code> constants.
    */
    CADCv3DPrimitive(int t)
        : CADCv3DSerializable()
    {
        type = t;
    }

    //! Sets the type of this primitive. See <code>TYPE_xxx</code> constants.
    /*! @param t The new type for this primitive.
    */
    void setType(int t)
    {
        type = t;
    }
    //! Returns the type of this primitive. See <code>TYPE_xxx</code> constants.
    /*! @return See above.
    */
    int getType() const
    {
        return type;
    }
    //! Adds a new index at the end of the primitive's index list.
    /*! @param i The new Index. Must be greader than or equal to 0.
    */
    void pushIndex(int i)
    {
        indices.push_back(i);
    }
    //! Returns the index at the specified position in the index list.
    /*! @param pos The position of the index to return.
    *  @return The indexes value or -1 if the specified position
    *  exceeds the number of indices.
    */
    int getIndex(int pos) const;
    //! Returns the total number of indices in the primitive.
    /*! @return See above.
    */
    int countIndices() const
    {
        return indices.size();
    }
    //! Inverts the indices of the primitive.
    void invert();
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
    //! Copies the content of another primitive to this one.
    /*! @param src The source primitive.
    *  @return A reference to this instance.
    */
    const CADCv3DPrimitive &copy(const CADCv3DPrimitive &src);
    //! The = operator (copies the source to this instance).
    /*! @param src The source instance.
    *  @return A reference to this instance.
    */
    const CADCv3DPrimitive &operator=(const CADCv3DPrimitive &src)
    {
        return copy(src);
    }

private:
    //! The vertex indices for this polygon.
    std::vector<int> indices;
    //! The type of the primitive. See <code>TYPE_xxx</code> constants.
    int type;
};

#endif

// END OF FILE
