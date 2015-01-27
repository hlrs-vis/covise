/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DVERTEX_H
#define _CADCV3DVERTEX_H
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

#include "CADCv3DSerializable.h"

//! Stores a 3D vertex.
/*! Note: This class is serializable for network transmission.
 */
class CADCv3DVertex : public CADCv3DSerializable
{
public:
    //! The default constructor.
    CADCv3DVertex()
        : CADCv3DSerializable()
    {
        xx.val = 0.0;
        yy.val = 0.0;
        zz.val = 0.0;
    }
    //! The copy constructor.
    /*! @param src The source instance.
    */
    CADCv3DVertex(const CADCv3DVertex &src)
        : CADCv3DSerializable()
    {
        copy(src);
    }
    //! The initializing constructor.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    */
    CADCv3DVertex(double x, double y, double z)
        : CADCv3DSerializable()
    {
        xx.val = x;
        yy.val = y;
        zz.val = z;
    }
    //! Returns the x coordinate.
    /*! @return The x coordinate.
    */
    double x() const
    {
        return xx.val;
    }
    //! Returns the y coordinate.
    /*! @return The y coordinate.
    */
    double y() const
    {
        return yy.val;
    }
    //! Returns the z coordinate.
    /*! @return The z coordinate.
    */
    double z() const
    {
        return zz.val;
    }
    //! Sets the x coordinate.
    /*! @param v The x coordinate.
    */
    void setX(double v)
    {
        xx.val = v;
    }
    //! Sets the y coordinate.
    /*! @param v The y coordinate.
    */
    void setY(double v)
    {
        yy.val = v;
    }
    //! Sets the z coordinate.
    /*! @param v The z coordinate.
    */
    void setZ(double v)
    {
        zz.val = v;
    }
    //! Calculates the size of the serialized object.
    /*! Implements <code>CADCv3DSerializable::calcSize()</code>.
    *  @return The size in bytes of the serialized object.
    */
    virtual unsigned int calcSize() const
    {
        return size();
    }
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
    //! Copies the content of another vertex to this one.
    /*! @param src The source vertex.
    *  @return A reference to this instance.
    */
    const CADCv3DVertex &copy(const CADCv3DVertex &src);
    //! The = operator (copies the source to this instance).
    /*! @param src The source instance.
    *  @return A reference to this instance.
    */
    const CADCv3DVertex &operator=(const CADCv3DVertex &src)
    {
        return copy(src);
    }

    //! Returns the default size of this class' instances.
    /*! @return The size in bytes of the serialized instacne.
    */
    static unsigned int size();

private:
    //! The x coordinate.
    CADCv3DDouble xx;
    //! The y coordinate.
    CADCv3DDouble yy;
    //! The z coordinate.
    CADCv3DDouble zz;
};

#endif

// END OF FILE
