/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DGEOLIST_H
#define _CADCV3DGEOLIST_H
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

#include <string>
#include <vector>

#include "CADCv3DSerializable.h"
#include "CADCv3DColor.h"
#include "CADCv3DVertex.h"
#include "CADCv3DPrimitive.h"
#include "CADCv3DGeometry.h"

//! This class provides a serializable container for flat polygonal geometries.
class CADCv3DGeoList : CADCv3DSerializable
{
public:
    //! The constructor.
    CADCv3DGeoList()
        : CADCv3DSerializable()
    {
        reset();
    }
    //! The copy constructor.
    /*! @param src The source instance.
    */
    CADCv3DGeoList(const CADCv3DGeoList &src)
        : CADCv3DSerializable()
    {
        reset();
        copy(src);
    }
    //! The destructor.
    virtual ~CADCv3DGeoList()
    {
        clear();
    }

    //! Creates a new geometry and sets the geometry iterator to point to it.
    /*! @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newGeo();
    //! Sets the geometry iterator to point to the first geometry in the list.
    /*! @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list was empty.
    */
    bool firstGeo();
    //! Sets the geometry iterator to point to the next geometry in the list.
    /*! @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextGeo();
    //! Creates a new primitive in the current geometry and sets the primitive iterator to point to it.
    /*! @param type The type of the primitive (a <code>TYPE_xxx</code> constant).
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newGeoPrimitive(int type);
    //! Sets the primitive iterator to point to the first primitive in the list.
    /*! @param type The variable where to store the type of the primitive in.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list was empty.
    */
    bool firstGeoPrimitive(int &type);
    //! Sets the primitive iterator to point to the next primitive in the list.
    /*! @param type The variable where to store the type of the primitive in.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextGeoPrimitive(int &type);
    //! Creates a new vertex in the current geometry and sets the vertex iterator to point to it.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newGeoVertex(double x, double y, double z);
    //! Sets the vertex iterator to point to the first vertex in the list and returns it's coordinates.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list was empty.
    */
    bool firstGeoVertex(double &x, double &y, double &z);
    //! Sets the vertex iterator to point to the next vertex in the list and returns it's coordinates.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextGeoVertex(double &x, double &y, double &z);
    //! Creates a new normal in the current geometry and sets the normal iterator to point to it.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newGeoNormal(double x, double y, double z);
    //! Sets the normal iterator to point to the first normal in the list and returns it's coordinates.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list was empty.
    */
    bool firstGeoNormal(double &x, double &y, double &z);
    //! Sets the normal iterator to point to the next normal in the list and returns it's coordinates.
    /*! @param x The x coordinate.
    *  @param y The y coordinate.
    *  @param z The z coordinate.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextGeoNormal(double &x, double &y, double &z);
    //! Creates a new color in the current geometry and sets the color iterator to point to it.
    /*! @param r The red component.
    *  @param g The green component.
    *  @param b The blue component.
    *  @param a The alpha component.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newGeoColor(float r, float g, float b, float a);
    //! Sets the color iterator to point to the first color in the list and returns it's rgba components.
    /*! @param r The red component.
    *  @param g The green component.
    *  @param b The blue component.
    *  @param a The alpha component.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list is empty.
    */
    bool firstGeoColor(float &r, float &g, float &b, float &a);
    //! Sets the color iterator to point to the next color in the list and returns it's rgba components.
    /*! @param r The red component.
    *  @param g The green component.
    *  @param b The blue component.
    *  @param a The alpha component.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextGeoColor(float &r, float &g, float &b, float &a);
    //! Creates a new index for the current primitive and sets the index iterator to point to it.
    /*! @param i The index which references a vertex in the vertex list.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool newPrimitiveIndex(int i);
    //! Sets the index iterator to point to the first index in the list and returns it's value.
    /*! @param i The index which references a vertex in the vertex list.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list is empty.
    */
    bool firstPrimitiveIndex(int &i);
    //! Sets the index iterator to point to the next index in the list and returns it's value.
    /*! @param i The index which references a vertex in the vertex list.
    *  @return <code>true</code> on success and <code>false</code> on errors
    *  or when the list's end was exceeded.
    */
    bool nextPrimitiveIndex(int &i);
    //! Returns the number of indices for the current primitive.
    /*! @param i The number of indices for the current primitive.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool getPrimitiveIndexCount(int &i) const;
    //! Inverts the indices of the current primitive.
    /*! You can use that if you inserted the indices previously in the wrong order.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool invertPrimitiveIndices();
    //! Sets the normal binding for the current geometry.
    /*! @param type The type of the normal binding.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool setGeoNormalBinding(int type);
    //! Returns the normal binding for the current geometry.
    /*! @param type The type of the normal binding.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool getGeoNormalBinding(int &type) const;
    //! Sets the color binding for the current geometry.
    /*! @param type The type of the color binding.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool setGeoColorBinding(int type);
    //! Returns the color binding for the current geometry.
    /*! @param type The type of the color binding.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool getGeoColorBinding(int &type) const;
    //! Sets the name of the current geometry.
    /*! @param name The new name for the current geometry.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool setGeoName(const std::string &name);
    //! Returns the name of the current geometry.
    /*! @param name The name of the current geometry.
    *  @return <code>true</code> on success and <code>false</code> on errors.
    */
    bool getGeoName(std::string &name) const;
    //! Clears the geo list.
    void clear();
    //! Resets the iterators for the geo list.
    void reset();
    //! Copies the content of another geo list to this one.
    /*! @param src The source geo list.
    *  @return A reference to this instance.
    */
    const CADCv3DGeoList &copy(const CADCv3DGeoList &src);
    //! The = operator (copies the source to this instance).
    /*! @param src The source instance.
    *  @return A reference to this instance.
    */
    const CADCv3DGeoList &operator=(const CADCv3DGeoList &src)
    {
        return copy(src);
    }

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
    *  @param data A memory location where to write the object to.
    *  The memory must have been allocated previously. You need to allocate at
    *  least <code>calcSize()</code> bytes for this method.
    */
    virtual void write(void *data) const;

    virtual int getNumberOfGeometries() const
    {
        return this->geometries.size();
    }
    virtual int getCurrentGeoIndex() const
    {
        return this->geoIter;
    }

private:
    //! The geometry list.
    std::vector<CADCv3DGeometry *> geometries;
    //! The geometry iterator.
    int geoIter;
    //! The primitive iterator.
    int prmIter;
    //! The vertex iterator.
    int vrtIter;
    //! The normals iterator.
    int nrmIter;
    //! The color iterator.
    int colIter;
    //! The polygon index iterator.
    int idxIter;
};

#endif

// END OF FILE
