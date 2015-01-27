/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DCOLOR_H
#define _CADCV3DCOLOR_H
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

//! Default ambient red component for cad models.
#define CADCV3D_AMBIENT_R 0.25f
//! Default ambient green component for cad models.
#define CADCV3D_AMBIENT_G 0.25f
//! Default ambient blue component for cad models.
#define CADCV3D_AMBIENT_B 0.25f
//! Default diffuse red component for cad models.
#define CADCV3D_DIFFUSE_R 1.0f
//! Default diffuse green component for cad models.
#define CADCV3D_DIFFUSE_G 1.0f
//! Default diffuse blue component for cad models.
#define CADCV3D_DIFFUSE_B 1.0f
//! Default specular red component for cad models.
#define CADCV3D_SPECULAR_R 0.8f
//! Default specular green component for cad models.
#define CADCV3D_SPECULAR_G 0.8f
//! Default specular blue component for cad models.
#define CADCV3D_SPECULAR_B 0.8f
//! Default emissive red component for cad models.
#define CADCV3D_EMISSIVE_R 0.0f
//! Default emissive green component for cad models.
#define CADCV3D_EMISSIVE_G 0.0f
//! Default emissive blue component for cad models.
#define CADCV3D_EMISSIVE_B 0.0f
//! Default shininess for cad models.
#define CADCV3D_SHININESS 9.0f

//! Stores a color with alpha channel.
/*! Note: This class is serializable for network transmission.
 */
class CADCv3DColor : public CADCv3DSerializable
{
public:
    //! The default constructor.
    CADCv3DColor()
        : CADCv3DSerializable()
    {
        r.val = 0.0;
        g.val = 0.0;
        b.val = 0.0;
        a.val = 1.0;
    }
    //! The copy constructor.
    /*! @param src The source instance.
    */
    CADCv3DColor(const CADCv3DColor &src)
        : CADCv3DSerializable()
    {
        copy(src);
    }
    //! The initializing constructor.
    /*! @param red The red component.
    *  @param green The green component.
    *  @param blue The blue component.
    *  @param alpha The alpha component.
    */
    CADCv3DColor(float red, float green, float blue, float alpha)
        : CADCv3DSerializable()
    {
        r.val = red;
        g.val = green;
        b.val = blue;
        a.val = alpha;
    }

    //! Returns the red component.
    /*! @return The red component.
    */
    float red() const
    {
        return r.val;
    }
    //! Returns the green component.
    /*! @return The green component.
    */
    float green() const
    {
        return g.val;
    }
    //! Returns the blue component.
    /*! @return The blue component.
    */
    float blue() const
    {
        return b.val;
    }
    //! Returns the alpha component.
    /*! @return The alpha component.
    */
    float alpha() const
    {
        return a.val;
    }
    //! Sets the red component.
    /*! @param v The red component.
    */
    void setRed(float v)
    {
        r.val = v;
    }
    //! Sets the green component.
    /*! @param v The green component.
    */
    void setGreen(float v)
    {
        g.val = v;
    }
    //! Sets the blue component.
    /*! @param v The blue component.
    */
    void setBlue(float v)
    {
        b.val = v;
    }
    //! Sets the alpha component.
    /*! @param v The alpha component.
    */
    void setAlpha(float v)
    {
        a.val = v;
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
    //! Copies the content of another color to this one.
    /*! @param src The source color.
    *  @return A reference to this instance.
    */
    const CADCv3DColor &copy(const CADCv3DColor &src);
    //! The = operator (copies the source to this instance).
    /*! @param src The source instance.
    *  @return A reference to this instance.
    */
    const CADCv3DColor &operator=(const CADCv3DColor &src)
    {
        return copy(src);
    }

    //! Returns the default size of this class' instances.
    /*! @return The size in bytes of the serialized instacne.
    */
    static unsigned int size();

private:
    //! The red component.
    CADCv3DFloat r;
    //! The green component.
    CADCv3DFloat g;
    //! The blue component.
    CADCv3DFloat b;
    //! The alpha component.
    CADCv3DFloat a;
};

#endif

// END OF FILE
