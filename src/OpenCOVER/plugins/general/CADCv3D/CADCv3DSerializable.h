/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3DSERIALIZABLE_H
#define _CADCV3DSERIALIZABLE_H
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

#include <QtGlobal>

//! This union is used for copying floats over the network.
typedef union
{
    //! The floating point value.
    float val;
    //! The 32 bit data.
    quint32 data;
} CADCv3DFloat;

//! This union is used for copying doubles over the network.
typedef union
{
    //! The double value.
    double val;
    //! The 32 bit data.
    quint64 data;
} CADCv3DDouble;

//! This interface must be inherited and implemented by any 3D class which can be serialized.
class CADCv3DSerializable
{
public:
    //! The constructor.
    CADCv3DSerializable()
    {
    }
    //! The destructor.
    virtual ~CADCv3DSerializable()
    {
    }

    //! Calculates the size of the serialized object.
    /*! @return The size in bytes of the serialized object.
    */
    virtual unsigned int calcSize() const = 0;
    //! Reads the object from a memory location.
    /*! @param data A memory location where to read the object from.
    *  @param size The maximum amount of bytes to read. This will not
    *  be exceeded. Set this to 0 to not set a maximum size.
    *  @return <code>true</code> if the data was copied completely or
    *  <code>false</code> if errors occured and no object was read in.
    */
    virtual bool read(const void *data, unsigned int size = 0) = 0;
    //! Writes the object to a memory location.
    /*! @param data A memory location where to read the object from.
    *  The memory must have been allocated previously. You need to allocate at
    *  least <code>calcSize()</code> bytes for this method.
    */
    virtual void write(void *data) const = 0;
};

#endif

// END OF FILE
