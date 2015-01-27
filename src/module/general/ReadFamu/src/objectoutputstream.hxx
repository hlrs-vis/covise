/** @file objectoutputstream.cxx
 * a stream that can be used for serialization.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "objectoutputstream.hxx"    // a stream that can be used for serialization.

#ifndef __objectoutputstream_hxx__
#define __objectoutputstream_hxx__

#include "os.h"
#include <string>


/**
 * a stream that can be used for serialization.
 */
class ObjectOutputStream
{
public:

    ObjectOutputStream() {};
    virtual ~ObjectOutputStream() {};
    
    // primitives
    virtual void writeCHAR   (CHAR_F c) = 0;               ///< writes a @c CHAR to the dump file
    virtual void writeINT    (INT_F i) = 0;                ///< writes an @c INT to the dump file
    virtual void writeLONG   (const LONG_F& l) = 0;        ///< writes an @c LONG to the dump file
    virtual void writeDouble (const double& d) = 0;      ///< writes a @c double to the dump file
    virtual void writeFloat  (const float& f) = 0;       ///< writes a @c float to the dump file
    virtual void writeBool   (const bool b) = 0;         ///< writes a @c bool to the dump file
    virtual void writePointCC(const PointCC p) = 0;      ///< writes a @c PointCC to the dump file
    
    // arrays
    virtual void writeArray(const double   arr[], const UINT_F& size) = 0;
    virtual void writeArray(const float    arr[], const UINT_F& size) = 0;
    virtual void writeArray(const INT_F    arr[], const UINT_F& size) = 0;
    virtual void writeArray(const UINT_F   arr[], const UINT_F& size) = 0;
    virtual void writeArray(const CHAR_F   arr[], const UINT_F& size) = 0;
    virtual void writeArray(const UCHAR_F  arr[], const UINT_F& size) = 0;
    virtual void writeArray(const bool     arr[], const UINT_F& size) = 0;
    virtual void writeArray(const PointCC  arr[], const UINT_F& size) = 0;
    
    // string
    virtual void writeString(const std::string& s) = 0; ///< writes a @c string to the dump file
    
    // buffer
    virtual void writeBuffer(const CHAR_F* arr, const UINT_F& noOfBytes) = 0; ///< writes an arbitrary number of bytes to the dump file  
    
    
};

#endif
