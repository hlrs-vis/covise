/** @file objectinputstream.cxx
 * a stream that can be used for deserialization.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "objectinputstream.hxx" // a stream that can be used for deserialization.

#ifndef __objectinputstream_hxx__
#define __objectinputstream_hxx__

#include "os.h"
//#include "typen.hxx" // commonly used definitions of types
#include <string>

/**
 * a stream that can be used for deserialization.
 */
class ObjectInputStream
{
public:

    ObjectInputStream() {};
    virtual ~ObjectInputStream() {};

    // primitives    
    virtual CHAR_F  readCHAR(void)    = 0;
    virtual INT_F   readINT(void)     = 0;
    virtual LONG_F  readLONG(void)    = 0;
    virtual double  readDouble(void)  = 0;
    virtual float   readFloat(void)   = 0;
    virtual bool    readBool(void)    = 0;
    virtual PointCC readPointCC(void) = 0;

    //arrays
    virtual void readArray(double*  arr[], UINT_F* size) = 0;
    virtual void readArray(float*   arr[], UINT_F* size) = 0;
    virtual void readArray(INT_F*   arr[], UINT_F* size) = 0;
    virtual void readArray(UINT_F*  arr[], UINT_F* size) = 0;
    virtual void readArray(CHAR_F*  arr[], UINT_F* size) = 0;
    virtual void readArray(UCHAR_F* arr[], UINT_F* size) = 0;
    virtual void readArray(bool*    arr[], UINT_F* size) = 0;
    virtual void readArray(PointCC* arr[], UINT_F* size) = 0;
    
    virtual void readArray(double  arr[], UINT_F size) = 0;
    virtual void readArray(float   arr[], UINT_F size) = 0;
    virtual void readArray(INT_F   arr[], UINT_F size) = 0;
    virtual void readArray(UINT_F  arr[], UINT_F size) = 0;
    virtual void readArray(CHAR_F  arr[], UINT_F size) = 0;
    virtual void readArray(UCHAR_F arr[], UINT_F size) = 0;
    virtual void readArray(bool    arr[], UINT_F size) = 0;
    virtual void readArray(PointCC arr[], UINT_F size) = 0;

    // string
    virtual std::string readString(void) = 0; ///< reads a @c string from the dump file

    // buffer
    virtual void readBuffer(CHAR_F* arr[], UINT_F* noOfBytes, const CHAR_F id = 'a') = 0;      ///< reads an arbitrary number of bytes from the dump file  
    virtual void readBuffer(CHAR_F arr[], const UINT_F& noOfBytes, const CHAR_F id = 'a') = 0; ///< reads an arbitrary number of bytes from the dump file  

};

#endif
