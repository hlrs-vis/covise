/** @file objectinputstreambinarybasic.cxx
 * a stream that can be used for deserialization/binary files, basic functionality.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "../../tools/sourcecode/objectinputstreambinarybasic.hxx" // a stream that can be used for deserialization/binary files, basic functionality.

#ifndef __objectinputstreambinarybasic_hxx__
#define __objectinputstreambinarybasic_hxx__

#include <string>
#include <fstream>

#include "objectinputstream.hxx"    // a stream that can be used for deserialization.
#include "archivefileheader.hxx"    // header of the archive file.
#include "archivefilefooter.hxx"    // footer of the archive file.
#include "OutputHandler.h"          // an output handler for displaying information on the screen.
  

/**
 * a stream that can be used for deserialization/binary files, basic functionality.
 */
class ObjectInputStreamBinaryBasic : public ObjectInputStream
{
public:

    ObjectInputStreamBinaryBasic(std::string filename,
                                 OutputHandler* outputHandler);
    virtual ~ObjectInputStreamBinaryBasic();
    
    // primitives    
    virtual CHAR_F  readCHAR(void);
    virtual INT_F   readINT(void);
    virtual LONG_F  readLONG(void);
    virtual double  readDouble(void);
    virtual float   readFloat(void);
    virtual bool    readBool(void);
    virtual PointCC readPointCC(void);

    //arrays
    virtual void readArray(double*  arr[], UINT_F* size);
    virtual void readArray(float*   arr[], UINT_F* size);
    virtual void readArray(INT_F*   arr[], UINT_F* size);
    virtual void readArray(UINT_F*  arr[], UINT_F* size);
    virtual void readArray(CHAR_F*  arr[], UINT_F* size);
    virtual void readArray(UCHAR_F* arr[], UINT_F* size);
    virtual void readArray(bool*    arr[], UINT_F* size);
    virtual void readArray(PointCC* arr[], UINT_F* size);
    
    virtual void readArray(double  arr[], UINT_F size);
    virtual void readArray(float   arr[], UINT_F size);
    virtual void readArray(INT_F   arr[], UINT_F size);
    virtual void readArray(UINT_F  arr[], UINT_F size);
    virtual void readArray(CHAR_F  arr[], UINT_F size);
    virtual void readArray(UCHAR_F arr[], UINT_F size);
    virtual void readArray(bool    arr[], UINT_F size);
    virtual void readArray(PointCC arr[], UINT_F size);

    // string
    virtual std::string readString(void); ///< reads a @c string from the dump file

    // buffer
    virtual void readBuffer(CHAR_F* arr[], UINT_F* noOfBytes, const CHAR_F id);      ///< reads an arbitrary number of bytes from the dump file  
    virtual void readBuffer(CHAR_F arr[], const UINT_F& noOfBytes, const CHAR_F id); ///< reads an arbitrary number of bytes from the dump file  

protected:
    virtual void readBufferAndVerify(CHAR_F arr[], const UINT_F& noOfBytes); ///< reads an arbitrary number of bytes from the dump file  
    
    OutputHandler*   _outputHandler;
    
    std::string      _filename;
    std::ifstream    _fileHandle;
    ULONG_F          _archiveIndex;
    UINT_F           _checksum;

private:
    ObjectInputStreamBinaryBasic(const ObjectInputStreamBinaryBasic& t);
    ObjectInputStreamBinaryBasic& operator = (const ObjectInputStreamBinaryBasic &t);
};

#endif
