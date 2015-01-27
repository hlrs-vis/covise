/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file objectinputstreambinarybasic.cxx
 * a stream that can be used for deserialization/binary files, basic functionality.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

#include "objectinputstreambinarybasic.hxx" // a stream that can be used for deserialization/binary files.
#include "objectstreambinarystructure.hxx" // definition of binary file format for object streams.

#ifdef _WIN32
#pragma warning(disable : 4267)
#pragma warning(disable : 383)
#endif

ObjectInputStreamBinaryBasic::ObjectInputStreamBinaryBasic(std::string filename,
                                                           OutputHandler *outputHandler)
    : ObjectInputStream()
    , _outputHandler(outputHandler)
    , _filename(filename)
    , _archiveIndex(0)
    , _checksum(0)
{
    _fileHandle.open(_filename.c_str(), std::ifstream::binary);
    std::string err = "cannot open results file ";
    ASSERT1(!_fileHandle.fail(), err, _filename, _outputHandler);
}

ObjectInputStreamBinaryBasic::~ObjectInputStreamBinaryBasic()
{
    _fileHandle.close();
}

// ---------------------- primitives ----------------------------
CHAR_F ObjectInputStreamBinaryBasic::readCHAR(void)
{
    CHAR_F c;
    readBuffer(&c, sizeof(CHAR_F), 'c');
    return c;
}

INT_F ObjectInputStreamBinaryBasic::readINT(void)
{
    INT_F i;
    readBuffer((CHAR_F *)&i, sizeof(INT_F), 'i');
    return i;
}

double ObjectInputStreamBinaryBasic::readDouble(void)
{
    double d;
    readBuffer((CHAR_F *)&d, sizeof(double), 'd');
    return d;
}

float ObjectInputStreamBinaryBasic::readFloat(void)
{
    float f;
    readBuffer((CHAR_F *)&f, sizeof(float), 'f');
    return f;
}

LONG_F ObjectInputStreamBinaryBasic::readLONG(void)
{
    LONG_F l;
    readBuffer((CHAR_F *)&l, sizeof(LONG_F), 'l');
    return l;
}

bool ObjectInputStreamBinaryBasic::readBool(void)
{
    bool b;
    readBuffer((CHAR_F *)&b, sizeof(bool), 'b');
    return b;
}

PointCC ObjectInputStreamBinaryBasic::readPointCC(void)
{
    PointCC p;
    readBuffer((CHAR_F *)&p, sizeof(PointCC), 'p');
    return p;
}

// ---------------------- string ----------------------------

std::string ObjectInputStreamBinaryBasic::readString(void)
{
    CHAR_F *arr = NULL;
    UINT_F noOfBytes = 0;
    readBuffer(&arr, &noOfBytes, 's');

    if (noOfBytes > 0)
    {
        // replace zero-bytes (==terminating flags) by whitepspace
        UINT_F i;
        for (i = 0; i < noOfBytes - 1; i++)
        {
            if (arr[i] == 0x00)
            {
                arr[i] = ' ';
            }
        }
    }
    std::string retval = arr;
    UINT_F length = retval.length();
    ASSERT(length == noOfBytes - 1, _outputHandler);
    DELETE_ARRAY(arr);
    return retval;
}

// ---------------------- arrays ----------------------------
void ObjectInputStreamBinaryBasic::readArray(double *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(double) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(double);
}

void ObjectInputStreamBinaryBasic::readArray(float *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(float) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(float);
}

void ObjectInputStreamBinaryBasic::readArray(INT_F *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(INT_F) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(INT_F);
}

void ObjectInputStreamBinaryBasic::readArray(UINT_F *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(UINT_F) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(UINT_F);
}

void ObjectInputStreamBinaryBasic::readArray(CHAR_F *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(CHAR_F) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(CHAR_F);
}

void ObjectInputStreamBinaryBasic::readArray(UCHAR_F *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(UCHAR_F) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(UCHAR_F);
}

void ObjectInputStreamBinaryBasic::readArray(bool *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(bool) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(bool);
}

void ObjectInputStreamBinaryBasic::readArray(PointCC *arr[], UINT_F *size)
{
    UINT_F noOfBytes = 0;
    readBuffer((CHAR_F **)arr, &noOfBytes, 'a');
    ASSERT1(noOfBytes % sizeof(PointCC) == 0, std::string("cannot read from results file "), _filename, _outputHandler);
    *size = noOfBytes / sizeof(PointCC);
}

void ObjectInputStreamBinaryBasic::readArray(double arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(double), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(float arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(float), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(INT_F arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(INT_F), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(UINT_F arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(UINT_F), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(CHAR_F arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(CHAR_F), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(UCHAR_F arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(UCHAR_F), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(bool arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(bool), 'a');
}

void ObjectInputStreamBinaryBasic::readArray(PointCC arr[], UINT_F size)
{
    readBuffer((CHAR_F *)arr, size * sizeof(PointCC), 'a');
}

// ---------------------- buffer ----------------------------

void ObjectInputStreamBinaryBasic::readBuffer(CHAR_F buffer[], const UINT_F &noOfBytes, const CHAR_F id)
{
    CHAR_F *bufferRead = new CHAR_F[noOfBytes];
    UINT_F noOfBytesRead = 0;
    readBuffer(&bufferRead, &noOfBytesRead, id);
    ASSERT1(noOfBytesRead == noOfBytes, std::string("cannot read from results file "), _filename, _outputHandler);

    UINT_F i;
    for (i = 0; i < noOfBytes; i++)
    {
        buffer[i] = bufferRead[i];
    }
    DELETE_ARRAY(bufferRead);
}

void ObjectInputStreamBinaryBasic::readBuffer(CHAR_F *arr[], UINT_F *noOfBytes, const CHAR_F id)
{
    // read & verify id
    CHAR_F id_read = 0;
    readBufferAndVerify(&id_read, sizeof(CHAR_F));
    ASSERT1(id_read == id, std::string("Error while trying to read from file: file corrupt or of wrong format."), _filename, _outputHandler);

    // read buffer size
    readBufferAndVerify((CHAR_F *)noOfBytes, sizeof(UINT_F));

    // read buffer
    CHAR_F *buff = new CHAR_F[*noOfBytes];
    readBufferAndVerify(buff, sizeof(CHAR_F) * (*noOfBytes));
    *arr = buff;
}

void ObjectInputStreamBinaryBasic::readBufferAndVerify(CHAR_F arr[], const UINT_F &noOfBytes)
{
    _fileHandle.read(arr, sizeof(CHAR_F) * noOfBytes);
    ASSERT1(!_fileHandle.fail(), std::string("cannot read from results file "), _filename, _outputHandler);
    _archiveIndex += sizeof(CHAR_F);

    // update _checksum. algorithm is basically taken from AMIGA bootblock checksum.
    UINT_F i;
    UCHAR_F *checkSumArr = (UCHAR_F *)&_checksum;
    for (i = 0; i < noOfBytes; i++)
    {
        checkSumArr[i % 4] = checkSumArr[i % 4] ^ (UCHAR_F)arr[i];
    }
}
