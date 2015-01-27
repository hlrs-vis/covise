/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file objectinputstreambinary.cxx
 * a stream that can be used for deserialization/binary files.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

#ifdef _WIN32
#pragma warning(disable : 383)
#endif

#include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.
#include "objectstreambinarystructure.hxx" // definition of binary file format for object streams.

ObjectInputStreamBinary::ObjectInputStreamBinary(std::string filename,
                                                 OutputHandler *outputHandler)

    : ObjectInputStreamBinaryBasic(filename, outputHandler)
    , _outputHandler(outputHandler)
{
    seekPosition(ARCHIVE_TABLE_POS_DATA); // set position for reading of actual data
}

ObjectInputStreamBinary::~ObjectInputStreamBinary()
{
    if (_fileHandle.eof())
    {
        seekPosition(ARCHIVE_TABLE_POS_CHECKSUM);
        UINT_F checksumRead = 0;
        _fileHandle.read((CHAR_F *)&checksumRead, sizeof(UINT_F)); // must not be read with readBuffer() because leading 'number of bytes' is missing in the file
    }
    _fileHandle.close();
}

void ObjectInputStreamBinary::seekPosition(const ULONG_F &tablePos)
{
    ULONG_F position = 0;
    _fileHandle.seekg((INT_F)tablePos, std::ios::beg);
    _fileHandle.read((CHAR_F *)&position, sizeof(ULONG_F));
    _fileHandle.seekg((INT_F)position, std::ios::beg);
}

void ObjectInputStreamBinary::writeFileInfo(std::string resultsFilename)
{
    (void)resultsFilename;
}
