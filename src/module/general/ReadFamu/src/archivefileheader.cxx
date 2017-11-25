/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file archivefileheader.cxx
 * header of the archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE // has to be before any #include statements
#endif
#include "archivefileheader.hxx" // container for headerrmation about an archive file.
#include <time.h>
#include <sstream>

#ifdef _WIN32
#pragma warning(disable : 383) // "value copied to temporary, reference to temporary used"
#endif

ArchiveFileHeader::ArchiveFileHeader(std::string version,
                                     double versionNumber,
                                     std::string content,
                                     std::string comment,
                                     std::string inputFile)
    : _version(version)
    , _versionNumber(versionNumber)
    , _content(content)
    , _comment(comment)
    , _inputFile(inputFile)
{
    time(&_saveTime);
}

void ArchiveFileHeader::writeObject(ObjectOutputStream *archive) const
{
    archive->writeString(_version);
    archive->writeDouble(_versionNumber);
    archive->writeString(_content);
    archive->writeString(_comment);
    archive->writeString(_inputFile);
    archive->writeBuffer((CHAR_F *)&_saveTime, sizeof(_saveTime));
}

ArchiveFileHeader::ArchiveFileHeader(ObjectInputStream *archive)
    : _version("")
    , _versionNumber(-1)
    , _content("")
    , _comment("")
    , _inputFile("")
{
    _version = archive->readString(); // don't put this in initialization list as order is important!
    _versionNumber = archive->readDouble();
    _content = archive->readString();
    _comment = archive->readString();
    _inputFile = archive->readString();
    archive->readBuffer((CHAR_F *)&_saveTime, sizeof(time_t));
}

std::string ArchiveFileHeader::getSaveTime(void) const
{
    std::ostringstream s;
    s << ctime(&_saveTime);
    return s.str();
}

std::string ArchiveFileHeader::getInfo(void) const
{
    std::ostringstream s;
    s << "================================ GENERAL INFO ==================================\n\n";
    s << "Version:     " << _version << "\n";
    s << "Version No.: " << _versionNumber << "\n";
    s << "Saving Time: " << ctime(&_saveTime) << "\n";
    s << "Content:     " << _content << "\n";
    s << "Comment:     " << _comment << "\n";
    s << "\n";
    s << "Input File:"
      << "\n";
    s << "================================== INPUT FILE ==================================\n\n";

    s << _inputFile << "\n\n";
    return s.str();
}
