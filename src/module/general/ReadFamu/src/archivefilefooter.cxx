/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file archivefilefooter.cxx
 * footer of the archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

#include "archivefilefooter.hxx" // footer of the archive file.
#include <time.h>
#include <sstream>

ArchiveFileFooter::ArchiveFileFooter()
{
}

void ArchiveFileFooter::writeObject(ObjectOutputStream *archive) const
{
    archive->writeString(_logFile);
}

ArchiveFileFooter::ArchiveFileFooter(ObjectInputStream *archive)
    : _logFile("")
{
    _logFile = archive->readString();
}

std::string ArchiveFileFooter::getInfo(void) const
{
    std::ostringstream s;
    s << "\n\n";
    s << "=================================== LOG FILE ===================================\n\n";
    s << _logFile << "\n";
    s << "\n\n";
    s << "================================ END OF LOG FILE ===============================\n\n";

    return s.str();
}
