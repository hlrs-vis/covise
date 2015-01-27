/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Debug.cpp

#include "Debug.h"

#include <QtGlobal> // qDebug

Logger::Logger()
{
}

Logger::~Logger()
{
    qDebug("%s", m_stream.str().c_str());
}
