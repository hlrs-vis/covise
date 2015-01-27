/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:             ++
// ++                                                                     ++
// ++ Author:                  ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#define DEFINE_HTTP_HEADERS
#include "header.h"

using namespace std;

Header::Header(char *name)
{
    m_name = new char[strlen(name) + 1];
    strcpy(m_name, name);
    m_value = NULL;
}

Header::Header(char *name, char *value)
{
    m_name = new char[strlen(name) + 1];
    strcpy(m_name, name);
    m_value = new char[strlen(value) + 1];
    strcpy(m_value, value);
}

void Header::Set(char *name, char *value)
{
    if (m_name != NULL)
        delete[] m_name;
    m_name = new char[strlen(name) + 1];
    strcpy(m_name, name);
    if (m_value != NULL)
        delete[] m_value;
    m_value = new char[strlen(value) + 1];
    strcpy(m_value, value);
}

Header::~Header()
{
    if (m_name != NULL)
        delete[] m_name;
    if (m_value != NULL)
        delete[] m_value;
}

void Header::print(void)
{
    cerr << m_name << ":" << m_value << endl;
}
