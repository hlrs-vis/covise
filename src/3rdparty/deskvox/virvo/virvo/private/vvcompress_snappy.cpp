// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#include "vvcompress.h"
#include "vvcompressedvector.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_SNAPPY


#include <snappy.h>


bool virvo::encodeSnappy(std::vector<unsigned char>& data)
{
    if (data.empty())
        return true;

    std::vector<unsigned char> compressed(snappy::MaxCompressedLength(data.size()));

    size_t len = 0;

    snappy::RawCompress((char const*)&data[0], data.size(), (char*)&compressed[0], &len);

    compressed.resize(len);

    data.swap(compressed);

    return true;
}


bool virvo::decodeSnappy(std::vector<unsigned char>& data)
{
    if (data.empty())
        return true;

    size_t len = 0;

    if (!snappy::GetUncompressedLength((char const*)&data[0], data.size(), &len))
        return false;

    std::vector<unsigned char> uncompressed(len);

    if (!snappy::RawUncompress((char const*)&data[0], data.size(), (char*)&uncompressed[0]))
        return false;

    data.swap(uncompressed);

    return true;
}


#else // HAVE_SNAPPY


bool virvo::encodeSnappy(std::vector<unsigned char>& /*data*/)
{
    return false;
}


bool virvo::decodeSnappy(std::vector<unsigned char>& /*data*/)
{
    return false;
}


#endif // !HAVE_SNAPPY


bool virvo::encodeSnappy(CompressedVector& data)
{
    if (data.getCompressionType() != Compress_None)
        return false;

    if (encodeSnappy(data.vector()))
    {
        data.setCompressionType(Compress_Snappy);
        return true;
    }

    return false;
}


bool virvo::decodeSnappy(CompressedVector& data)
{
    if (data.getCompressionType() != Compress_Snappy)
        return false;

    if (decodeSnappy(data.vector()))
    {
        data.setCompressionType(Compress_None);
        return true;
    }

    return false;
}
