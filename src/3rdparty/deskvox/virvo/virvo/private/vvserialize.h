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

#ifndef VV_PRIVATE_SERIALIZE_H
#define VV_PRIVATE_SERIALIZE_H

#define DESKVOX_USE_ASIO_BINARY_ARCHIVE 1

#include "archives.h"

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>

#include <cstdio>
#include <vector>

namespace virvo
{

    template<class T>
    bool serialize(std::vector<char>& buffer, T const& object)
    {
        typedef boost::iostreams::back_insert_device<std::vector<char> > sink_type;
        typedef boost::iostreams::stream<sink_type> stream_type;

        try
        {
            sink_type sink(buffer);
            stream_type stream(sink);

            {
                // Create a serializer
#if DESKVOX_USE_ASIO_BINARY_ARCHIVE
                boost::archive::binary_oarchive archive(stream);
#else
                boost::archive::text_oarchive archive(stream);
#endif

                // Serialize the message
                archive << object;

                // Don't forget to flush the stream!!!
                stream.flush();
            }
            // ~archive

            return stream.good();
        }
        catch (std::exception& e)
        {
#ifndef NDEBUG
            printf("virvo::serialize: %s\n", e.what());
#else
            static_cast<void>(e);
#endif
            return false;
        }
    }

    template<class T>
    bool deserialize(T& object, std::vector<char> const& buffer)
    {
        typedef boost::iostreams::basic_array_source<char> source_type;
        typedef boost::iostreams::stream<source_type> stream_type;

        if (buffer.empty())
            return false;

        try
        {
            source_type source(&buffer[0], buffer.size());
            stream_type stream(source);

            // Create a deserialzer
#if DESKVOX_USE_ASIO_BINARY_ARCHIVE
            boost::archive::binary_iarchive archive(stream);
#else
            boost::archive::text_iarchive archive(stream);
#endif

            // Deserialize the message
            archive >> object;

            return stream.good();
        }
        catch (std::exception& e)
        {
#ifndef NDEBUG
            printf("virvo::serialize: %s\n", e.what());
#else
            static_cast<void>(e);
#endif
            return false;
        }
    }

} // namespace virvo

#endif // !VV_PRIVATE_SERIALIZE_H
