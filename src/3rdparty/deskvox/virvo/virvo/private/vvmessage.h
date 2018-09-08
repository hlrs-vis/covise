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

#ifndef VV_PRIVATE_MESSAGE_H
#define VV_PRIVATE_MESSAGE_H

#include "vvexport.h"
#include "vvserialize.h"

#include <boost/function.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/uuid/uuid.hpp>

#include <cassert>
#include <stdexcept>

namespace virvo
{

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------

    class Message
    {
        friend class Connection;
        friend class ConnectionManager;

        typedef std::vector<char> DataType;

        struct Header
        {
            // The unique ID of this message
            boost::uuids::uuid id_; // POD, 16 bytes
            // The type of this message
            unsigned type_;
            // The length of this message
            unsigned size_;

            Header(boost::uuids::uuid const& id, unsigned type, unsigned size)
                : id_(id)
                , type_(type)
                , size_(size)
            {
            }
        };

        // The message data
        DataType data_;
        // The message header
        Header header_;

    private:
        // Creates a new unique ID for this message
        VVAPI static boost::uuids::uuid GenerateID();

    public:
        enum Type {
    //--- Client -> Server
            CameraMatrix,       // type = virvo::messages::CameraMatrix
            CurrentFrame,
            Disconnect,
            GpuInfo,
            ObjectDirection,    // type = vvVector3
            Parameter,          // type = virvo::messages::Param
            Position,           // type = vvVector3
            RemoteServerType,   // type = vvRenderer::RendererType
            ServerInfo,
            Statistics,
            TransFunc,          // type = vvTransFunc
            TransFuncChanged,   // type = bool
            ViewingDirection,   // type = vvVector3
            Volume,             // type = vvVolDesc
            VolumeFile,         // type = std::string
            WindowResize,       // type = virvo::messages::WindowResize
    //--- Server -> Client
            Image,              // type = virvo::Image
            IbrImage,           // type = virvo::IbrImage

            LastType // <-- MUST be the last item!
        };

    public:
        VVAPI explicit Message(unsigned type = 0);

        // Creates a message from the given buffer. NOT SERIALIZED!
        template<class InputIterator>
        explicit Message(unsigned type, InputIterator first, InputIterator last);

        // Creates a serialized message
        template<class T>
        explicit Message(unsigned type, T const& object);

        // Serialize the given object.
        // Returns true on success, false otherwise.
        template<class T>
        bool serialize(T const& object);

        // Deserialize the message.
        // Returns true on success, false otherwise.
        template<class T>
        bool deserialize(T& object) const;

        // Deserialize the message.
        // Throws std::runtime_error on failure.
        template<class T>
        T deserialize() const;

        template<class T>
        bool reset(unsigned type, T const& object);

        // Returns the unique ID of this message
        boost::uuids::uuid const& id() const {
            return header_.id_;
        }

        // Returns the type of this message
        unsigned type() const {
            return header_.type_;
        }

        // Returns the size of the message
        unsigned size() const
        {
            assert( header_.size_ == data_.size() );
            return static_cast<unsigned>(data_.size());
        }

        // Returns an iterator to the first element of the data
        DataType::iterator begin() {
            return data_.begin();
        }

        // Returns an iterator to the element following the last element of the data
        DataType::iterator end() {
            return data_.end();
        }

        // Returns an iterator to the first element of the data
        DataType::const_iterator begin() const {
            return data_.begin();
        }

        // Returns an iterator to the element following the last element of the data
        DataType::const_iterator end() const {
            return data_.end();
        }
    };

    template<class InputIterator>
    Message::Message(unsigned type, InputIterator first, InputIterator last)
        : data_(first, last)
        , header_(GenerateID(), type, static_cast<unsigned>(data_.size()))
    {
    }

    template<class T>
    Message::Message(unsigned type, T const& object)
        : data_()
        , header_(GenerateID(), type, 0)
    {
        if (!serialize(object))
        {
            throw std::runtime_error("serialization error");
        }
    }

    template<class T>
    bool Message::serialize(T const& object)
    {
        data_.clear();

        if (::virvo::serialize(data_, object))
        {
            // Set the size of the serialized message
            header_.size_ = static_cast<unsigned>(data_.size());
            return true;
        }

        return false;
    }

    template<class T>
    bool Message::deserialize(T& object) const
    {
        assert( header_.size_ == data_.size() );

        return ::virvo::deserialize(object, data_);
    }

    template<class T>
    T Message::deserialize() const
    {
        T object;

        if (!deserialize(object))
            throw std::runtime_error("deserialization error");

        return object;
    }

    template<class T>
    bool Message::reset(unsigned type, T const& object)
    {
        if (serialize(object))
        {
            header_.type_ = type;
            return true;
        }

        return false;
    }

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------

    typedef boost::shared_ptr<Message> MessagePointer;

    inline MessagePointer makeMessage(unsigned type = 0)
    {
        return boost::make_shared<Message>(type);
    }

    // Creates a message from the given buffer. NOT SERIALIZED!
    template<class InputIterator>
    MessagePointer makeMessage(unsigned type, InputIterator first, InputIterator last)
    {
        return boost::make_shared<Message>(type, first, last);
    }

    // Creates a serialized message
    template<class T>
    MessagePointer makeMessage(unsigned type, T const& object)
    {
        return boost::make_shared<Message>(type, object);
    }

} // namespace virvo

#endif // !VV_PRIVATE_MESSAGE_H
