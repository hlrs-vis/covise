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


#ifndef VV_PRIVATE_COMPRESSED_VECTOR_H
#define VV_PRIVATE_COMPRESSED_VECTOR_H


#include <vector>
#include <stddef.h>

#include <boost/serialization/vector.hpp>


namespace virvo
{


// The compression type of a CompressedVector
enum CompressionType
{
    // The buffer is not compressed
    Compress_None,
    // The buffer is compressed using the SNAPPY algorithm
    Compress_Snappy,
    // JPEG compression
    Compress_JPEG,
    // PNG compression (zlib)
    Compress_PNG
};

template<class A>
void serialize(A& a, CompressionType& t, unsigned /*version*/)
{
    a & static_cast<unsigned>(t);
}


// A (not neccessarily) compressed vector.
class CompressedVector
{
public:
    typedef std::vector<unsigned char> VectorType;

    typedef VectorType::value_type          value_type;
    typedef VectorType::reference           reference;
    typedef VectorType::const_reference     const_reference;
    typedef VectorType::pointer             pointer;
    typedef VectorType::const_pointer       const_pointer;
    typedef VectorType::iterator            iterator;
    typedef VectorType::const_iterator      const_iterator;

    // Default constructor.
    // Constructs an empty buffer.
    CompressedVector()
        : data()
        , type(Compress_None)
    {
    }

    // Copy construct from another compressed vector
    CompressedVector(CompressedVector const& rhs)
        : data(rhs.data)
        , type(rhs.type)
    {
    }

    // Copy construct from another vector
    explicit CompressedVector(VectorType const& rhs, CompressionType type = Compress_None)
        : data(rhs)
        , type(type)
    {
    }

    // Construct a buffer of the given size
    explicit CompressedVector(size_t size, CompressionType type = Compress_None)
        : data(size)
        , type(type)
    {
    }

    // Copy construct from a sequence
    template<class Iterator>
    CompressedVector(Iterator first, Iterator last, CompressionType type = Compress_None)
        : data(first, last)
        , type(type)
    {
    }

    // Copy assignment from another compressed vector
    CompressedVector& operator =(CompressedVector const& rhs)
    {
        data = rhs.data;
        type = rhs.type;

        return *this;
    }

    // Returns the underlying buffer
    VectorType& vector()
    {
        return data;
    }

    // Returns the underlying buffer
    VectorType const& vector() const
    {
        return data;
    }

    // Returns a pointer to the underlying buffer
    pointer ptr()
    {
        return &data[0];
    }

    // Returns a pointer to the underlying buffer
    const_pointer ptr() const
    {
        return &data[0];
    }

    // Returns a reference to the n-th element of the buffer
    reference operator [](size_t index)
    {
        return data[index];
    }

    // Returns a reference to the n-th element of the buffer
    const_reference operator [](size_t index) const
    {
        return data[index];
    }

    // Returns whether the underlying buffer is empty
    bool empty() const
    {
        return data.empty();
    }

    // Returns the size of the buffer
    size_t size() const
    {
        return data.size();
    }

    // Returns an iterator to the beginning of the buffer
    iterator begin()
    {
        return data.begin();
    }

    // Returns an iterator to the beginning of the buffer
    const_iterator begin() const
    {
        return data.begin();
    }

    // Returns an iterator past the end of the buffer
    iterator end()
    {
        return data.end();
    }

    // Returns an iterator past the end of the buffer
    const_iterator end() const
    {
        return data.end();
    }

    // Resize the buffer
    void resize(size_t size, CompressionType ctype = Compress_None)
    {
        this->data.resize(size);
        this->type = ctype;
    }

    // Copy from a sequence
    template<class Iterator>
    void assign(Iterator first, Iterator last, CompressionType ctype = Compress_None)
    {
        this->data.assign(first, last);
        this->type = ctype;
    }

    // Returns the current compression type
    CompressionType getCompressionType() const
    {
        return type;
    }

    // Sets the compression type
    void setCompressionType(CompressionType type)
    {
        this->type = type;
    }

    // Swap this buffer with another buffer
    void swap(CompressedVector& rhs)
    {
        using std::swap;

        swap(data, rhs.data);
        swap(type, rhs.type);
    }

    // Swap this buffer with another buffer
    void swap(VectorType& rhs, CompressionType ctype = Compress_None)
    {
        this->data.swap(rhs);
        this->type = ctype;
    }

private:
    // The data (not neccessarily compressed...)
    VectorType data;
    // The compression type
    CompressionType type;

public:
    template<class A>
    void serialize(A& a, unsigned/*version*/)
    {
        a & data;
        a & type;
    }
};


// Swap two compressed vectors
inline void swap(CompressedVector& lhs, CompressedVector& rhs)
{
    lhs.swap(rhs);
}


} // namespace virvo


#endif
