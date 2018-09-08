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

//============================================================================
// Legacy serialization functions -- DEPRECATED
//============================================================================


inline uint8_t virvo::serialization::read8(uint8_t* src)
{
  uint8_t val = 0;
  read(src, &val);
  return val;
}

inline uint16_t virvo::serialization::read16(uint8_t* src, virvo::serialization::EndianType end)
{
  uint16_t val = 0;
  read(src, &val, end);
  return val;
}

inline uint32_t virvo::serialization::read32(uint8_t* src, virvo::serialization::EndianType end)
{
  uint32_t val = 0;
  read(src, &val, end);
  return val;
}

inline uint64_t virvo::serialization::read64(uint8_t* src, virvo::serialization::EndianType end)
{
  uint64_t val = 0;
  read(src, &val, end);
  return val;
}

inline float virvo::serialization::readFloat(uint8_t* src, virvo::serialization::EndianType end)
{
  float val = 0.0f;
  read(src, &val, end);
  return val;
}

inline uint8_t virvo::serialization::read8(FILE* src)
{
  uint8_t val = 0;
  read(src, &val);
  return val;
}

inline uint16_t virvo::serialization::read16(FILE* src, virvo::serialization::EndianType end)
{
  uint16_t val = 0;
  read(src, &val, end);
  return val;
}

inline uint32_t virvo::serialization::read32(FILE* src, virvo::serialization::EndianType end)
{
  uint32_t val = 0;
  read(src, &val, end);
  return val;
}

inline uint64_t virvo::serialization::read64(FILE* src, virvo::serialization::EndianType end)
{
  uint64_t val = 0;
  read(src, &val, end);
  return val;
}

inline float virvo::serialization::readFloat(FILE* src, virvo::serialization::EndianType end)
{
  float val = 0;
  read(src, &val, end);
  return val;
}

inline uint32_t virvo::serialization::read32(std::ifstream& src, virvo::serialization::EndianType end)
{
  uint32_t val = 0;
  read(src, &val, end);
  return val;
}

inline uint64_t virvo::serialization::read64(std::ifstream& src, virvo::serialization::EndianType end)
{
  uint64_t val = 0;
  read(src, &val, end);
  return val;
}

inline size_t virvo::serialization::write8(uint8_t* dst, uint8_t val)
{
  return write(dst, val);
}

inline size_t virvo::serialization::write16(uint8_t* dst, uint16_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::write32(uint8_t* dst, uint32_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::write64(uint8_t* dst, uint64_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::writeFloat(uint8_t* dst, float val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::write8(FILE* dst, uint8_t val)
{
  return write(dst, val);
}

inline size_t virvo::serialization::write16(FILE* dst, uint16_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::write32(FILE* dst, uint32_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::write64(FILE* dst, uint64_t val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

inline size_t virvo::serialization::writeFloat(FILE* dst, float val, virvo::serialization::EndianType end)
{
  return write(dst, val, end);
}

