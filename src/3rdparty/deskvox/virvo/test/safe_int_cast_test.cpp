// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "safe_int_cast.h"

#include <limits>
#include <sstream>
#include <string>

#include <boost/cstdint.hpp>

#include <gtest/gtest.h>

using namespace virvo;
using namespace virvo::details;

template <class T>
struct Limits : public std::numeric_limits<typename boost::remove_cv<T>::type>
{
};

TEST(SafeIntCast, ToBool)
{
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<     bool >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<     bool >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<   int8_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<   int8_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  uint8_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  uint8_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int16_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int16_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint16_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint16_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int32_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int32_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint32_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint32_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int64_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits<  int64_t >::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint64_t >::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<bool>( Limits< uint64_t >::max() ) );
}

TEST(SafeIntCast, ToInt8)
{
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<          bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<      int8_t>( Limits<const     bool>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<        int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<      int8_t>( Limits<const   int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<       uint8_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const  uint8_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int8_t>( Limits<       int16_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const  int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<      uint16_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int8_t>( Limits<       int32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const  int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<      uint32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int8_t>( Limits<       int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const  int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int8_t>( Limits<      uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<      int8_t>( Limits<const uint64_t>::max() ) );
}

TEST(SafeIntCast, ToUInt8)
{
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const     bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const     bool>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const   int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const   int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const  uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const  uint8_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int16_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const uint16_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const uint32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const  int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const uint8_t>( Limits<const uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const uint8_t>( Limits<const uint64_t>::max() ) );
}

TEST(SafeIntCast, ToInt16)
{
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<    bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<    bool>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<  int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<  int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits< uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits< uint8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits< int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits< int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<uint16_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits<uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits< int32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits< int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<uint32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits<uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits< int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits< int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<const int16_t>( Limits<uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<const int16_t>( Limits<uint64_t>::max() ) );
}

TEST(SafeIntCast, ToUInt16)
{
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const     bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const     bool>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const   int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const   int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const  uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const  uint8_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const  int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const  int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const uint16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const  int32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const  int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const uint32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const  int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const  int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint16_t>( Limits<const uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint16_t>( Limits<const uint64_t>::max() ) );
}

TEST(SafeIntCast, ToInt32)
{
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<    bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<    bool>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<  int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<  int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< uint8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<uint16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<uint16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< int32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits< int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<uint32_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable< int32_t>( Limits<uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable< int32_t>( Limits< int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable< int32_t>( Limits< int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int32_t>( Limits<uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable< int32_t>( Limits<uint64_t>::max() ) );
}

TEST(SafeIntCast, ToUInt32)
{
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<    bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<    bool>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits<  int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<  int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits< uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits< uint8_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits< int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits< int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<uint16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits< int32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits< int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<uint32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits< int64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits< int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint32_t>( Limits<uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint32_t>( Limits<uint64_t>::max() ) );
}

TEST(SafeIntCast, ToInt64)
{
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<    bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<    bool>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<  int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<  int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< uint8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<uint16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<uint16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<uint32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<uint32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int64_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits< int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable< int64_t>( Limits<uint64_t>::min() ) );
    EXPECT_FALSE( IsSafeIntCastable< int64_t>( Limits<uint64_t>::max() ) );
}

TEST(SafeIntCast, ToUInt64)
{
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<    bool>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<    bool>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint64_t>( Limits<  int8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<  int8_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits< uint8_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits< uint8_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint64_t>( Limits< int16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits< int16_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint16_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint16_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint64_t>( Limits< int32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits< int32_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint32_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint32_t>::max() ) );
    EXPECT_FALSE( IsSafeIntCastable<uint64_t>( Limits< int64_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits< int64_t>::max() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint64_t>::min() ) );
    EXPECT_TRUE ( IsSafeIntCastable<uint64_t>( Limits<uint64_t>::max() ) );
}
