/** @file objectstreambinarystructure.hxx
 * definition of binary file format for object streams.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "objectstreambinarystructure.hxx"  // definition of binary file format for object streams.

#ifndef __objectstreambinary_hxx__
#define __objectstreambinary_hxx__
 
 
// structure of the file
// =====================
// position table:
//      0.. 7:  4 bytes: position of checksum
//      8..15   8 bytes: position of header
//     16..23:  8 bytes: position of footer
//     24..31:  8 bytes: position of actual data
// checksum:
//     32..35:  [checksum]
// header:
//     36ff.    [header]
// actual data:
//     ff.:     [actual data]
// footer:
//     ff.:     [footer]


#define ARCHIVE_TABLE_POS_CHECKSUM 0
#define ARCHIVE_TABLE_POS_HEADER 8
#define ARCHIVE_TABLE_POS_FOOTER 16
#define ARCHIVE_TABLE_POS_DATA 24

#define ARCHIVE_POS_CHECKSUM 32     // a priori known
#define ARCHIVE_POS_HEADER 36       // a priori known

#endif
