/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UDP_MESSAGE_TYPES_H
#define UDP_MESSAGE_TYPES_H

#include <util/coExport.h>
#include <string>
#include <vector>


/***********************************************************************\
 **                                                                     **
 **    UDP Message classes                              Version: 1.1    **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic udp message structure		            **
 **                                                                     **
 **   Classes      : Message, ShmMessage                                **
 **                                                                     **
 **												                        **
 **                                                                     **
 **   Author       : D. Grieger (HLRS)                                  **
 **                                                                     **
 **   History      :  24.05.2019 created								**
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

enum udp_msg_type : int
{
	EMPTY = 0,
	AVATAR_HMD_POSITION ,
	AVATAR_CONTROLLER_POSITION,
	AUDIO_STREAM,
	MIDI_STREAM,
};


//NETEXPORT extern std::vector<std::string> udp_msg_types_vector = {
//	"EMPTY",
//	"MESSAGE_AVATAR_HMD_POSITION",
//	"MESSAGE_AVATAR_CONTROLLER_POSITION"
//
//};
}
#endif
