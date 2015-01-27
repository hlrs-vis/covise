/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CMS_Struct_h
#define __CMS_Struct_h

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Xenomai Plugin (does nothing)                               *.
 **                                                                          **
 **                                                                          **
 ** Author: S. Franz		                                             **
 **                                                                          **
 ** History:  								     **
 ** Nov-01  v1	    				       		             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

struct CANMsgIface
{
    virtual ~CANMsgIface(){};
    //virtual const can_frame& getFrame()=0 const; // const = schreibschutz für zurückgegebenes objekt
    virtual can_frame &theCANframe() = 0; // const = schreibschutz für zurückgegebenes objekt
    //virtual void setFrame(can_frame newframe)=0;

    uint32_t cycle_time;
};

// struct CANMsgIface2
// {
//    virtual ~CANMsgIface2(){};
//
//    union a
//    {
//       can_frame frame;
//
//       struct b
//       {
//          uint32_t ID;
//          uint8_t DLC;
//
//          struct c
//          {
//          } cansignals;
// ;
//       } canmsg;
//
//    } values;
//
//    uint32_t cycle_time;
//
//    const can_frame& getFrame() // const = schreibschutz für zurückgegebenes objekt
//    {
//       return values.frame;
//    }
//
//    const a::b& getCANmsg() {
//       return values.canmsg;
//    }
//
//    const a::b::c& getCANsignals() {
//       return values.canmsg.cansignals;
//    }
// };

#endif
