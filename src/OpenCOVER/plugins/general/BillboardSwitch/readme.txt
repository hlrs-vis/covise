/****************************************************************************\ 
 **                                                            (C)2007 ZAIK  **
 **                                                                          **
 ** Description:  BillboardSwitch Plugin                                     **
 **        implements a new VRML Node Type BillboardSwitch.                  **
 **        it is a mashup between / and based upon                           **
 **                the Billboard and Switch Nodes.                           **
 **                                                                          **
 **        Based on the angle under which the BillboardSwitch is watched,    **
 **        the node switches between its childs and billboards them.         **
 **                                                                          **
 **                         structure for BillboardSwitch node:              **
 **                 BillboardSwitch {                                        **
 **                     exposedField   SFVec3f axisOfRotation                **
 **                     field          MFFloat angle  []                     **
 **                     eventOut       MFInt   activeChildChanged            **
 **                     exposedFields  MFNode  choice  []                    **
 **                     exposedFields  MFNode  alternative  []               **
 **                 }                                                        **
 **                                                                          **
 **        with axisOfRotation like Axis in Billboard Node                   **
 **             angle are the angles under which the childs are switched     **
 **             activeChildChanged indicates if the active Child changed     **
 **             choice are the different childs which are switched           **
 **                                     and billboarded                      **
 **             alternative is a workaround for other VRML Browsers          **
 **                         with the PROTO definition, the BillboardSwitch   **
 **                         works like a normal Billboard with alternative   **
 **                         as its childs.                                   **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: Hauke Fuehres                                                    **
 **                                                                          **
 \****************************************************************************/


structure for BillboardSwitch node:
                 BillboardSwitch {
                    exposedField   SFVec3f axisOfRotation
                    field          MFFloat angle  []
                    eventOut       MFInt   activeChildChanged
                    exposedFields  MFNode  choice  []
                }



For compatibility with other VRML Browsers use the following PROTO definition in VRML files with a BillboardSwitch. It uses a simple Billboard with the alternative as its children. 

PROTO BillboardSwitch [
       exposedField   SFVec3f axisOfRotation 0 1 0
       field          MFFloat angle 0.0
       eventOut       MFInt32   activeChildChanged
       exposedField  MFNode  choice[]
       exposedField  MFNode  alternative[]
 ]
{
  Billboard {
  axisOfRotation IS axisOfRotation
  children IS alternative
  }
}
