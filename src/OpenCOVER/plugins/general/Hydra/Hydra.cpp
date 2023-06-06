/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Tracker Plugin for Razer Hydra                              **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** dec-11  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Hydra.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <sixense.h>
namespace std
{
__declspec(dllexport) void _Xlength_error(char const *){};
__declspec(dllexport) void _Xout_of_range(char const *){};
/*class locale::facet
   {
       void _Facet_Register(class std::locale::facet *);
   }
   __declspec(dllexport) void locale::facet::_Facet_Register(class std::locale::facet *){};*/

/*1>sixense_s_x64.lib(DeSawtoothFilter.obj) : error LNK2019: unresolved external symbol "__declspec(dllimport) public: __cdecl std::_Container_base12::~_Container_base12(void)" (__imp_??1_Container_base12@std@@QEAA@XZ) referenced in function "int `public: __cdecl std::_Deque_val<class sixenseMath::Vector3,class std::allocator<class sixenseMath::Vector3> >::_Deque_val<class sixenseMath::Vector3,class std::allocator<class sixenseMath::Vector3> >(class _Deque_val<class sixenseMath::Vector3,class std::allocator<class sixenseMath::Vector3> >::allocator<class sixenseMath::Vector3>)'::`1'::dtor$0" (?dtor$0@?0???0?$_Deque_val@VVector3@sixenseMath@@V?$allocator@VVector3@sixenseMath@@@std@@@std@@QEAA@V?$allocator@VVector3@sixenseMath@@@1@@Z@4HA)
1>sixense_s_x64.lib(PacketStreamManager.obj) : error LNK2001: unresolved external symbol "__declspec(dllimport) public: __cdecl std::_Container_base12::~_Container_base12(void)" (__imp_??1_Container_base12@std@@QEAA@XZ)
1>sixense_s_x64.lib(ControllerDataStream.obj) : error LNK2001: unresolved external symbol "__declspec(dllimport) public: __cdecl std::_Container_base12::~_Container_base12(void)" (__imp_??1_Container_base12@std@@QEAA@XZ)
1>sixense_s_x64.lib(LogFileParser.obj) : error LNK2019: unresolved external symbol "__declspec(dllimport) public: void __cdecl std::_Container_base0::_Swap_all(struct std::_Container_base0 &)" (__imp_?_Swap_all@_Container_base0@std@@QEAAXAEAU12@@Z) referenced in function "public: void __cdecl LogFileParser::load(void)" (?load@LogFileParser@@QEAAXXZ)
1>sixense_s_x64.lib(USBManagerWin32.obj) : error LNK2001: unresolved external symbol "__declspec(dllimport) public: void __cdecl std::_Container_base0::_Swap_all(struct std::_Container_base0 &)" (__imp_?_Swap_all@_Container_base0@std@@QEAAXAEAU12@@Z)
1>sixense_s_x64.lib(LogFileParser.obj) : error LNK2019: unresolved external symbol "__declspec(dllimport) public: void __cdecl std::_Container_base0::_Orphan_all(void)" (__imp_?_Orphan_all@_Container_base0@std@@QEAAXXZ) referenced in function "public: void __cdecl LogFileParser::load(void)" (?load@LogFileParser@@QEAAXXZ)
1>sixense_s_x64.lib(USBManagerWin32.obj) : error LNK2001: unresolved external symbol "__declspec(dllimport) public: void __cdecl std::_Container_base0::_Orphan_all(void)" (__imp_?_Orphan_all@_Container_base0@std@@QEAAXXZ)
1>C:\src\covise\angusopt\lib\OpenCOVER\plugins\Hydra.dll : fatal error LNK1120: 5 unresolved externals
*/
}

Hydra::Hydra()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Hydra::Hydra\n");

    int init = sixenseInit();
    int activebase = sixenseSetActiveBase(0);
    int basecolor = sixenseSetBaseColor(255, 0, 0);
    int reshigh = sixenseSetHighPriorityBindingEnabled(1);
    int autoenable = sixenseAutoEnableHemisphereTracking(1);
    frame = 0;
}

// this is called if the plugin is removed at runtime
Hydra::~Hydra()
{
    fprintf(stderr, "Hydra::~Hydra\n");
    sixenseExit();
}

void Hydra::preFrame()
{
    sixenseGetAllNewestData(&acd);

    for (unsigned int i = 0; i < 1; i++)
    {

        fprintf(stderr, "VRC %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ %d %d]",
                acd.controllers[i].buttons, acd.controllers[i].pos[0], acd.controllers[i].pos[1], acd.controllers[i].pos[2],
                acd.controllers[i].rot_mat[0][0], acd.controllers[i].rot_mat[0][1], acd.controllers[i].rot_mat[0][2],
                acd.controllers[i].rot_mat[1][0], acd.controllers[i].rot_mat[1][1], acd.controllers[i].rot_mat[1][2],
                acd.controllers[i].rot_mat[2][0], acd.controllers[i].rot_mat[2][1], acd.controllers[i].rot_mat[2][2],
                acd.controllers[i].joystick_x, acd.controllers[i].joystick_y);
    }
}

unsigned int Hydra::button(int station)
{
    return acd.controllers[station].buttons;
}

COVERPLUGIN(Hydra)
