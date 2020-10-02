#pragma once

/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2020 HLRS  **
 **                                                                          **
 ** Description: Camera position and orientation optimization                **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April 2020  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <memory>

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include "UDP.h"

#define SHOW_UDP_LIVE_OBJECTS 0 // if 1 use udp to visualize livedata

class Orientation;
enum class FitnessFunctionType;

//Free functions
int calcNumberOfSensors();
int convert(int sensorPos);

void calcVisibility();
void optimize(FitnessFunctionType);

void updateAllSensors(std::vector<Orientation>);

class UI;
class SensorPlacementPlugin :public opencover::coVRPlugin 
{
public:
  SensorPlacementPlugin();
  ~SensorPlacementPlugin();
  bool init() override;
  bool update() override;
  void preFrame() override;
  bool destroy() override;


private:
  std::unique_ptr<UI> m_UI;
  std::unique_ptr<UDP> m_udp;
};
