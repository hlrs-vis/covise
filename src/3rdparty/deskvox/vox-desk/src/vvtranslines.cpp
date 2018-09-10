// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, schulze@cs.brown.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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

#include "vvtranslinesFX.h"

const static int clickError = 2;
const static FXColor selectedColor = FXRGB(0,0,0);
const static FXColor deselectedColor = FXRGB(180,180,180);

/**************************************************/

VVTransferLine::VVTransferLine()
{
  xPos = 0;
  width = 0;
  angle = 0;
  max = 0;
  selected = false;
}

VVTransferLine::~VVTransferLine()
{
}

bool
VVTransferLine::clicked(int x)
{
  if(xPos - clickError < x && xPos + clickError > x) return true;

  return false;
}

/**************************************************/

/**************************************************/

VVTransferHat::VVTransferHat():VVTransferLine()
{

}

VVTransferHat::~VVTransferHat()
{

}

void
VVTransferHat::draw(FXCanvas* canvas)
{
  int cWidth = canvas->getWidth();
  int cHeight = canvas->getHeight();

  int h = (int)(max*cHeight);
  int w = (int)(.5*width*cWidth);

  FXDCWindow dc(canvas);

  // Set foreground color
  dc.setForeground(selectedColor);

  // Draw line
  dc.drawLine(xPos-w, (int)((1.0-max)*cHeight)-1, xPos+w, (int)((1.0-max)*cHeight)-1);
  if(angle > 0 && angle < 90.0)
  {
    double a = 3.14159265*angle/180.0;
    dc.drawLine(xPos-w-(int)(h*tanf(a)),cHeight, xPos-w, (int)((1.0-max)*cHeight)-1);
    dc.drawLine(xPos+w+(int)(h*tanf(a)),cHeight, xPos+w, (int)((1.0-max)*cHeight)-1);
  }
  else if(angle == 0)
  {
    dc.drawLine(0, (int)((1.0-max)*cHeight)-1, cWidth, (int)((1.0-max)*cHeight)-1);
  }
  else
  {
    dc.drawLine(xPos-w, (int)((1.0-max)*cHeight)-1, xPos-w, cHeight);
    dc.drawLine(xPos+w, (int)((1.0-max)*cHeight)-1, xPos+w, cHeight);
  }

  if(!selected)
  {
    dc.setForeground(deselectedColor);
    dc.drawLine(xPos,0,xPos,cHeight);
  }
  else
  {
    dc.setLineWidth(2);
    dc.drawLine(xPos-2,0,xPos-2,cHeight);
    dc.drawLine(xPos+2,0,xPos+2,cHeight);
  }
}

/**************************************************/

/**************************************************/

VVTransferRamp::VVTransferRamp():VVTransferLine()
{

}

VVTransferRamp::~VVTransferRamp()
{

}

void
VVTransferRamp::draw(FXCanvas* canvas)
{
  int cWidth = canvas->getWidth();
  int cHeight = canvas->getHeight();

  int h = (int)(max*cHeight);

  FXDCWindow dc(canvas);

  // Set foreground color
  dc.setForeground(selectedColor);

  // Draw line
  if(angle != 0 && angle < 90.0 && angle > -90.0)
  {
    double a = 3.14159265*angle/180.0;
    int offset = (int)((h/2)*tanf(a));
    dc.drawLine(xPos-offset,cHeight, xPos+offset, (int)((1.0-max)*cHeight));
    if(angle < 0)
    {
      dc.drawLine(xPos+offset,(int)((1.0-max)*cHeight), 0, (int)((1.0-max)*cHeight));
    }
    else
    {
      dc.drawLine(xPos+offset,(int)((1.0-max)*cHeight), cWidth, (int)((1.0-max)*cHeight));
    }
  }
  else
  {
    dc.drawLine(0, (int)((1.0-max)*cHeight), cWidth, (int)((1.0-max)*cHeight));
  }

  if(!selected)
  {
    dc.setForeground(deselectedColor);
    dc.drawLine(xPos,0,xPos,cHeight);
  }
  else
  {
    dc.setLineWidth(2);
    dc.drawLine(xPos-2,0,xPos-2,cHeight);
    dc.drawLine(xPos+2,0,xPos+2,cHeight);
  }
}

/**************************************************/

/**************************************************/

VVTransferBlank::VVTransferBlank():VVTransferLine()
{

}

VVTransferBlank::~VVTransferBlank()
{

}

void
VVTransferBlank::draw(FXCanvas* canvas)
{
  int cWidth = canvas->getWidth();
  int cHeight = canvas->getHeight();

  int h = (int)(max*cHeight);
  int w = (int)(.5*width*cWidth);

  FXDCWindow dc(canvas);

  // Set foreground color
  dc.setForeground(selectedColor);

  dc.drawRectangle(xPos-w, 1, w*2, cHeight-2);

  if(!selected)
  {
    dc.setForeground(deselectedColor);
    dc.drawLine(xPos,0,xPos,cHeight);
  }
  else
  {
    dc.setLineWidth(2);
    dc.drawLine(xPos-2,0,xPos-2,cHeight);
    dc.drawLine(xPos+2,0,xPos+2,cHeight);
  }
}

/**************************************************/
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
