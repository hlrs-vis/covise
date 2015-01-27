/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "QuickNavDrawable.h"
#include "ViewPoint.h"

using namespace osg;

void preHUD(int width, int height)
{
    // Save the current state
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // set the orthographic projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, (double)width, (double)height, 0.0, -10, 10);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

void postHUD()
{
    //return to the original projection
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    // Restore the current state
    glPopAttrib();
    glPopClientAttrib();
}

void drawHUD()
{
    if (!ViewPoints::actSharedVPData->isEnabled)
        return;
    //
    // Get data about current viewpoint
    //
    int actVpIndex = ViewPoints::actSharedVPData->index;
    char actVpName[SHARED_VP_NAME_LENGTH];
    int actHasClipPlaneArr[6];

    strncpy(actVpName, ViewPoints::actSharedVPData->name, SHARED_VP_NAME_LENGTH);

    for (int i = 0; i < 6; i++)
        actHasClipPlaneArr[i] = ViewPoints::actSharedVPData->hasClipPlane[i];

    int totNumVp = ViewPoints::actSharedVPData->totNum;

    // setup orthographic projection
    int width, height;
    width = cover->frontWindowHorizontalSize;
    height = cover->frontWindowVerticalSize;
    preHUD(width, height);

    //
    // Draw Bar
    //
    Vec2 lineStart(32.0f, height - 32.0f);
    Vec2 lineEnd(400.0f, height - 32.0f);
    float lineLength = lineEnd[0] - lineStart[0];

    glColor3f(0.7f, 0.7f, 0.7f);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    {
        glVertex2f(lineStart[0], lineStart[1]);
        glVertex2f(lineEnd[0], lineEnd[1]);
    }
    glEnd();

    float cursorHeight = 15.0f;
    float cursorOffset = lineStart[0];

    if (actVpIndex > -1)
        cursorOffset += actVpIndex * lineLength / (totNumVp - 1);

    Vec2 cursorStart(cursorOffset, lineStart[1] - (cursorHeight * 0.5f));
    Vec2 cursorEnd(cursorOffset, lineStart[1] + (cursorHeight * 0.5f));

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    {
        glVertex2f(cursorStart[0], cursorStart[1]);
        glVertex2f(cursorEnd[0], cursorEnd[1]);
    }
    glEnd();

    //
    // Draw ClipPlane status
    //
    Vec2 quadDimension(8, 8);
    Vec2 statusStart(lineEnd[0] - (8.5f * quadDimension[0]), height - ((height - lineStart[1]) / 2.0f) - (quadDimension[1] / 2.0f));

    glBegin(GL_QUADS);
    {
        for (int i = 0; i < 6; i++)
        {
            // choose color
            if (actHasClipPlaneArr[i] == 1)
            {
                // 	      if ((current_vp == activeVP) && useClipPlanesCheck_->getState())
                // 		  glColor4fv(enabledColor.vec);
                // 	      else
                glColor4f(0.6f, 0.6f, 0.6f, 1.0f);
            }
            else
            {
                glColor4f(0.3f, 0.3f, 0.3f, 1.0f);
            }

            Vec2 quadStart(statusStart[0] + i * quadDimension[0] * 1.5f, statusStart[1]);

            glVertex2f(quadStart[0], quadStart[1]);
            glVertex2f(quadStart[0] + quadDimension[0], quadStart[1]);
            glVertex2f(quadStart[0] + quadDimension[0], quadStart[1] + quadDimension[1]);
            glVertex2f(quadStart[0], quadStart[1] + quadDimension[1]);
        }
    }
    glEnd();

    //
    // Draw Text
    //
    //    Vec4 textColor(1, 1, 1, 1);
    //    Vec4 shadowColor(0, 0, 0, 0);
    //    Vec2 textPos(lineStart[0], height-8);

    // convert our coordinate system into pfu convention
    //    textPos[0] = textPos[0] / width;
    //    textPos[1] = (textPos[1] - height) / -height;

    //    pfuDrawMessageRGB(chan, actVpName, 0, PFU_LEFT_JUSTIFIED, textPos[0], textPos[1], PFU_FONT_TINY, textColor, shadowColor);

    postHUD();
}

void QuickNavDrawable::drawImplementation(osg::RenderInfo &) const
{
    drawHUD();
}
