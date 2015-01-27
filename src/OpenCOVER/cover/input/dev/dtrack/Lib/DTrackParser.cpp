/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* DTrackParser: C++ source file, A.R.T. GmbH 17.6.13
 *
 * DTrackParser: functions to process DTrack UDP packets (ASCII protocol)
 * Copyright (C) 2013, Advanced Realtime Tracking GmbH
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 * Version v2.4.0
 *
 * Purpose:
 *  - DTrack2 network protocol due to: 'Technical Appendix DTrack v2.0'
 *  - for ARTtrack Controller versions v0.2 (and compatible versions)
 *  - tested under Linux (gcc) and MS Windows 2000/XP (MS Visual C++)
 *
 */

#include "DTrackParser.hpp"

/**
 * 	\brief	Constructor.
 */
DTrackParser::DTrackParser()
{
    // reset actual DTrack data:
    act_framecounter = 0;
    act_timestamp = -1;

    act_num_body = act_num_flystick = act_num_meatool = act_num_mearef = act_num_hand = act_num_human = 0;
    act_num_inertial = 0;
    act_num_marker = 0;
}

/**
 * 	\brief Destructor.
 */
DTrackParser::~DTrackParser()
{
    //
}

/**
 * 	\brief Set default values at start of a new frame.
 */
void DTrackParser::startFrame()
{
    act_framecounter = 0;
    act_timestamp = -1; // i.e. not available
    loc_num_bodycal = loc_num_handcal = -1; // i.e. not available
    loc_num_flystick1 = loc_num_meatool = 0;
}

/**
 * 	\brief Final adjustments after processing all data for a frame.
 */
void DTrackParser::endFrame()
{
    int j, n;

    // set number of calibrated standard bodies, if necessary:
    if (loc_num_bodycal >= 0)
    { // '6dcal' information was available
        n = loc_num_bodycal - loc_num_flystick1 - loc_num_meatool;
        if (n > act_num_body)
        { // adjust length of vector
            act_body.resize(n);
            for (j = act_num_body; j < n; j++)
            {
                memset(&act_body[j], 0, sizeof(DTrack_Body_Type_d));
                act_body[j].id = j;
                act_body[j].quality = -1;
            }
        }
        act_num_body = n;
    }

    // set number of calibrated Fingertracking hands, if necessary:
    if (loc_num_handcal >= 0)
    { // 'glcal' information was available
        if (loc_num_handcal > act_num_hand)
        { // adjust length of vector
            act_hand.resize(loc_num_handcal);
            for (j = act_num_hand; j < loc_num_handcal; j++)
            {
                memset(&act_hand[j], 0, sizeof(DTrack_Hand_Type_d));
                act_hand[j].id = j;
                act_hand[j].quality = -1;
            }
        }
        act_num_hand = loc_num_handcal;
    }
}

/**
 * 	\brief Parses a single line of data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line	one line of data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine(char **line)
{
    if (!line)
        return false;

    // line of frame counter:
    if (!strncmp(*line, "fr ", 3))
    {
        *line += 3;
        return parseLine_fr(line);
    }

    // line of timestamp:
    if (!strncmp(*line, "ts ", 3))
    {
        *line += 3;
        return parseLine_ts(line);
    }

    // line of additional inofmation about number of calibrated bodies:
    if (!strncmp(*line, "6dcal ", 6))
    {
        *line += 6;
        return parseLine_6dcal(line);
    }

    // line of standard body data:
    if (!strncmp(*line, "6d ", 3))
    {
        *line += 3;
        return parseLine_6d(line);
    }

    // line of Flystick data (older format):
    if (!strncmp(*line, "6df ", 4))
    {
        *line += 4;
        return parseLine_6df(line);
    }

    // line of Flystick data (newer format):
    if (!strncmp(*line, "6df2 ", 5))
    {
        *line += 5;
        return parseLine_6df2(line);
    }

    // line of measurement tool data (older format):
    if (!strncmp(*line, "6dmt ", 5))
    {
        *line += 5;
        return parseLine_6dmt(line);
    }

    // line of measurement tool data (newer format):
    if (!strncmp(*line, "6dmt2 ", 6))
    {
        *line += 6;
        return parseLine_6dmt2(line);
    }

    // line of measurement reference data:
    if (!strncmp(*line, "6dmtr ", 6))
    {
        *line += 6;
        return parseLine_6dmtr(line);
    }

    // line of additional inofmation about number of calibrated Fingertracking hands:
    if (!strncmp(*line, "glcal ", 6))
    {
        *line += 6;
        return parseLine_glcal(line);
    }

    // line of A.R.T. Fingertracking hand data:
    if (!strncmp(*line, "gl ", 3))
    {
        *line += 3;
        return parseLine_gl(line);
    }

    // line of 6dj human model data:
    if (!strncmp(*line, "6dj ", 4))
    {
        *line += 4;
        return parseLine_6dj(line);
    }

    // line of 6di inertial data:
    if (!strncmp(*line, "6di ", 4))
    {
        *line += 4;
        return parseLine_6di(line);
    }

    // line of single marker data:
    if (!strncmp(*line, "3d ", 3))
    {
        *line += 3;
        return parseLine_3d(line);
    }

    return true; // ignore unknown line identifiers (could be valid in future DTracks)
}

/**
 * 	\brief Parses a single line of frame counter data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of 'fr' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_fr(char **line)
{
    if (!(*line = string_get_ui(*line, &act_framecounter)))
    {
        act_framecounter = 0;
        return false;
    }

    return true;
}

/**
 * 	\brief Parses a single line of timestamp data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of 'ts' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_ts(char **line)
{
    if (!(*line = string_get_d(*line, &act_timestamp)))
    {
        act_timestamp = -1;
        return false;
    }

    return true;
}

/**
 * 	\brief Parses a single line of additional information about number of calibrated bodies in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6dcal' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6dcal(char **line)
{
    if (!(*line = string_get_i(*line, &loc_num_bodycal)))
    {
        return false;
    }

    return true;
}

/**
 * 	\brief Parses a single line of standard body data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6d' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6d(char **line)
{
    int i, j, n, id;
    double d;

    // disable all existing data
    for (i = 0; i < act_num_body; i++)
    {
        memset(&act_body[i], 0, sizeof(DTrack_Body_Type_d));
        act_body[i].id = i;
        act_body[i].quality = -1;
    }
    // get number of standard bodies (in line)
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // get data of standard bodies
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "id", &id, NULL, &d)))
        {
            return false;
        }
        // adjust length of vector
        if (id >= act_num_body)
        {
            act_body.resize(id + 1);
            for (j = act_num_body; j <= id; j++)
            {
                memset(&act_body[j], 0, sizeof(DTrack_Body_Type_d));
                act_body[j].id = j;
                act_body[j].quality = -1;
            }
            act_num_body = id + 1;
        }
        act_body[id].id = id;
        act_body[id].quality = d;
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_body[id].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_body[id].rot)))
        {
            return false;
        }
    }
    return true;
}

/**
 * 	\brief Parses a single line of Flystick data (older format) data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6df' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6df(char **line)
{
    int i, j, k, n, iarr[2];
    double d;

    // get number of calibrated Flysticks
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    loc_num_flystick1 = n;
    // adjust length of vector
    if (n != act_num_flystick)
    {
        act_flystick.resize(n);
        act_num_flystick = n;
    }
    // get data of Flysticks
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "idi", iarr, NULL, &d)))
        {
            return false;
        }
        if (iarr[0] != i)
        { // not expected
            return false;
        }
        act_flystick[i].id = iarr[0];
        act_flystick[i].quality = d;
        act_flystick[i].num_button = 8;
        k = iarr[1];
        for (j = 0; j < 8; j++)
        {
            act_flystick[i].button[j] = k & 0x01;
            k >>= 1;
        }
        act_flystick[i].num_joystick = 2; // additionally to buttons 5-8
        if (iarr[1] & 0x20)
        {
            act_flystick[i].joystick[0] = -1;
        }
        else if (iarr[1] & 0x80)
        {
            act_flystick[i].joystick[0] = 1;
        }
        else
        {
            act_flystick[i].joystick[0] = 0;
        }
        if (iarr[1] & 0x10)
        {
            act_flystick[i].joystick[1] = -1;
        }
        else if (iarr[1] & 0x40)
        {
            act_flystick[i].joystick[1] = 1;
        }
        else
        {
            act_flystick[i].joystick[1] = 0;
        }
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_flystick[i].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_flystick[i].rot)))
        {
            return false;
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of Flystick data (newer format) data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6df2' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6df2(char **line)
{
    int i, j, k, l, n, iarr[3];
    double d;
    char sfmt[20];

    // get number of calibrated Flysticks
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // adjust length of vector
    if (n != act_num_flystick)
    {
        act_flystick.resize(n);
        act_num_flystick = n;
    }
    // get number of Flysticks
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // get data of Flysticks
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "idii", iarr, NULL, &d)))
        {
            return false;
        }
        if (iarr[0] != i)
        { // not expected
            return false;
        }
        act_flystick[i].id = iarr[0];
        act_flystick[i].quality = d;
        if ((iarr[1] > DTRACK_FLYSTICK_MAX_BUTTON) || (iarr[1] > DTRACK_FLYSTICK_MAX_JOYSTICK))
        {
            return false;
        }
        act_flystick[i].num_button = iarr[1];
        act_flystick[i].num_joystick = iarr[2];
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_flystick[i].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_flystick[i].rot)))
        {
            return false;
        }
        strcpy(sfmt, "");
        j = 0;
        while (j < act_flystick[i].num_button)
        {
            strcat(sfmt, "i");
            j += 32;
        }
        j = 0;
        while (j < act_flystick[i].num_joystick)
        {
            strcat(sfmt, "d");
            j++;
        }
        if (!(*line = string_get_block(*line, sfmt, iarr, NULL, act_flystick[i].joystick)))
        {
            return false;
        }
        k = l = 0;
        for (j = 0; j < act_flystick[i].num_button; j++)
        {
            act_flystick[i].button[j] = iarr[k] & 0x01;
            iarr[k] >>= 1;
            l++;
            if (l == 32)
            {
                k++;
                l = 0;
            }
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of measurement tool data (older format) in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6dmt' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6dmt(char **line)
{
    int i, j, k, n, iarr[3];
    double d;

    // get number of calibrated measurement tools
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    loc_num_meatool = n;
    // adjust length of vector
    if (n != act_num_meatool)
    {
        act_meatool.resize(n);
        act_num_meatool = n;
    }
    // get data of measurement tools
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "idi", iarr, NULL, &d)))
        {
            return false;
        }
        if (iarr[0] != i)
        { // not expected
            return false;
        }
        act_meatool[i].id = iarr[0];
        act_meatool[i].quality = d;

        act_meatool[i].num_button = 1;

        k = iarr[1];
        for (j = 0; j < act_meatool[i].num_button; j++)
        {
            act_meatool[i].button[j] = k & 0x01;
            k >>= 1;
        }
        for (j = act_meatool[i].num_button; j < DTRACK_MEATOOL_MAX_BUTTON; j++)
        {
            act_meatool[i].button[j] = 0;
        }

        act_meatool[i].tipradius = 0.0;

        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_meatool[i].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_meatool[i].rot)))
        {
            return false;
        }

        for (j = 0; j < 6; j++)
            act_meatool[i].cov[j] = 0.0;
    }

    return true;
}

/**
 * 	\brief Parses a single line of measurement tool data (newer format) data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6dmt2' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6dmt2(char **line)
{
    int i, j, k, l, n, iarr[2];
    double darr[2];
    char sfmt[20];

    // get number of calibrated measurement tools
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    loc_num_meatool = 0;
    // adjust length of vector
    if (n != act_num_meatool)
    {
        act_meatool.resize(n);
        act_num_meatool = n;
    }
    // get data of measurement tools
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "idid", iarr, NULL, darr)))
        {
            return false;
        }
        if (iarr[0] != i)
        { // not expected
            return false;
        }
        act_meatool[i].id = iarr[0];
        act_meatool[i].quality = darr[0];

        act_meatool[i].num_button = iarr[1];
        if (act_meatool[i].num_button > DTRACK_MEATOOL_MAX_BUTTON)
            act_meatool[i].num_button = DTRACK_MEATOOL_MAX_BUTTON;

        for (j = act_meatool[i].num_button; j < DTRACK_MEATOOL_MAX_BUTTON; j++)
        {
            act_meatool[i].button[j] = 0;
        }

        act_meatool[i].tipradius = darr[1];

        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_meatool[i].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_meatool[i].rot)))
        {
            return false;
        }

        strcpy(sfmt, "");
        j = 0;
        while (j < act_meatool[i].num_button)
        {
            strcat(sfmt, "i");
            j += 32;
        }

        if (!(*line = string_get_block(*line, sfmt, iarr, NULL, NULL)))
        {
            return false;
        }
        k = l = 0;
        for (j = 0; j < act_meatool[i].num_button; j++)
        {
            act_meatool[i].button[j] = iarr[k] & 0x01;
            iarr[k] >>= 1;
            l++;
            if (l == 32)
            {
                k++;
                l = 0;
            }
        }

        if (!(*line = string_get_block(*line, "dddddd", NULL, NULL, act_meatool[i].cov)))
        {
            return false;
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of measurement reference data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6dmtr' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6dmtr(char **line)
{
    int i, n, id;
    double d;

    // get number of measurement references
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }

    // adjust length of vector
    if (n != act_num_mearef)
    {
        act_mearef.resize(n);
        act_num_mearef = n;
    }

    // reset data
    for (i = 0; i < n; i++)
    {
        act_mearef[i].id = i;
        act_mearef[i].quality = -1;
    }

    // get number of calibrated measurement references
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }

    // get data of measurement references
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "id", &id, NULL, &d)))
        {
            return false;
        }
        if (id < 0 || id >= (int)act_mearef.size())
        {
            return false;
        }
        act_mearef[id].quality = d;

        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_mearef[id].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_mearef[id].rot)))
        {
            return false;
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of additional information about number of calibrated Fingertracking hands in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of 'glcal' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_glcal(char **line)
{
    if (!(*line = string_get_i(*line, &loc_num_handcal)))
    { // get number of calibrated hands
        return false;
    }

    return true;
}

/**
 * 	\brief Parses a single line of A.R.T. Fingertracking hand data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of 'gl' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_gl(char **line)
{
    int i, j, n, iarr[3], id;
    double d, darr[6];

    // disable all existing data
    for (i = 0; i < act_num_hand; i++)
    {
        memset(&act_hand[i], 0, sizeof(DTrack_Hand_Type_d));
        act_hand[i].id = i;
        act_hand[i].quality = -1;
    }
    // get number of hands (in line)
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // get data of hands
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "idii", iarr, NULL, &d)))
        {
            return false;
        }
        id = iarr[0];
        if (id >= act_num_hand)
        { // adjust length of vector
            act_hand.resize(id + 1);
            for (j = act_num_hand; j <= id; j++)
            {
                memset(&act_hand[j], 0, sizeof(DTrack_Hand_Type_d));
                act_hand[j].id = j;
                act_hand[j].quality = -1;
            }
            act_num_hand = id + 1;
        }
        act_hand[id].id = iarr[0];
        act_hand[id].lr = iarr[1];
        act_hand[id].quality = d;
        if (iarr[2] > DTRACK_HAND_MAX_FINGER)
        {
            return false;
        }
        act_hand[id].nfinger = iarr[2];
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_hand[id].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_hand[id].rot)))
        {
            return false;
        }
        // get data of fingers
        for (j = 0; j < act_hand[id].nfinger; j++)
        {
            if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_hand[id].finger[j].loc)))
            {
                return false;
            }
            if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_hand[id].finger[j].rot)))
            {
                return false;
            }
            if (!(*line = string_get_block(*line, "dddddd", NULL, NULL, darr)))
            {
                return false;
            }
            act_hand[id].finger[j].radiustip = darr[0];
            act_hand[id].finger[j].lengthphalanx[0] = darr[1];
            act_hand[id].finger[j].anglephalanx[0] = darr[2];
            act_hand[id].finger[j].lengthphalanx[1] = darr[3];
            act_hand[id].finger[j].anglephalanx[1] = darr[4];
            act_hand[id].finger[j].lengthphalanx[2] = darr[5];
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of 6dj human model data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6dj' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6dj(char **line)
{
    int i, j, n, iarr[2], id;
    double d, darr[6];

    // get number of calibrated human models
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // adjust length of vector
    if (n != act_num_human)
    {
        act_human.resize(n);
        act_num_human = n;
    }
    for (i = 0; i < act_num_human; i++)
    {
        memset(&act_human[i], 0, sizeof(DTrack_Human_Type_d));
        act_human[i].id = i;
        act_human[i].num_joints = 0;
    }

    // get number of human models
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    int id_human;
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "ii", iarr, NULL, NULL)))
        {
            return false;
        }
        if (iarr[0] > act_num_human - 1) // not expected
            return false;

        id_human = iarr[0];
        act_human[id_human].id = iarr[0];
        act_human[id_human].num_joints = iarr[1];

        for (j = 0; j < iarr[1]; j++)
        {
            if (!(*line = string_get_block(*line, "id", &id, NULL, &d)))
            {
                return false;
            }
            act_human[id_human].joint[j].id = id;
            act_human[id_human].joint[j].quality = d;

            if (!(*line = string_get_block(*line, "dddddd", NULL, NULL, darr)))
            {
                return false;
            }
            memcpy(act_human[id_human].joint[j].loc, &darr, 3 * sizeof(double));
            memcpy(act_human[id_human].joint[j].ang, &darr[3], 3 * sizeof(double));

            if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_human[id_human].joint[j].rot)))
            {
                return false;
            }
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of 6di inertial data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '6di' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_6di(char **line)
{
    int i, j, n, iarr[2], id, st;
    double d;

    // disable all existing data
    for (i = 0; i < act_num_inertial; i++)
    {
        memset(&act_inertial[i], 0, sizeof(DTrack_Inertial_Type_d));
        act_inertial[i].id = i;
        act_inertial[i].st = 0;
        act_inertial[i].error = 0;
    }
    // get number of calibrated inertial bodies
    if (!(*line = string_get_i(*line, &n)))
    {
        return false;
    }
    // get data of inertial bodies
    for (i = 0; i < n; i++)
    {
        if (!(*line = string_get_block(*line, "iid", iarr, NULL, &d)))
        {
            return false;
        }
        id = iarr[0];
        st = iarr[1];
        // adjust length of vector
        if (id >= act_num_inertial)
        {
            act_inertial.resize(id + 1);
            for (j = act_num_inertial; j <= id; j++)
            {
                memset(&act_inertial[j], 0, sizeof(DTrack_Inertial_Type_d));
                act_inertial[i].id = i;
                act_inertial[i].st = 0;
                act_inertial[i].error = 0;
            }
            act_num_inertial = id + 1;
        }
        act_inertial[id].id = id;
        act_inertial[id].st = st;
        act_inertial[id].error = d;
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_inertial[id].loc)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddddddddd", NULL, NULL, act_inertial[id].rot)))
        {
            return false;
        }
    }

    return true;
}

/**
 * 	\brief Parses a single line of single marker data in one tracking data packet.
 *
 *	Updates internal data structures.
 *
 *	@param[in,out]  line		line of '3d' data in one tracking data packet
 *	@return	Parsing succeeded?
 */
bool DTrackParser::parseLine_3d(char **line)
{
    int i;

    // get number of markers
    if (!(*line = string_get_i(*line, &act_num_marker)))
    {
        act_num_marker = 0;
        return false;
    }
    if (act_num_marker > (int)act_marker.size())
    {
        act_marker.resize(act_num_marker);
    }
    // get data of single markers
    for (i = 0; i < act_num_marker; i++)
    {
        if (!(*line = string_get_block(*line, "id", &act_marker[i].id, NULL, &act_marker[i].quality)))
        {
            return false;
        }
        if (!(*line = string_get_block(*line, "ddd", NULL, NULL, act_marker[i].loc)))
        {
            return false;
        }
    }

    return true;
}

/**
 * 	\brief	Get number of calibrated standard bodies (as far as known).
 *
 *	Refers to last received frame.
 *	@return		number of calibrated standard bodies
 */
int DTrackParser::getNumBody()
{
    return act_num_body;
}

/**
 * 	\brief	Get standard body data
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	id	id, range 0 .. (max standard body id - 1)
 *	@return		id-th standard body data
 */
DTrack_Body_Type_d *DTrackParser::getBody(int id)
{
    if ((id >= 0) && (id < act_num_body))
        return &act_body.at(id);
    return NULL;
}

/**
 * 	\brief	Get number of calibrated Flysticks.
 *
 *	Refers to last received frame.
 *	@return		number of calibrated Flysticks
 */
int DTrackParser::getNumFlyStick()
{
    return act_num_flystick;
}

/**
 * 	\brief	Get Flystick data.
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	id	id, range 0 .. (max flystick id - 1)
 *	@return		id-th Flystick data.
 */
DTrack_FlyStick_Type_d *DTrackParser::getFlyStick(int id)
{
    if ((id >= 0) && (id < act_num_flystick))
        return &act_flystick.at(id);
    return NULL;
}

/**
 * 	\brief	Get number of calibrated measurement tools.
 *
 *	Refers to last received frame.
 *	@return		number of calibrated measurement tools
 */
int DTrackParser::getNumMeaTool()
{
    return act_num_meatool;
}

/**
 * 	\brief	Get measurement tool data.
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	id	id, range 0 .. (max tool id - 1)
 *	@return		id-th measurement tool data.
 */
DTrack_MeaTool_Type_d *DTrackParser::getMeaTool(int id)
{
    if ((id >= 0) && (id < act_num_meatool))
        return &act_meatool.at(id);
    return NULL;
}

/**
 * 	\brief	Get number of calibrated measurement references.
 *
 *	Refers to last received frame.
 *	@return		number of calibrated measurement references
 */
int DTrackParser::getNumMeaRef()
{
    return act_num_mearef;
}

/**
 * 	\brief	Get measurement reference data.
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	id	id, range 0 .. (max measurement reference id - 1)
 *	@return		id-th measurement reference data.
 */
DTrack_MeaRef_Type_d *DTrackParser::getMeaRef(int id)
{
    if ((id >= 0) && (id < act_num_mearef))
        return &act_mearef.at(id);
    return NULL;
}

/**
 * 	\brief	Get number of calibrated Fingertracking hands (as far as known).
 *
 *	Refers to last received frame.
 *	@return		number of calibrated fingertracking hands
 */
int DTrackParser::getNumHand()
{
    return act_num_hand;
}

/**
 * 	\brief	Get Fingertracking hand data.
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	id	id, range 0 .. (max hand id - 1)
 *	@return		id-th Fingertracking hand data
 */
DTrack_Hand_Type_d *DTrackParser::getHand(int id)
{
    if ((id >= 0) && (id < act_num_hand))
        return &act_hand.at(id);
    return NULL;
}

/**
* 	\brief	Get human data
*
*	Refers to last received frame. Currently not tracked human models get a num_joints 0
*	@return		  id-th human model data
*/
int DTrackParser::getNumHuman()
{
    return act_num_human;
}

/**
* 	\brief	Get human data
*
*	Refers to last received frame. Currently not tracked human models get a num_joints 0
*	@param[in]	id	id, range 0 .. (max standard body id - 1)
*	@return		id-th human model data
*/
DTrack_Human_Type *DTrackParser::getHuman(int id)
{
    if ((id >= 0) && (id < act_num_human))
        return &act_human.at(id);
    return NULL;
}

/**
* 	\brief	Get number of calibrated inertial bodies
*
*	Refers to last received frame.
*	@return		number of calibrated inertial bodies
*/
int DTrackParser::getNumInertial()
{
    return act_num_inertial;
}

/**
* 	\brief	Get i data
*
*	Refers to last received frame. Currently not tracked inertial body get a state of 0
*	@param[in]	id	id, range 0 .. (max inertial body id - 1)
*	@return		id-th inertial body data
*/
DTrack_Inertial_Type_d *DTrackParser::getInertial(int id)
{
    if ((id >= 0) && (id < act_num_inertial))
        return &act_inertial.at(id);
    return NULL;
}

/**
 * 	\brief	Get number of tracked single markers.
 *
 *	Refers to last received frame.
 *	@return	number of tracked single markers
 */
int DTrackParser::getNumMarker()
{
    return act_num_marker;
}

/**
 * 	\brief	Get single marker data.
 *
 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
 *	@param[in]	index	index, range 0 .. (max marker id - 1)
 *	@return		i-th single marker data
 */
DTrack_Marker_Type_d *DTrackParser::getMarker(int index)
{
    if ((index >= 0) && (index < act_num_marker))
        return &act_marker.at(index);
    return NULL;
}

/**
 * 	\brief	Get frame counter.
 *
 *	Refers to last received frame.
 *	@return		frame counter
 */
unsigned int DTrackParser::getFrameCounter()
{
    return act_framecounter;
}

/**
 * 	\brief	Get timestamp.
 *
 *	Refers to last received frame.
 *	@return		timestamp (-1 if information not available)
 */
double DTrackParser::getTimeStamp()
{
    return act_timestamp;
}
