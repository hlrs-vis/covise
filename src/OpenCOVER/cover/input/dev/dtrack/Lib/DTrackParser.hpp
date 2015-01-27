/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* DTrackParser: C++ header file, A.R.T. GmbH 17.6.13
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

#ifndef _ART_DTRACKPARSER_HPP_
#define _ART_DTRACKPARSER_HPP_

#include "DTrackDataTypes.h"
#include "DTrackParse.hpp"

using namespace DTrackSDK_Datatypes;
using namespace DTrackSDK_Parse;

#include <vector>

/**
 * 	\brief DTrack Parser class.
 */
class DTrackParser
{
public:
    /**
	 * 	\brief Constructor.
	 */
    DTrackParser();

    /**
	 * 	\brief Destructor.
	 */
    virtual ~DTrackParser();

    /**
	 * 	\brief Set default values at start of a new frame.
	 */
    virtual void startFrame();

    /**
	 * 	\brief Final adjustments after processing all data for a frame.
	 */
    virtual void endFrame();

    /**
	 * 	\brief Parses a single line of data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		one line of data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    virtual bool parseLine(char **line);

    /**
	 * 	\brief	Get frame counter.
	 *
	 *	Refers to last received frame.
	 *	@return		frame counter
	 */
    unsigned int getFrameCounter();

    /**
	 * 	\brief	Get timestamp.
	 *
	 *	Refers to last received frame.
	 *	@return		timestamp (-1 if information not available)
	 */
    double getTimeStamp();

    /**
	 * 	\brief	Get number of calibrated standard bodies (as far as known).
	 *
	 *	Refers to last received frame.
	 *	@return		number of calibrated standard bodies
	 */
    int getNumBody();

    /**
	 * 	\brief	Get standard body data
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max standard body id - 1)
	 *	@return		id-th standard body data
	 */
    DTrack_Body_Type_d *getBody(int id);

    /**
	 * 	\brief	Get number of calibrated Flysticks.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated Flysticks.
	 */
    int getNumFlyStick();

    /**
	 * 	\brief	Get Flystick data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max flystick id - 1)
	 *	@return		id-th Flystick data.
	 */
    DTrack_FlyStick_Type_d *getFlyStick(int id);

    /**
	 * 	\brief	Get number of calibrated measurement tools.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated measurement tools.
	 */
    int getNumMeaTool();

    /**
	 * 	\brief	Get measurement tool data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max tool id - 1)
	 *	@return		id-th measurement tool data.
	 */
    DTrack_MeaTool_Type_d *getMeaTool(int id);

    /**
	 * 	\brief	Get number of calibrated measurement references.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated measurement references.
	 */
    int getNumMeaRef();

    /**
	 * 	\brief	Get measurement reference data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max tool id - 1)
	 *	@return		id-th measurement reference data.
	 */
    DTrack_MeaRef_Type_d *getMeaRef(int id);

    /**
	 * 	\brief	Get number of calibrated Fingertracking hands (as far as known).
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated Fingertracking hands (as far as known).
	 */
    int getNumHand();

    /**
	 * 	\brief	Get Fingertracking hand data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max hand id - 1)
	 *	@return		id-th Fingertracking hand data
	 */
    DTrack_Hand_Type_d *getHand(int id);

    /**
	* 	\brief	Get number of calibrated human models (as far as known).
	*
	*	Refers to last received frame.
	*	@return		number of calibrated human models
	*/
    int getNumHuman();

    /**
	* 	\brief	Get human model data
	*
	*	Refers to last received frame. Currently not tracked human models get a num_joints 0
	*	@param[in]	id	id, range 0 .. (max standard body id - 1)
	*	@return		id-th human model data
	*/
    DTrack_Human_Type *getHuman(int id);

    /**
	* 	\brief	Get number of calibrated inertial bodies
	*
	*	Refers to last received frame.
	*	@return		number of calibrated inertial bodies
	*/
    int getNumInertial();

    /**
	* 	\brief	Get inertial data
	*
	*	Refers to last received frame. Currently not tracked inertial body get a state of 0
	*	@param[in]	id	id, range 0 .. (max inertial body id - 1)
	*	@return		id-th inertial body data
	*/
    DTrack_Inertial_Type_d *getInertial(int id);

    /**
	 * 	\brief	Get number of tracked single markers.
	 *
	 *	Refers to last received frame.
	 *	@return	number of tracked single markers
	 */
    int getNumMarker();

    /**
	 * 	\brief	Get single marker data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	index	index, range 0 .. (max marker id - 1)
	 *	@return		i-th single marker data
	 */
    DTrack_Marker_Type_d *getMarker(int index);

protected:
    /**
	 * 	\brief Parses a single line of frame counter data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of 'fr' data DTrackParserin one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_fr(char **line);

    /**
	 * 	\brief Parses a single line of timestamp data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of 'ts' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_ts(char **line);

    /**
	 * 	\brief Parses a single line of additional information about number of calibrated bodies in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dcal' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6dcal(char **line);

    /**
	 * 	\brief Parses a single line of standard body data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6d' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6d(char **line);

    /**
	 * 	\brief Parses a single line of Flystick data (older format) data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6df' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6df(char **line);

    /**
	 * 	\brief Parses a single line of Flystick data (newer format) data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6df2' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6df2(char **line);

    /**
	 * 	\brief Parses a single line of measurement tool data (older format) in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dmt' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6dmt(char **line);

    /**
	 * 	\brief Parses a single line of measurement tool data (newer format) data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dmt2' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6dmt2(char **line);

    /**
	 * 	\brief Parses a single line of measurement reference data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dmtr' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6dmtr(char **line);

    /**
	 * 	\brief Parses a single line of additional information about number of calibrated Fingertracking hands in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of 'glcal' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_glcal(char **line);

    /**
	 * 	\brief Parses a single line of A.R.T. Fingertracking hand data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of 'gl' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_gl(char **line);

    /**
	 * 	\brief Parses a single line of 6dj human model data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dj' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6dj(char **line);

    /**
	 * 	\brief Parses a single line of 6di inertial data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6di' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_6di(char **line);

    /**
	 * 	\brief Parses a single line of single marker data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '3d' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
    bool parseLine_3d(char **line);

protected:
    unsigned int act_framecounter; //!< frame counter
    double act_timestamp; //!< timestamp (-1, if information not available)
    int act_num_body; //!< number of calibrated standard bodies (as far as known)
    std::vector<DTrack_Body_Type_d> act_body; //!< array containing standard body data
    int act_num_flystick; //!< number of calibrated Flysticks
    std::vector<DTrack_FlyStick_Type_d> act_flystick; //!< array containing Flystick data
    int act_num_meatool; //!< number of calibrated measurement tools
    std::vector<DTrack_MeaTool_Type_d> act_meatool; //!< array containing measurement tool data
    int act_num_mearef; //!< number of calibrated measurement references
    std::vector<DTrack_MeaRef_Type_d> act_mearef; //!< array containing measurement reference data
    int act_num_hand; //!< number of calibrated Fingertracking hands (as far as known)
    std::vector<DTrack_Hand_Type_d> act_hand; //!< array containing Fingertracking hands data
    int act_num_human; //!< number of calibrated human models
    std::vector<DTrack_Human_Type> act_human; //!< array containing human model data
    int act_num_inertial; //!< number of tracked single markers
    std::vector<DTrack_Inertial_Type_d> act_inertial; //!< array containing single marker data
    int act_num_marker; //!< number of tracked single markers
    std::vector<DTrack_Marker_Type_d> act_marker; //!< array containing single marker data

    int loc_num_bodycal; //!< internal use, local number of calibrated bodies
    int loc_num_handcal; //!< internal use, local number of hands
    int loc_num_flystick1; //!< internal use, local number of old flysticks
    int loc_num_meatool; //!< internal use, local number of measurementtools
};

#endif /* _ART_DTRACKPARSER_HPP_ */
