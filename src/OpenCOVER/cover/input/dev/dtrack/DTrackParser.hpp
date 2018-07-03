/* DTrackParser: C++ header file, A.R.T. GmbH
 *
 * Functions for receiving and sending UDP/TCP packets
 *
 * Copyright (c) 2013-2017, Advanced Realtime Tracking GmbH
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Version v2.5.0
 *
 * Purpose:
 *  - DTrack2 network protocol due to: 'Technical Appendix DTrack v2.0'
 *  - for ARTtrack Controller versions v0.2 (and compatible versions)
 *
 */

#ifndef _ART_DTRACKPARSER_HPP_
#define _ART_DTRACKPARSER_HPP_

#include "DTrackDataTypes.h"

#include <vector>

using namespace DTrackSDK_Datatypes;

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
	unsigned int getFrameCounter() const;
	
	/**
	 * 	\brief	Get timestamp.
	 *
	 *	Refers to last received frame.
	 *	@return		timestamp (-1 if information not available)
	 */
	double getTimeStamp() const;
	
	/**
	 * 	\brief	Get number of calibrated standard bodies (as far as known).
	 *
	 *	Refers to last received frame.
	 *	@return		number of calibrated standard bodies
	 */
	int getNumBody() const;
	
	/**
	 * 	\brief	Get standard body data
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max standard body id - 1)
	 *	@return		id-th standard body data
	 */
	const DTrack_Body_Type_d* getBody(int id) const;
	
	/**
	 * 	\brief	Get number of calibrated Flysticks.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated Flysticks.
	 */
	int getNumFlyStick() const;
	
	/**
	 * 	\brief	Get Flystick data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max flystick id - 1)
	 *	@return		id-th Flystick data.
	 */
	const DTrack_FlyStick_Type_d* getFlyStick(int id) const;
	
	/**
	 * 	\brief	Get number of calibrated measurement tools.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated measurement tools.
	 */
	int getNumMeaTool() const;
	
	/**
	 * 	\brief	Get measurement tool data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max tool id - 1)
	 *	@return		id-th measurement tool data.
	 */
	const DTrack_MeaTool_Type_d* getMeaTool(int id) const;
	
	/**
	 * 	\brief	Get number of calibrated measurement references.
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated measurement references.
	 */
	int getNumMeaRef() const;
	
	/**
	 * 	\brief	Get measurement reference data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max tool id - 1)
	 *	@return		id-th measurement reference data.
	 */
	const DTrack_MeaRef_Type_d* getMeaRef(int id) const;
	
	/**
	 * 	\brief	Get number of calibrated Fingertracking hands (as far as known).
	 *
	 *	Refers to last received frame.
	 *	@return	Number of calibrated Fingertracking hands (as far as known).
	 */
	int getNumHand() const;
	
	/**
	 * 	\brief	Get Fingertracking hand data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	id	id, range 0 .. (max hand id - 1)
	 *	@return		id-th Fingertracking hand data
	 */
	const DTrack_Hand_Type_d* getHand(int id) const;
	
	/**
	* 	\brief	Get number of calibrated human models (as far as known).
	*
	*	Refers to last received frame.
	*	@return		number of calibrated human models
	*/
	int getNumHuman() const;
	
	/**
	* 	\brief	Get human model data
	*
	*	Refers to last received frame. Currently not tracked human models get a num_joints 0
	*	@param[in]	id	id, range 0 .. (max standard body id - 1)
	*	@return		id-th human model data
	*/
	const DTrack_Human_Type_d* getHuman(int id) const;
	
	/**
	* 	\brief	Get number of calibrated inertial bodies
	*
	*	Refers to last received frame.
	*	@return		number of calibrated inertial bodies
	*/
	int getNumInertial() const;
	
	/**
	* 	\brief	Get inertial data
	*
	*	Refers to last received frame. Currently not tracked inertial body get a state of 0
	*	@param[in]	id	id, range 0 .. (max inertial body id - 1)
	*	@return		id-th inertial body data
	*/
	const DTrack_Inertial_Type_d* getInertial(int id) const;
	
	/**
	 * 	\brief	Get number of tracked single markers.
	 *
	 *	Refers to last received frame.
	 *	@return	number of tracked single markers
	 */
	int getNumMarker() const;
	
	/**
	 * 	\brief	Get single marker data.
	 *
	 *	Refers to last received frame. Currently not tracked bodies get a quality of -1.
	 *	@param[in]	index	index, range 0 .. (max marker id - 1)
	 *	@return		i-th single marker data
	 */
	const DTrack_Marker_Type_d* getMarker(int index) const;
	
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
	 * 	\brief Parses a single line of 6d covariance data in one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *
	 *	@param[in,out]  line		line of '6dcov' data in one tracking data packet
	 *	@return	Parsing succeeded?
	 */
	bool parseLine_6dcov(char **line);
	
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
	unsigned int act_framecounter;                    //!< frame counter
	double act_timestamp;                             //!< timestamp (-1, if information not available)
	int act_num_body;                                 //!< number of calibrated standard bodies (as far as known)
	std::vector<DTrack_Body_Type_d> act_body;         //!< array containing standard body data
	int act_num_flystick;                             //!< number of calibrated Flysticks
	std::vector<DTrack_FlyStick_Type_d> act_flystick; //!< array containing Flystick data
	int act_num_meatool;                              //!< number of calibrated measurement tools
	std::vector<DTrack_MeaTool_Type_d> act_meatool;   //!< array containing measurement tool data
	int act_num_mearef;                               //!< number of calibrated measurement references
	std::vector<DTrack_MeaRef_Type_d> act_mearef;     //!< array containing measurement reference data
	int act_num_hand;                                 //!< number of calibrated Fingertracking hands (as far as known)
	std::vector<DTrack_Hand_Type_d> act_hand;         //!< array containing Fingertracking hands data
	int act_num_human;                                //!< number of calibrated human models
	std::vector<DTrack_Human_Type_d> act_human;         //!< array containing human model data
	int act_num_inertial;                             //!< number of tracked single markers
	std::vector<DTrack_Inertial_Type_d> act_inertial; //!< array containing single marker data
	int act_num_marker;                               //!< number of tracked single markers
	std::vector<DTrack_Marker_Type_d> act_marker;     //!< array containing single marker data
	
	int loc_num_bodycal;	//!< internal use, local number of calibrated bodies
	int loc_num_handcal;	//!< internal use, local number of hands
	int loc_num_flystick1;	//!< internal use, local number of old flysticks
	int loc_num_meatool1;   //!< internal use, local number of old measurementtools
};


#endif /* _ART_DTRACKPARSER_HPP_ */
