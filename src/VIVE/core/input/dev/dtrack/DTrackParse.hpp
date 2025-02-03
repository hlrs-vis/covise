/* DTrackParse: C++ header file
 *
 * Functions for processing data
 *
 * Copyright 2007-2021, Advanced Realtime Tracking GmbH & Co. KG
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
 * Version v2.7.0
 * 
 */

#ifndef _ART_DTRACKPARSE_H_
#define _ART_DTRACKPARSE_H_

#include <string>

namespace DTrackSDK_Parse {

/**
 *	\brief	Search next line in buffer
 *	@param[in] 	str		buffer (total)
 *	@param[in] 	start	start position within buffer
 *	@param[in] 	len		buffer length in bytes
 *	@return		begin of line, NULL if no new line in buffer
 */
char* string_nextline(char* str, char* start, int len);

/**
 * 	\brief	Read next 'int' value from string
 *
 *	@param[in] 	str		string
 *	@param[out] i		read value
 *	@return		pointer behind read value in str; NULL in case of error
 */
char* string_get_i(char* str, int* i);

/**
 * 	\brief	Read next 'unsigned int' value from string
 *
 *	@param[in] 	str		string
 *	@param[out]	ui		read value
 *	@return		pointer behind read value in str; NULL in case of error
 */
char* string_get_ui(char* str, unsigned int* ui);

/**
 * 	\brief	Read next 'double' value from string
 *
 *	@param[in] 	str		string
 *	@param[out] d		read value
 *	@return 	pointer behind read value in str; NULL in case of error
 */
char* string_get_d(char* str, double* d);

/**
 * 	\brief	Read next 'float' value from string
 *
 *	@param[in] 	str		string
 *	@param[out] f 		read value
 *	@return 	pointer behind read value in str; NULL in case of error
 */
char* string_get_f(char* str, float* f);

/**
 * 	\brief Process next block '[...]' in string
 *
 *	@param[in] 	str		string
 *	@param[in] 	fmt		format string ('i' for 'int', 'f' for 'float')
 *	@param[out] idat	array for 'int' values (long enough due to fmt)
 *	@param[out] fdat	array for 'float' values (long enough due to fmt)
 *	@param[out] ddat	array for 'double' values (long enough due to fmt)
 *	@return 	pointer behind read value in str; NULL in case of error
 */
char* string_get_block(char* str, const char* fmt, int* idat = NULL, float* fdat = NULL, double *ddat = NULL);

/**
 * 	\brief	Read next 'word' value from string
 *
 *	@param[in] 	str		string
 *	@param[out] w		read value
 *	@return	pointer behind read value in str; NULL in case of error
 */
char* string_get_word(char* str, std::string& w);

/**
 * 	\brief	Read next 'quoted text' value from string
 *
 *	@param[in] 	str		string
 *	@param[out] qt		read value (without quotes)
 *	@return pointer behind read value in str; NULL in case of error
 */
char* string_get_quoted_text(char* str, std::string& qt);

/**
 * 	\brief	Compare strings regarding DTrack2 parameter rules
 *
 *	@param 	str		string
 *	@param 	p		parameter string
 *	@return pointer behind parameter in str; NULL in case of error
 */
char* string_cmp_parameter(char* str, const char* p);

}

#endif /* _ART_DTRACKPARSE_H_ */
