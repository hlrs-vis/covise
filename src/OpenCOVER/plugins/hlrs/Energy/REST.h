/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _REST_H
#define _REST_H

#include <curl/curl.h>
#include <string>

/**
 * @brief Function to perform a CURL request
 * 
 * @param url The URL to send the request to
 * @param response The response data received from the request (storage)
 * @return bool True if the request was successful, false otherwise
 */
bool performCurlRequest(const std::string& url, std::string& response);

/**
 * @brief Function to cleanup the CURL library (need to be called once for each application)
 */
void cleanupcurl();

#endif
