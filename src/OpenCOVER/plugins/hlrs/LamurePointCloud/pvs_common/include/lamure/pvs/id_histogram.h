// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_ID_HISTOGRAM_H_
#define LAMURE_PVS_ID_HISTOGRAM_H_

#include <vector>
#include <map>
#include <utility>

#include <lamure/pvs/pvs.h>
#include <lamure/types.h>

namespace lamure
{
namespace pvs
{

class id_histogram
{
public:
	id_histogram();
	~id_histogram();

	void create(const void* pixelData, const size_t& numPixels);
	
	std::map<model_t, std::vector<node_t>> get_visible_nodes(const size_t& numPixels, const float& visibilityThreshold) const;
	std::map<model_t, std::map<node_t, size_t>> get_histogram() const;

private:
	std::map<model_t, std::map<node_t, size_t>> histogram_;			// first key is model ID, second key is node ID, final value is amount of visible pixels
};

}
}

#endif
