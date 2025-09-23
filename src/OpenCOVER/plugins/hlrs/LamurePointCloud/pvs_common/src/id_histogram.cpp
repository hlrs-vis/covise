// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/id_histogram.h"
#include <iostream>
#include <fstream>
#include <bitset>

namespace lamure
{
namespace pvs
{

id_histogram::id_histogram()
{
}

id_histogram::~id_histogram()
{
}

void id_histogram::
create(const void* pixelData, const size_t& numPixels)
{
	histogram_.clear();
	const unsigned int* pixelDataInt = (unsigned int*)pixelData;

	for(size_t index = 0; index < numPixels; ++index)
	{
		unsigned int pixelValue = pixelDataInt[index];
		model_t modelID = (pixelValue >> 24) & 0xFF;					// RGBA-value is written in order AGBR, so skip 24 bits to get to model ID within alpha channel.

		modelID = 255 - modelID;										// Debug thingie. Helps to create a more visible object by starting at higher alpha values.

		if(modelID != 255)
		{
			node_t nodeID = pixelValue & 0xFFFFFF;						// RGBA-value is written in order AGBR, so first 24 bits are node ID.
			histogram_[modelID][nodeID]++;
		}
	}
}

std::map<model_t, std::vector<node_t>> id_histogram::
get_visible_nodes(const size_t& numPixels, const float& visibilityThreshold) const
{
	std::map<model_t, std::vector<node_t>> visibleNodes;

	for(std::map<model_t, std::map<node_t, size_t>>::const_iterator modelIter = histogram_.begin(); modelIter != histogram_.end(); ++modelIter)
	{
		for(std::map<node_t, size_t>::const_iterator nodeIter = modelIter->second.begin(); nodeIter != modelIter->second.end(); ++nodeIter)
		{
			if(((float)nodeIter->second / (float)numPixels) * 100.0f >= visibilityThreshold)
			{
				visibleNodes[modelIter->first].push_back(nodeIter->first);
			}
		}
	}

	return visibleNodes;
}

std::map<model_t, std::map<node_t, size_t>> id_histogram::
get_histogram() const
{
	return histogram_;
}

}
}
