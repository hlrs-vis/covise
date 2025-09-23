// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <iostream>
#include <queue>
#include <utility>

#include "lamure/pvs/grid_optimizer_irregular.h"
#include "lamure/pvs/grid_irregular.h"

namespace lamure
{
namespace pvs
{

void grid_optimizer_irregular::
optimize_grid(grid* input_grid, const float& equality_threshold)
{
	grid_irregular* irr_grid = (grid_irregular*)input_grid;

	for(long current_cell_index = 0; current_cell_index < irr_grid->get_cell_count(); ++current_cell_index)
	{
		bool is_original_cell = irr_grid->is_cell_at_index_original(current_cell_index);
		const view_cell* current_cell = irr_grid->get_cell_at_index(current_cell_index);

		std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<std::pair<float, size_t>>> merge_options;

		for(long compare_cell_index = 0; compare_cell_index < irr_grid->get_cell_count(); ++compare_cell_index)
		{
			if(compare_cell_index == current_cell_index)
			{
				continue;
			}

			const view_cell* compare_cell = irr_grid->get_cell_at_index(compare_cell_index);

			// Check equality.
			size_t equality_counter = 0;
			size_t total_nodes = 0;

			view_cell_regular* current_view_cell_regular = (view_cell_regular*)current_cell;
			view_cell_regular* compare_view_cell_regular = (view_cell_regular*)compare_cell;

			for(model_t model_index = 0; model_index < irr_grid->get_num_models(); ++model_index)
			{
				boost::dynamic_bitset<> bitset_one = boost::dynamic_bitset<>(current_view_cell_regular->get_bitset(model_index));
				bitset_one.resize(input_grid->get_num_nodes(model_index));
				boost::dynamic_bitset<> bitset_two = boost::dynamic_bitset<>(compare_view_cell_regular->get_bitset(model_index));
				bitset_two.resize(input_grid->get_num_nodes(model_index));

				boost::dynamic_bitset<> common_bits = bitset_one & bitset_two;

				equality_counter += common_bits.count();
				total_nodes += common_bits.size();
			}

			float equality = (float)equality_counter / (float)total_nodes;
			float error = 1.0f - equality;

			// This test will already cancel some unecessary tests, yet a further threshold test will be done within join_cells.
			if(equality >= equality_threshold)
			{
				merge_options.push(std::pair<float, size_t>(error, compare_cell_index));
			}
		}

		// Merge view cells.
		while(merge_options.size() > 0)
		{
			if(irr_grid->join_cells(current_cell_index, merge_options.top().second, merge_options.top().first, equality_threshold) && is_original_cell)
			{
				--current_cell_index;
				break;
			}

			merge_options.pop();
		}

		double current_percentage_done = ((double)current_cell_index / (double)irr_grid->get_cell_count()) * 100.0;
		std::cout << "\rgrid optimization in progress [" << current_percentage_done << "]       " << std::flush;
	}

	std::cout << "\rgrid optimization in progress [100]       " << std::endl;
}

}
}