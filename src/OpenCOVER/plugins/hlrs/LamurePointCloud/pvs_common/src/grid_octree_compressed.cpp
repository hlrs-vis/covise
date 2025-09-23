// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <fstream>
#include <sstream>
#include <climits>

#include "lamure/pvs/grid_octree_compressed.h"

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace lamure
{
namespace pvs
{

grid_octree_compressed::
grid_octree_compressed() : grid_octree_compressed(1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_octree_compressed::
grid_octree_compressed(const size_t& octree_depth, const double& size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids) : grid_octree(octree_depth, size, position_center, ids)
{
}

grid_octree_compressed::
~grid_octree_compressed()
{
}

std::string grid_octree_compressed::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_octree_compressed::
get_grid_identifier()
{
	return "octree_compressed";
}

void grid_octree_compressed::
save_grid_to_file(const std::string& file_path) const
{
	save_octree_grid(file_path, get_grid_identifier());
}

void grid_octree_compressed::
save_visibility_to_file(const std::string& file_path) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out | std::ios::binary);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	std::vector<std::string> compressed_data_blocks;

	// Iterate over view cells.
	size_t num_cells = cell_count_recursive(root_node_);
	for(size_t cell_index = 0; cell_index < num_cells; ++cell_index)
	{
		std::string current_cell_data = "";
		const view_cell* current_cell = cells_by_indices_[cell_index];

		// Iterate over models in the scene.
		for(lamure::model_t model_id = 0; model_id < ids_.size(); ++model_id)
		{
			node_t num_nodes = ids_.at(model_id);
			char current_byte = 0x00;

			size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);
			size_t character_counter = 0;
			std::string current_line_data(line_length, 0x00);

			// Iterate over nodes in the model.
			for(lamure::node_t node_id = 0; node_id < num_nodes; ++node_id)
			{
				if(current_cell->get_visibility(model_id, node_id))
				{
					current_byte |= 1 << (node_id % CHAR_BIT);
				}

				// Flush character if either 8 bits are written or if the node id is the last one.
				if((node_id + 1) % CHAR_BIT == 0 || node_id == (num_nodes - 1))
				{
					current_line_data[character_counter] = current_byte;
					character_counter++;

					current_byte = 0x00;
				}
			}

			current_cell_data = current_cell_data + current_line_data;
		}

		// Compression of binary data using boost gzip.
		std::stringstream stream_uncompressed, stream_compressed;
		stream_uncompressed << current_cell_data;

		boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
		in.push(boost::iostreams::gzip_compressor());
		in.push(stream_uncompressed);
		boost::iostreams::copy(in, stream_compressed);

		std::string output_string = stream_compressed.str();
		compressed_data_blocks.push_back(output_string);
	}

	// Save sizes of compressed data blocks.
	for(size_t current_block_index = 0; current_block_index < compressed_data_blocks.size(); ++current_block_index)
	{
		uint64_t current_block_size = compressed_data_blocks[current_block_index].size();
		file_out.write(reinterpret_cast<char*>(&current_block_size), sizeof(current_block_size));
	}

	// Save compressed data blocks.
	for(size_t current_block_index = 0; current_block_index < compressed_data_blocks.size(); ++current_block_index)
	{
		file_out.write(compressed_data_blocks[current_block_index].c_str(), compressed_data_blocks[current_block_index].length());
	}

	file_out.close();
}

bool grid_octree_compressed::
load_grid_from_file(const std::string& file_path)
{
	return load_octree_grid(file_path, get_grid_identifier());
}

bool grid_octree_compressed::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	// Read access points to data blocks.
	visibility_block_sizes_.clear();

	for(size_t current_block_index = 0; current_block_index < this->get_cell_count(); ++current_block_index)
	{
		uint64_t block_size;
		file_in.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
		visibility_block_sizes_.push_back(block_size);
	}

	// Read compressed data blocks.
	std::vector<std::string> compressed_data_blocks;

	for(size_t current_block_index = 0; current_block_index < this->get_cell_count(); ++current_block_index)
	{
		size_t block_size = visibility_block_sizes_[current_block_index];
    std::vector<char> current_block_data(block_size);
		file_in.read(&current_block_data[0], block_size);

		std::string current_block(&current_block_data[0], block_size);
		compressed_data_blocks.push_back(current_block);
	}

	size_t num_cells = cell_count_recursive(root_node_);
	for(size_t cell_index = 0; cell_index < num_cells; ++cell_index)
	{
		view_cell* current_cell = cells_by_indices_[cell_index];

		// Decompress.
		std::stringstream stream_uncompressed, stream_compressed;
		stream_compressed << compressed_data_blocks[cell_index];

		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
		inbuf.push(boost::iostreams::gzip_decompressor());
		inbuf.push(stream_compressed);
		boost::iostreams::copy(inbuf, stream_uncompressed);
		
		// Apply visibility data.
		for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
		{
			node_t num_nodes = ids_.at(model_index);
			size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);
      std::vector<char> current_line_data(line_length);

			stream_uncompressed.read(&current_line_data[0], line_length);

			// Used to avoid continuing resize within visibility data.
			current_cell->set_visibility(model_index, num_nodes - 1, false);

			for(node_t character_index = 0; character_index < line_length; ++character_index)
			{
				char current_byte = current_line_data[character_index];
				
				for(unsigned short bit_index = 0; bit_index < CHAR_BIT; ++bit_index)
				{
					bool visible = ((current_byte >> bit_index) & 1) == 0x01;
					current_cell->set_visibility(model_index, (character_index * CHAR_BIT) + bit_index, visible);
				}
			}
		}
	}

	file_in.close();
	return true;
}

bool grid_octree_compressed::
load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index)
{
	std::lock_guard<std::mutex> lock(mutex_);

	view_cell* current_cell = cells_by_indices_[cell_index];

	// First check if visibility data is already loaded.
	if(current_cell->contains_visibility_data())
	{
		return true;
	}

	// If no visibility data exists, open the file and load them.
	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	if(visibility_block_sizes_.size() == 0)
	{
		// Read access points to data blocks.
		for(size_t current_block_index = 0; current_block_index < this->get_cell_count(); ++current_block_index)
		{
			uint64_t block_size;
			file_in.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
			visibility_block_sizes_.push_back(block_size);
		}
	}

	// Read compressed data block.
	size_t block_size = visibility_block_sizes_[cell_index];
  std::vector<char> current_block_data(block_size);

	// Find proper position in file. First data is compresses block sizes (one 64 bit integer per view cell).
	size_t file_position = visibility_block_sizes_.size() * sizeof(uint64_t);
	for(size_t visibility_cell_index = 0; visibility_cell_index < cell_index; ++visibility_cell_index)
	{
		file_position += visibility_block_sizes_[visibility_cell_index];
	}

	file_in.seekg(file_position);
	file_in.read(&current_block_data[0], block_size);
	std::string compressed_data_block(&current_block_data[0], block_size);

	// Decompress.
	std::stringstream stream_uncompressed, stream_compressed;
	stream_compressed << compressed_data_block;

	boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
	inbuf.push(boost::iostreams::gzip_decompressor());
	inbuf.push(stream_compressed);
	boost::iostreams::copy(inbuf, stream_uncompressed);

	// Apply visibility data.
	for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		node_t num_nodes = ids_.at(model_index);
		size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);
    std::vector<char> current_line_data(line_length);

		stream_uncompressed.read(&current_line_data[0], line_length);

		// Used to avoid continuing resize within visibility data.
		current_cell->set_visibility(model_index, num_nodes - 1, false);

		for(node_t character_index = 0; character_index < line_length; ++character_index)
		{
			char current_byte = current_line_data[character_index];
			
			for(unsigned short bit_index = 0; bit_index < CHAR_BIT; ++bit_index)
			{
				bool visible = ((current_byte >> bit_index) & 1) == 0x01;
				current_cell->set_visibility(model_index, (character_index * CHAR_BIT) + bit_index, visible);
			}
		}
	}

	file_in.close();
	return true;
}

}
}