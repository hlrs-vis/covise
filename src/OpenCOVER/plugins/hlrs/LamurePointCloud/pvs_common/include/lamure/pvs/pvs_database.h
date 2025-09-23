// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_PVS_DATABASE_H
#define LAMURE_PVS_PVS_DATABASE_H

#include <set>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"
#include <lamure/semaphore.h>

namespace lamure
{
namespace pvs
{

class pvs_database
{
public:
	virtual ~pvs_database();
	static pvs_database* get_instance();

	bool load_pvs_from_file(const std::string& grid_file_path, const std::string& pvs_file_path, const bool& do_preload);
	
	// Some helper function not affecting the PVS runtime behavior, but allow to globally have calls which will deal with the different grid types properly.
	grid* load_grid_from_file(const std::string& grid_file_path) const;
	grid* load_grid_from_file(const std::string& grid_file_path, const std::string& pvs_file_path) const;
	grid* create_grid_by_type(const std::string& grid_type) const;
	grid* create_grid_by_type(const std::string& grid_type, const size_t& num_cells_x, const size_t& num_cells_y, const size_t& num_cells_z, const double& bounds_size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids) const;


	virtual void set_viewer_position(const scm::math::vec3d& position);
	virtual bool get_viewer_visibility(const model_t& model_id, const node_t node_id) const;

	void activate(const bool& act);
	bool is_activated() const;

	const grid* get_visibility_grid() const;
	const grid* get_bounding_grid() const;
	void clear_visibility_grid();

protected:
	pvs_database();

	static pvs_database* instance_;

private:
	void loading_thread_loop();
	void load_visibility_data_async(uint64_t cell_index);
	std::queue<uint64_t> loading_queue_;
	semaphore semaphore_;

	// Grid storing the major visibility data of the scene.
	grid* visibility_grid_;

	// Grid storing the visibility data from further away to fill the room that is left by the visibility grid.
	grid* bounding_grid_;

	// The PVS needs the current viewer position to properly answer visibility requests.
	scm::math::vec3d position_viewer_;
	const view_cell* viewer_cell_;

	// If the PVS is not activated, it will always return true on visibility requests.
	bool activated_;

	// If true, the complete visibility data will be loaded on initialization.
	// If false, visibilibity of view cells will be loaded depending on the necessity to do so (e.g. if the user enters a view cell).
	bool do_preload_;
	bool shutdown_;
	std::string pvs_file_path_;

	scm::math::vec3d smallest_cell_size_;
	std::set<size_t> previously_loaded_cell_indices_;

	std::thread visibility_data_loading_thread_;

	// Used to achieve thread safety.
	mutable std::mutex mutex_;
	mutable std::mutex loading_mutex_;
};

}
}

#endif
