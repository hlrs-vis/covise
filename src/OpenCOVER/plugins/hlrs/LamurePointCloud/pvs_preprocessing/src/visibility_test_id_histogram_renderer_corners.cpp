// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/visibility_test_id_histogram_renderer_corners.h"
#include "lamure/pvs/glut_wrapper.h"
#include "lamure/pvs/utils.h"
#include "lamure/pvs/pvs_database.h"

#include "lamure/ren/model_database.h"
#include "lamure/ren/cut_database.h"
#include "lamure/ren/policy.h"

#include <boost/program_options.hpp>

namespace lamure
{
namespace pvs
{

visibility_test_id_histogram_renderer_corners::
visibility_test_id_histogram_renderer_corners()
{
	resolution_x_ = 1920;
	resolution_y_ = 1080;
	video_memory_budget_ = 2048;
	main_memory_budget_ = 4096;
	max_upload_budget_ = 64;
}

visibility_test_id_histogram_renderer_corners::
~visibility_test_id_histogram_renderer_corners()
{
	shutdown();
}

int visibility_test_id_histogram_renderer_corners::
initialize(int& argc, char** argv)
{
 	namespace po = boost::program_options;
    namespace fs = boost::filesystem;

    const std::string exec_name = (argc > 0) ? fs::basename(argv[0]) : "";
    scm::shared_ptr<scm::core> scm_core(new scm::core(1, argv));

    putenv((char *)"__GL_SYNC_TO_VBLANK=0");

    std::string resource_file_path = "";

    // These value are read, but not used. Yet ignoring them in the terminal parameters would lead to misinterpretation.
    std::string pvs_output_file_path = "";
    std::string visibility_test_type = "";
    std::string grid_type = "";
    unsigned int grid_size = 1;
    unsigned int num_steps = 11;
    double oversize_factor = 1.5;
    float optimization_threshold = 1.0f;

	po::options_description desc("Usage: " + exec_name + " [OPTION]... INPUT\n\n"
                               "Allowed Options");
    desc.add_options()
      ("help", "print help message")
      ("width,w", po::value<int>(&resolution_x_)->default_value(1024), "specify window width (default=1024)")
      ("height,h", po::value<int>(&resolution_y_)->default_value(1024), "specify window height (default=1024)")
      ("resource-file,f", po::value<std::string>(&resource_file_path), "specify resource input-file")
      ("vram,v", po::value<unsigned>(&video_memory_budget_)->default_value(2048), "specify graphics memory budget in MB (default=2048)")
      ("mem,m", po::value<unsigned>(&main_memory_budget_)->default_value(4096), "specify main memory budget in MB (default=4096)")
      ("upload,u", po::value<unsigned>(&max_upload_budget_)->default_value(64), "specify maximum video memory upload budget per frame in MB (default=64)")
    // The following parameters are used by the main app only, yet must be identified nonetheless since otherwise they are dealt with as file paths.
      ("pvs-file,p", po::value<std::string>(&pvs_output_file_path), "specify output file of calculated pvs data")
      ("vistest", po::value<std::string>(&visibility_test_type)->default_value("histogramrenderer"), "specify type of visibility test to be used. ('histogramrenderer', 'randomhistogramrenderer')")
      ("gridtype", po::value<std::string>(&grid_type)->default_value("octree"), "specify type of grid to store visibility data ('regular', 'octree', 'hierarchical')")
      ("gridsize", po::value<unsigned int>(&grid_size)->default_value(1), "specify size/depth of the grid used for the visibility test (depends on chosen grid type)")
      ("oversize", po::value<double>(&oversize_factor)->default_value(1.5), "factor the grid bounds will be scaled by, default is 1.5 (grid bounds will exceed scene bounds by factor of 1.5)")
      ("optithresh", po::value<float>(&optimization_threshold)->default_value(1.0f), "specify the threshold at which common data are converged. Default is 1.0, which means data must be 100 percent equal.")
      ("numsteps,n", po::value<unsigned int>(&num_steps)->default_value(11), "specify the number of intervals the occlusion values will be split into (visibility analysis only)");
      ;

    po::variables_map vm;

    try
    {    
		auto parsed_options = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
		po::store(parsed_options, vm);
		po::notify(vm);

		std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed_options.options, po::include_positional);
		bool no_input = !vm.count("input") && to_pass_further.empty();

		if (resource_file_path == "")
		{
			if (vm.count("help") || no_input)
			{
				std::cout << desc;
				return 0;
			}
		}

		// no explicit input -> use unknown options
		if (!vm.count("input") && resource_file_path == "") 
		{
			resource_file_path = "auto_generated.rsc";
			std::fstream ofstr(resource_file_path, std::ios::out);
			if (ofstr.good()) 
			{
			for (auto argument : to_pass_further)
			{
				ofstr << argument << std::endl;
			}
		}
		else
		{
			throw std::runtime_error("Cannot open file");
		}
			ofstr.close();
		}
	}
	catch (std::exception& e)
	{
		std::cout << "Warning: No input file specified. \n" << desc;
		return 0;
	}

	lamure::pvs::glut_wrapper::initialize(argc, argv, resolution_x_, resolution_y_, nullptr);

    std::pair< std::vector<std::string>, std::vector<scm::math::mat4f> > model_attributes;
    std::set<lamure::model_t> visible_set;
    std::set<lamure::model_t> invisible_set;
    model_attributes = read_model_string(resource_file_path, &visible_set, &invisible_set);

    std::vector<scm::math::mat4f> & model_transformations = model_attributes.second;
    std::vector<std::string> const& model_filenames = model_attributes.first;

    lamure::ren::policy* policy = lamure::ren::policy::get_instance();
    policy->set_max_upload_budget_in_mb(max_upload_budget_);
    policy->set_render_budget_in_mb(video_memory_budget_);
    policy->set_out_of_core_budget_in_mb(main_memory_budget_);
    policy->set_window_width(resolution_x_);
    policy->set_window_height(resolution_y_);

	lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    management_ = new management_id_histogram_renderer_corners(model_filenames, model_transformations, visible_set, invisible_set);
    glut_wrapper::set_management(management_);
    management_->set_pvs_file_path(pvs_output_file_path);

    // Calculate bounding box of whole scene.
    for(lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
    {
        // Cast required from boxf to bounding_box.
        const scm::gl::boxf& box_model_root = database->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];
        vec3r min_vertex(box_model_root.min_vertex() + database->get_model(model_id)->get_bvh()->get_translation());
        vec3r max_vertex(box_model_root.max_vertex() + database->get_model(model_id)->get_bvh()->get_translation());
        bounding_box model_root_box(min_vertex, max_vertex);

        if(model_id == 0)
        {
            scene_bounds_ = bounding_box(model_root_box);
        }
        else
        {
            scene_bounds_.expand(model_root_box);
        }  
    }

    return 0;
}

void visibility_test_id_histogram_renderer_corners::
test_visibility(grid* visibility_grid)
{
	// Does the visibility test in the main loop, returns once all images are rendered and analyzed.
	management_->set_grid(visibility_grid);
	glutMainLoop();
}

void visibility_test_id_histogram_renderer_corners::
shutdown()
{
	// Renderer shutdown.
    if (management_ != nullptr)
    {
        delete lamure::pvs::pvs_database::get_instance();

        delete lamure::ren::cut_database::get_instance();
        delete lamure::ren::controller::get_instance();
        delete lamure::ren::model_database::get_instance();
        delete lamure::ren::policy::get_instance();
        delete lamure::ren::ooc_cache::get_instance();

        delete management_;
        management_ = nullptr;
    }

    glut_wrapper::quit();
}

bounding_box visibility_test_id_histogram_renderer_corners::
get_scene_bounds() const
{
	return scene_bounds_;
}

}
}
