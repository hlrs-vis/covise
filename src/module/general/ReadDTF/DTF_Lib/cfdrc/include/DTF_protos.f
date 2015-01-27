! F90 prototypes/interfaces for DTF API
! DTF Version 6.0.11

MODULE DTF_protos

  INTERFACE

    FUNCTION dtf_query_dtf_version_f(dtf_version)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_dtf_version_f
      character(len=80), intent(inout) :: dtf_version
    END FUNCTION

    FUNCTION dtf_query_file_version_f(fh, dtf_version)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_file_version_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: dtf_version
    END FUNCTION

    FUNCTION dtf_ok_f()
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_ok_f
    END FUNCTION

    FUNCTION dtf_last_error_f()
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_last_error_f
    END FUNCTION

    FUNCTION dtf_info_last_error_f(error_string)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_info_last_error_f
      character(len=80), intent(inout) :: error_string
    END FUNCTION

    FUNCTION dtf_clear_error_f()
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_clear_error_f
    END FUNCTION

    FUNCTION dtf_new_file_f(filename)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_new_file_f
      character(len=80), intent(inout) :: filename
    END FUNCTION

    FUNCTION dtf_open_file_f(filename)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_open_file_f
      character(len=80), intent(inout) :: filename
    END FUNCTION

    FUNCTION dtf_close_file_f(fh)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_close_file_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_close_all_files_f()
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_close_all_files_f
    END FUNCTION

    FUNCTION dtf_set_scaling_d_f(fh, scaling)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_scaling_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      real(selected_real_kind(8)), intent(inout) :: scaling
    END FUNCTION

    FUNCTION dtf_set_scaling_s_f(fh, scaling)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_scaling_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      real(selected_real_kind(4)), intent(inout) :: scaling
    END FUNCTION

    FUNCTION dtf_query_scaling_d_f(fh)
      IMPLICIT NONE
      real(selected_real_kind(8)) :: dtf_query_scaling_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_query_scaling_s_f(fh)
      IMPLICIT NONE
      real(selected_real_kind(4)) :: dtf_query_scaling_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_set_application_f(fh, application)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_application_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: application
    END FUNCTION

    FUNCTION dtf_query_application_f(fh, application)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_application_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: application
    END FUNCTION

    FUNCTION dtf_set_appversion_f(fh, appversion)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_appversion_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: appversion
    END FUNCTION

    FUNCTION dtf_query_appversion_f(fh, appversion)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_appversion_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: appversion
    END FUNCTION

    FUNCTION dtf_set_title_f(fh, title)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_title_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: title
    END FUNCTION

    FUNCTION dtf_query_title_f(fh, title)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_title_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: title
    END FUNCTION

    FUNCTION dtf_set_origin_f(fh, origin)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_origin_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: origin
    END FUNCTION

    FUNCTION dtf_query_origin_f(fh, origin)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_origin_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(inout) :: origin
    END FUNCTION

    FUNCTION dtf_query_cretime_f(fh)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_cretime_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_query_modtime_f(fh)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_modtime_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_check_patches_f(filename)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_check_patches_f
      character(len=80), intent(inout) :: filename
    END FUNCTION

    FUNCTION dtf_file_info_f(filename)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_file_info_f
      character(len=80), intent(inout) :: filename
    END FUNCTION

    FUNCTION dtf_file_contents_f(filename, max_array_print, is_html)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_file_contents_f
      character(len=80), intent(inout) :: filename
      integer(selected_int_kind(8)), intent(inout) :: max_array_print
      integer(selected_int_kind(8)), intent(inout) :: is_html
    END FUNCTION

    FUNCTION dtf_test_file_f(filename)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_test_file_f
      character(len=80), intent(inout) :: filename
    END FUNCTION

    FUNCTION dtf_query_nsims_f(fh)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nsims_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_add_sim_f(fh, descr)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_sim_f
      integer(selected_int_kind(8)), intent(in) :: fh
      character(len=80), intent(in) :: descr
    END FUNCTION

    FUNCTION dtf_copy_sim_f(fh, simnum, descr)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_copy_sim_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: descr
    END FUNCTION

    FUNCTION dtf_delete_sim_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_sim_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

    FUNCTION dtf_update_simdescr_f(fh, simnum, descr)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_simdescr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: descr
    END FUNCTION

    FUNCTION dtf_query_simdescr_f(fh, simnum, descr)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_simdescr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: descr
    END FUNCTION

    FUNCTION dtf_query_minmax_sim_d_f(fh, simnum, minmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_minmax_sim_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      real(selected_real_kind(8)), dimension(6), intent(inout) :: minmax
    END FUNCTION

    FUNCTION dtf_query_minmax_sim_s_f(fh, simnum, minmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_minmax_sim_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      real(selected_real_kind(4)), dimension(6), intent(inout) :: minmax
    END FUNCTION

    FUNCTION dtf_query_nzones_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzones_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

    FUNCTION dtf_query_zonetype_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zonetype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_isstruct_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_isstruct_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_iscartesian_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_iscartesian_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_isunstruct_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_isunstruct_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_ispoint_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ispoint_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_ispoly_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ispoly_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_is_2d_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_is_2d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_delete_zone_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_is_blanking_data_present_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_is_blanking_data_present_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_dims_f(fh, simnum, zonenum, dim)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_dims_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(3), intent(inout) :: dim
    END FUNCTION

    FUNCTION dtf_query_nnodes_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nnodes_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_nnodes_struct_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nnodes_struct_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_update_grid_d_f(fh, simnum, zonenum, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_grid_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_update_grid_s_f(fh, simnum, zonenum, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_grid_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_read_grid_d_f(fh, simnum, zonenum, nodenum, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_grid_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nodenum
      real(selected_real_kind(8)), dimension(*), intent(inout) :: x
      real(selected_real_kind(8)), dimension(*), intent(inout) :: y
      real(selected_real_kind(8)), dimension(*), intent(inout) :: z
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: blanking
    END FUNCTION

    FUNCTION dtf_read_grid_s_f(fh, simnum, zonenum, nodenum, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_grid_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nodenum
      real(selected_real_kind(4)), dimension(*), intent(inout) :: x
      real(selected_real_kind(4)), dimension(*), intent(inout) :: y
      real(selected_real_kind(4)), dimension(*), intent(inout) :: z
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: blanking
    END FUNCTION

    FUNCTION dtf_query_minmax_zone_d_f(fh, simnum, zonenum, minmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_minmax_zone_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      real(selected_real_kind(8)), dimension(6), intent(inout) :: minmax
    END FUNCTION

    FUNCTION dtf_query_minmax_zone_s_f(fh, simnum, zonenum, minmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_minmax_zone_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      real(selected_real_kind(4)), dimension(6), intent(inout) :: minmax
    END FUNCTION

    FUNCTION dtf_read_blanking_f(fh, simnum, zonenum, nodenum, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_blanking_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nodenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: blanking
    END FUNCTION

    FUNCTION dtf_update_blanking_f(fh, simnum, zonenum, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_blanking_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_add_struct_d_f(fh, simnum, dim, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_struct_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), dimension(3), intent(in) :: dim
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_add_struct_s_f(fh, simnum, dim, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_struct_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), dimension(3), intent(in) :: dim
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_update_struct_double_f(fh, simnum, zonenum, dim, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_struct_double_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(3), intent(in) :: dim
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_update_struct_single_f(fh, simnum, zonenum, dim, x, y, z, blanking)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_struct_single_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(3), intent(in) :: dim
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(*), intent(in) :: blanking
    END FUNCTION

    FUNCTION dtf_add_unstruct_d_f(fh, simnum, nnodes, x, y, z, ncells, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_unstruct_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(6), intent(in) :: ncells
      integer(selected_int_kind(8)), dimension(*), intent(in) :: cells
    END FUNCTION

    FUNCTION dtf_add_unstruct_s_f(fh, simnum, nnodes, x, y, z, ncells, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_unstruct_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(6), intent(in) :: ncells
      integer(selected_int_kind(8)), dimension(*), intent(in) :: cells
    END FUNCTION

    FUNCTION dtf_update_unstruct_double_f(fh, simnum, zonenum, nnodes, x, y, z, ncells, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_unstruct_double_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(6), intent(in) :: ncells
      integer(selected_int_kind(8)), dimension(*), intent(in) :: cells
    END FUNCTION

    FUNCTION dtf_update_unstruct_single_f(fh, simnum, zonenum, nnodes, x, y, z, ncells, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_unstruct_single_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), dimension(6), intent(in) :: ncells
      integer(selected_int_kind(8)), dimension(*), intent(in) :: cells
    END FUNCTION

    FUNCTION dtf_add_poly_d_f(fh, simnum, nnodes, x, y, z, n_faces_total, n_nodes_per_face, len_f2n, f2n, len_f2c, f2c, is_2D)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_poly_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), intent(in) :: n_faces_total
      integer(selected_int_kind(8)), dimension(*), intent(in) :: n_nodes_per_face
      integer(selected_int_kind(8)), intent(in) :: len_f2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2n
      integer(selected_int_kind(8)), intent(in) :: len_f2c
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2c
      integer(selected_int_kind(8)), intent(in) :: is_2D
    END FUNCTION

    FUNCTION dtf_add_poly_s_f(fh, simnum, nnodes, x, y, z, n_faces_total, n_nodes_per_face, len_f2n, f2n, len_f2c, f2c, is_2D)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_poly_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), intent(in) :: n_faces_total
      integer(selected_int_kind(8)), dimension(*), intent(in) :: n_nodes_per_face
      integer(selected_int_kind(8)), intent(in) :: len_f2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2n
      integer(selected_int_kind(8)), intent(in) :: len_f2c
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2c
      integer(selected_int_kind(8)), intent(in) :: is_2D
    END FUNCTION

    FUNCTION dtf_update_poly_double_f(fh, simnum, zonenum, nnodes, x, y, z, n_faces_total, n_nodes_per_face, len_f2n, f2n, len_f2c,&
    & f2c, is_2D)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_poly_double_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), intent(in) :: n_faces_total
      integer(selected_int_kind(8)), dimension(*), intent(in) :: n_nodes_per_face
      integer(selected_int_kind(8)), intent(in) :: len_f2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2n
      integer(selected_int_kind(8)), intent(in) :: len_f2c
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2c
      integer(selected_int_kind(8)), intent(in) :: is_2D
    END FUNCTION

    FUNCTION dtf_update_poly_single_f(fh, simnum, zonenum, nnodes, x, y, z, n_faces_total, n_nodes_per_face, len_f2n, f2n, len_f2c,&
    & f2c, is_2D)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_poly_single_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nnodes
      real(selected_real_kind(4)), dimension(*), intent(in) :: x
      real(selected_real_kind(4)), dimension(*), intent(in) :: y
      real(selected_real_kind(4)), dimension(*), intent(in) :: z
      integer(selected_int_kind(8)), intent(in) :: n_faces_total
      integer(selected_int_kind(8)), dimension(*), intent(in) :: n_nodes_per_face
      integer(selected_int_kind(8)), intent(in) :: len_f2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2n
      integer(selected_int_kind(8)), intent(in) :: len_f2c
      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2c
      integer(selected_int_kind(8)), intent(in) :: is_2D
    END FUNCTION

    FUNCTION dtf_query_ispoly_sorted_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ispoly_sorted_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_sort_poly_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_sort_poly_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_poly_sizes_f(fh, simnum, zonenum, nnodes, n_faces_total, n_bfaces_total, n_interface_faces_total, n_cells_to&
    &tal, len_f2n, len_f2c, len_c2n, len_c2f)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_poly_sizes_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nnodes
      integer(selected_int_kind(8)), intent(inout) :: n_faces_total
      integer(selected_int_kind(8)), intent(inout) :: n_bfaces_total
      integer(selected_int_kind(8)), intent(inout) :: n_interface_faces_total
      integer(selected_int_kind(8)), intent(inout) :: n_cells_total
      integer(selected_int_kind(8)), intent(inout) :: len_f2n
      integer(selected_int_kind(8)), intent(inout) :: len_f2c
      integer(selected_int_kind(8)), intent(inout) :: len_c2n
      integer(selected_int_kind(8)), intent(inout) :: len_c2f
    END FUNCTION

    FUNCTION dtf_query_nfaces_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nfaces_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_nfaces_struct_f( fh,  simnum,  zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nfaces_struct_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_faces_f(fh, simnum, zonenum, n_faces_of_type, n_faces_of_kind)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_faces_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(7), intent(inout) :: n_faces_of_type
      integer(selected_int_kind(8)), dimension(21), intent(inout) :: n_faces_of_kind
    END FUNCTION

    FUNCTION dtf_query_f2n_pos_f(fh, simnum, zonenum, facenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_f2n_pos_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
    END FUNCTION

    FUNCTION dtf_query_facekind_f(fh, sim, zone, facenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_facekind_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
      integer(selected_int_kind(8)), intent(inout) :: facenum
    END FUNCTION

    FUNCTION dtf_query_ncells_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ncells_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_cells_f(fh, simnum, zonenum, n_cells_of_type)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_cells_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(11), intent(inout) :: n_cells_of_type
    END FUNCTION

    FUNCTION dtf_query_celltype_f(fh, simnum, zonenum, cellnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_celltype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
    END FUNCTION

    FUNCTION dtf_query_c2n_pos_f(fh, simnum, zonenum, cellnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_c2n_pos_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
    END FUNCTION

    FUNCTION dtf_query_n2c_f(fh, simnum, zonenum, nodenum, n_cells_per_node)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_n2c_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nodenum
      integer(selected_int_kind(8)), intent(inout) :: n_cells_per_node
    END FUNCTION

    FUNCTION dtf_read_n2c_f(fh, simnum, zonenum, nodenum, n2c)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_n2c_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: nodenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: n2c
    END FUNCTION

    FUNCTION dtf_query_f2n_f(fh, simnum, zonenum, facenum, n_nodes_per_face)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_f2n_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: n_nodes_per_face
    END FUNCTION

    FUNCTION dtf_read_f2n_f(fh, simnum, zonenum, facenum, f2n)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_f2n_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: f2n
    END FUNCTION

    FUNCTION dtf_query_f2c_f(fh, simnum, zonenum, facenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_f2c_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
    END FUNCTION

    FUNCTION dtf_read_f2c_f(fh, simnum, zonenum, facenum, f2c)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_f2c_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: f2c
    END FUNCTION

    FUNCTION dtf_query_c2n_f(fh, simnum, zonenum, cellnum, n_nodes_per_cell)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_c2n_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: n_nodes_per_cell
    END FUNCTION

    FUNCTION dtf_read_c2n_f(fh, simnum, zonenum, cellnum, c2n)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_c2n_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: c2n
    END FUNCTION

    FUNCTION dtf_query_c2f_f(fh, simnum, zonenum, cellnum, n_faces_per_cell)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_c2f_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: n_faces_per_cell
    END FUNCTION

    FUNCTION dtf_read_c2f_f(fh, simnum, zonenum, cellnum, c2f)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_c2f_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cellnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: c2f
    END FUNCTION

    FUNCTION dtf_read_virtual_nodenums_f(fh, simnum, zonenum, v_nn)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_virtual_nodenums_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: v_nn
    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_read_struct_zone_virtual_nodenums_f(fh, simnum, zonenum, v_nn)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_struct_zone_virtual_nodenums_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), dimension(*), intent(inout) :: v_nn
!    END FUNCTION

    FUNCTION dtf_read_virtual_facenums_f(fh, simnum, zonenum, v_fn)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_virtual_facenums_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: v_fn
    END FUNCTION

    FUNCTION dtf_read_virtual_cellnums_f(fh, simnum, zonenum, v_cn)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_virtual_cellnums_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: v_cn
    END FUNCTION

    FUNCTION dtf_query_nvzds_of_topotype_f(fh, simnum, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nvzds_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_query_nvzds_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nvzds_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

    FUNCTION dtf_read_vzdnums_of_topotype_f(fh, simnum, topotype, vzdnums)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vzdnums_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: topotype
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: vzdnums
    END FUNCTION

    FUNCTION dtf_query_vzd_by_num_f(fh, simnum, num, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vzd_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: num
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_query_vzd_by_name_f(fh, simnum, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vzd_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_vzd_by_num_f(fh, simnum, num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_vzd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(inout) :: num
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_vzd_by_name_f(fh, simnum, name, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_vzd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(inout) :: name
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_vzd_by_num_f(fh, simnum, num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_vzd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: num
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_vzd_by_name_f(fh, simnum, name, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_vzd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(in) :: name
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!    END FUNCTION

    FUNCTION dtf_query_vz_bcrec_num_f(fh, simnum, zonenum, bcrec_num)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vz_bcrec_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: bcrec_num
    END FUNCTION

    FUNCTION dtf_query_bf2bcr_f(fh, simnum, zonenum, nbfaces_of_type)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(7), intent(inout) :: nbfaces_of_type
    END FUNCTION

    FUNCTION dtf_read_bf2bcr_f(fh, simnum, zonenum, facenum, bf2f, bf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: bf2f
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: bf2r
    END FUNCTION

    FUNCTION dtf_read_bf2nbcr_f(fh, simnum, zonenum, facenum, bf2n, bf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bf2nbcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: bf2n
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: bf2r
    END FUNCTION

    FUNCTION dtf_update_bf2bcr_f(fh, simnum, zonenum, nboundary_faces, bf2f, bf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nboundary_faces
      integer(selected_int_kind(8)), dimension(*), intent(in) :: bf2f
      integer(selected_int_kind(8)), dimension(*), intent(in) :: bf2r
    END FUNCTION

    FUNCTION dtf_update_bf2n_bf2bcr_f(fh, simnum, zonenum, nboundary_faces, bf2n, bf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bf2n_bf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nboundary_faces
      integer(selected_int_kind(8)), dimension(*), intent(in) :: bf2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: bf2r
    END FUNCTION

    FUNCTION dtf_read_xf2bcr_f(fh, simnum, zonenum, facenum, xf2f, xf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_xf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: xf2f
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: xf2r
    END FUNCTION

    FUNCTION dtf_read_xf2nbcr_f(fh, simnum, zonenum, facenum, xf2n, xf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_xf2nbcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: facenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: xf2n
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: xf2r
    END FUNCTION

    FUNCTION dtf_update_xf2bcr_f(fh, simnum, zonenum, ninterface_faces, xf2f, xf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_xf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: ninterface_faces
      integer(selected_int_kind(8)), dimension(*), intent(in) :: xf2f
      integer(selected_int_kind(8)), dimension(*), intent(in) :: xf2r
    END FUNCTION

    FUNCTION dtf_update_xf2n_xf2bcr_f(fh, simnum, zonenum, nboundary_faces, xf2n, xf2r)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_xf2n_xf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: nboundary_faces
      integer(selected_int_kind(8)), dimension(*), intent(in) :: xf2n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: xf2r
    END FUNCTION

    FUNCTION dtf_query_xf2bcr_f(fh, simnum, zonenum, nxfaces_of_type)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_xf2bcr_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), dimension(7), intent(inout) :: nxfaces_of_type
    END FUNCTION

    FUNCTION dtf_query_bfnum_by_fnum_f(fh, simnum, zonenum, global_face_index)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bfnum_by_fnum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: global_face_index
    END FUNCTION

    FUNCTION dtf_query_ifnum_by_fnum_f(fh, simnum, zonenum, global_face_index)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ifnum_by_fnum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: global_face_index
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_add_zd_f(fh, simnum, zonenum, name, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_add_zd_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      character(len=80), intent(in) :: name
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_add_zd_strings_f(fh, simnum, zonenum, name, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_zd_strings_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdInt_f(fh,  simnum,  zonenum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdInt_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdIntArray_f(fh,  simnum,  zonenum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdIntArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdSingle_f(fh,  simnum,  zonenum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdSingle_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdSingleArray_f(fh,  simnum,  zonenum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdSingleArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdDouble_f(fh,  simnum,  zonenum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdDouble_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdDoubleArray_f(fh,  simnum,  zonenum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdDoubleArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdString_f(fh,  simnum,  zonenum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdString_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddZdStringArray_f(fh,  simnum,  zonenum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddZdStringArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtf_delete_zd_by_num_f(fh, simnum, zonenum, datanum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_zd_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
    END FUNCTION

    FUNCTION dtf_delete_zd_by_name_f(fh, simnum, zonenum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_zd_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_zd_by_num_f(fh, simnum, zonenum, datanum, name, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_zd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: datanum
!      character(len=80), intent(in) :: name
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_update_zd_strings_by_num_f(fh, simnum, zonenum, datanum, name, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_zd_strings_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdIntByNum_f(fh,  simnum,  zonenum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdIntByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdIntArrayByNum_f(fh,  simnum,  zonenum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdIntArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdSingleByNum_f(fh,  simnum,  zonenum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdSingleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdSingleArrayByNum_f(fh,  simnum,  zonenum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdSingleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdDoubleByNum_f(fh,  simnum,  zonenum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdDoubleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdDoubleArrayByNum_f(fh,  simnum,  zonenum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdDoubleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdStringByNum_f(fh,  simnum,  zonenum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdStringByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdStringArrayByNum_f(fh,  simnum,  zonenum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdStringArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_zd_by_name_f(fh, simnum, zonenum, name, newname, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_zd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      character(len=80), intent(in) :: name
!      character(len=80), intent(in) :: newname
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_update_zd_strings_by_name_f(fh, simnum, zonenum, name, newname, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_zd_strings_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdIntByName_f(fh,  simnum,  zonenum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdIntByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdIntArrayByName_f(fh,  simnum,  zonenum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdIntArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdSingleByName_f(fh,  simnum,  zonenum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdSingleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdSingleArrayByName_f(fh,  simnum,  zonenum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdSingleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdDoubleByName_f(fh,  simnum,  zonenum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdDoubleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdDoubleArrayByName_f(fh,  simnum,  zonenum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdDoubleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdStringByName_f(fh,  simnum,  zonenum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdStringByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateZdStringArrayByName_f(fh,  simnum,  zonenum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateZdStringArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtf_query_nzds_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzds_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_nzds_of_topotype_f(fh, simnum, zonenum, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzds_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_read_zdnums_of_topotype_f(fh, simnum, zonenum, topotype, nums)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zdnums_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: topotype
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: nums
    END FUNCTION

    FUNCTION dtf_query_zd_by_num_f(fh, simnum, zonenum, datanum, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zd_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_query_zd_by_name_f(fh, simnum, zonenum, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zd_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_zd_by_num_f(fh, simnum, zonenum, datanum, element_num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_zd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(inout) :: datanum
!      integer(selected_int_kind(8)), intent(inout) :: element_num
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtf_read_zd_strings_by_num_f(fh, simnum, zonenum, datanum, element_num, data, datatype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zd_strings_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
      integer(selected_int_kind(8)), intent(inout) :: datatype
    END FUNCTION

    FUNCTION dtfReadZdIntByNum_f(fh,  simnum,  zonenum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdIntByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      integer(selected_int_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdIntArrayByNum_f(fh,  simnum,  zonenum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdIntArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdSingleByNum_f(fh,  simnum,  zonenum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdSingleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(4)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdSingleArrayByNum_f(fh,  simnum,  zonenum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdSingleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      real(selected_real_kind(4)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdDoubleByNum_f(fh,  simnum,  zonenum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdDoubleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdDoubleArrayByNum_f(fh,  simnum,  zonenum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdDoubleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      real(selected_real_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdStringByNum_f(fh,  simnum,  zonenum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdStringByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdStringArrayByNum_f(fh,  simnum,  zonenum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdStringArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      character(len=80), dimension(*), intent(inout) :: data
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_zd_by_name_f(fh, simnum, zonenum, name, element_num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_zd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      character(len=80), intent(inout) :: name
!      integer(selected_int_kind(8)), intent(inout) :: element_num
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtf_read_zd_strings_by_name_f(fh, simnum, zonenum, name, element_num, data, datatype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zd_strings_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
      integer(selected_int_kind(8)), intent(inout) :: datatype
    END FUNCTION

    FUNCTION dtfReadZdIntByName_f(fh,  simnum,  zonenum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdIntByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      integer(selected_int_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdIntArrayByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdIntArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdSingleByName_f(fh,  simnum,  zonenum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdSingleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(4)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdSingleArrayByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdSingleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(4)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdDoubleByName_f(fh,  simnum,  zonenum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdDoubleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdDoubleArrayByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdDoubleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdStringByName_f(fh,  simnum,  zonenum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdStringByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadZdStringArrayByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadZdStringArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: data
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_query_zd_minmax_by_name_f(fh, simnum, zonenum, name, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_query_zd_minmax_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      character(len=80), intent(inout) :: name
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtf_delete_all_zd_of_topotype_f(fh, simnum, zonenum, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_all_zd_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtfQueryZdIntMinMaxByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQueryZdIntMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), dimension(2), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfQueryZdSingleMinMaxByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQueryZdSingleMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(4)), dimension(2), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfQueryZdDoubleMinMaxByName_f(fh,  simnum,  zonenum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQueryZdDoubleMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(8)), dimension(2), intent(inout) :: data
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_add_sd_f(fh, simnum, name, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_add_sd_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(in) :: name
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_add_sd_strings_f(fh, simnum, name, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_sd_strings_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdInt_f(fh,  simnum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdInt_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdIntArray_f(fh,  simnum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdIntArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdSingle_f(fh,  simnum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdSingle_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdSingleArray_f(fh,  simnum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdSingleArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdDouble_f(fh,  simnum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdDouble_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdDoubleArray_f(fh,  simnum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdDoubleArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdString_f(fh,  simnum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdString_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfAddSdStringArray_f(fh,  simnum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfAddSdStringArray_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtf_delete_sd_by_num_f(fh, simnum, datanum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_sd_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
    END FUNCTION

    FUNCTION dtf_delete_sd_by_name_f(fh, simnum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_sd_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
    END FUNCTION

    FUNCTION dtf_query_nsds_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nsds_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_sd_by_num_f(fh, simnum, datanum, name, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_sd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: datanum
!      character(len=80), intent(in) :: name
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_update_sd_strings_by_num_f(fh, simnum, datanum, name, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_sd_strings_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdIntByNum_f(fh,  simnum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdIntByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdIntArrayByNum_f(fh,  simnum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdIntArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdSingleByNum_f(fh,  simnum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdSingleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdSingleArrayByNum_f(fh,  simnum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdSingleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdDoubleByNum_f(fh,  simnum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdDoubleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdDoubleArrayByNum_f(fh,  simnum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdDoubleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdStringByNum_f(fh,  simnum,  datanum,  name,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdStringByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdStringArrayByNum_f(fh,  simnum,  datanum,  name,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdStringArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: datanum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_update_sd_by_name_f(fh, simnum, name, newname, n, data, datatype, units, topotype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_sd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(in) :: name
!      character(len=80), intent(in) :: newname
!      integer(selected_int_kind(8)), intent(in) :: n
!      void, intent(in) :: data
!      integer(selected_int_kind(8)), intent(in) :: datatype
!      character(len=32), intent(in) :: units
!      integer(selected_int_kind(8)), intent(in) :: topotype
!    END FUNCTION

    FUNCTION dtf_update_sd_strings_by_name_f(fh, simnum, name, newname, n, data, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_sd_strings_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), intent(in) :: data
      integer(selected_int_kind(8)), intent(in) :: datatype
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdIntByName_f(fh,  simnum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdIntByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdIntArrayByName_f(fh,  simnum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdIntArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      integer(selected_int_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdSingleByName_f(fh,  simnum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdSingleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      real(selected_real_kind(4)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdSingleArrayByName_f(fh,  simnum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdSingleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(4)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdDoubleByName_f(fh,  simnum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdDoubleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      real(selected_real_kind(8)), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdDoubleArrayByName_f(fh,  simnum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdDoubleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      real(selected_real_kind(8)), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdStringByName_f(fh,  simnum,  name,  newname,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdStringByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      character(len=80), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtfUpdateSdStringArrayByName_f(fh,  simnum,  name,  newname,  n,  data,  units,  topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfUpdateSdStringArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: newname
      integer(selected_int_kind(8)), intent(in) :: n
      character(len=80), dimension(*), intent(in) :: data
      character(len=32), intent(in) :: units
      integer(selected_int_kind(8)), intent(in) :: topotype
    END FUNCTION

    FUNCTION dtf_query_nsds_of_topotype_f(fh, simnum, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nsds_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_read_sdnums_of_topotype_f(fh, simnum, topotype, nums)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_sdnums_of_topotype_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: topotype
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: nums
    END FUNCTION

    FUNCTION dtf_query_sd_by_num_f(fh, simnum, datanum, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_sd_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

    FUNCTION dtf_query_sd_by_name_f(fh, simnum, name, n, datatype, units, topotype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_sd_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n
      integer(selected_int_kind(8)), intent(inout) :: datatype
      character(len=32), intent(inout) :: units
      integer(selected_int_kind(8)), intent(inout) :: topotype
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_sd_by_num_f(fh, simnum, datanum, element_num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_sd_by_num_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(inout) :: datanum
!      integer(selected_int_kind(8)), intent(inout) :: element_num
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtf_read_sd_strings_by_num_f(fh, simnum, datanum, element_num, data, datatype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_sd_strings_by_num_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
      integer(selected_int_kind(8)), intent(inout) :: datatype
    END FUNCTION

    FUNCTION dtfReadSdIntByNum_f(fh,  simnum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdIntByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      integer(selected_int_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdIntArrayByNum_f(fh,  simnum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdIntArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdSingleByNum_f(fh,  simnum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdSingleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(4)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdSingleArrayByNum_f(fh,  simnum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdSingleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      real(selected_real_kind(4)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdDoubleByNum_f(fh,  simnum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdDoubleByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdDoubleArrayByNum_f(fh,  simnum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdDoubleArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      real(selected_real_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdStringByNum_f(fh,  simnum,  datanum,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdStringByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdStringArrayByNum_f(fh,  simnum,  datanum,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdStringArrayByNum_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: datanum
      character(len=80), dimension(*), intent(inout) :: data
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_read_sd_by_name_f(fh, simnum, name, element_num, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_read_sd_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(inout) :: name
!      integer(selected_int_kind(8)), intent(inout) :: element_num
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtf_read_sd_strings_by_name_f(fh, simnum, name, element_num, data, datatype)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_sd_strings_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: data
      integer(selected_int_kind(8)), intent(inout) :: element_num
      integer(selected_int_kind(8)), intent(inout) :: datatype
    END FUNCTION

    FUNCTION dtfReadSdIntByName_f(fh,  simnum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdIntByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      integer(selected_int_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdIntArrayByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdIntArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdSingleByName_f(fh,  simnum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdSingleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(4)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdSingleArrayByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdSingleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(4)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdDoubleByName_f(fh,  simnum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdDoubleByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      real(selected_real_kind(8)), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdDoubleArrayByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdDoubleArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(8)), dimension(*), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdStringByName_f(fh,  simnum,  name,  element_num,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdStringByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: element_num
      character(len=80), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfReadSdStringArrayByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfReadSdStringArrayByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: data
    END FUNCTION

! Non-Fortran VOID datatype in arguments
! (See the EXTERNAL declarations at the bottom)
!    FUNCTION dtf_query_sd_minmax_by_name_f(fh, simnum, name, data, datatype)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_query_sd_minmax_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      character(len=80), intent(inout) :: name
!      void, intent(inout) :: data
!      integer(selected_int_kind(8)), intent(inout) :: datatype
!    END FUNCTION

    FUNCTION dtfQuerySdIntMinMaxByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQuerySdIntMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), dimension(2), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfQuerySdSingleMinMaxByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQuerySdSingleMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(4)), dimension(2), intent(inout) :: data
    END FUNCTION

    FUNCTION dtfQuerySdDoubleMinMaxByName_f(fh,  simnum,  name,  data)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtfQuerySdDoubleMinMaxByName_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      character(len=80), intent(inout) :: name
      real(selected_real_kind(8)), dimension(2), intent(inout) :: data
    END FUNCTION

    FUNCTION dtf_query_nbcrecords_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nbcrecords_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_bcrecord_f(fh, simnum, zonenum, bcnum, key, type, name, n_categories, n_bcvals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(inout) :: key
      character(len=80), intent(inout) :: type
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n_categories
      integer(selected_int_kind(8)), intent(inout) :: n_bcvals
    END FUNCTION

    FUNCTION dtf_update_bcrecord_f(fh, simnum, zonenum, bcnum, key, type, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(in) :: key
      character(len=80), intent(in) :: type
      character(len=80), intent(in) :: name
    END FUNCTION

    FUNCTION dtf_copy_bcrecord_f(fh, simnum_from, zonenum_from, bcnum, simnum_to, zonenum_to)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_copy_bcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum_from
      integer(selected_int_kind(8)), intent(in) :: zonenum_from
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(in) :: simnum_to
      integer(selected_int_kind(8)), intent(in) :: zonenum_to
    END FUNCTION

    FUNCTION dtf_delete_bcrecord_f(fh, simnum, zonenum, bcnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_bcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
    END FUNCTION

    FUNCTION dtf_query_bc_category_f(fh, simnum, zonenum, bcnum, catnum, name, value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bc_category_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(inout) :: catnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: value
    END FUNCTION

    FUNCTION dtf_query_bc_category_value_f(fh, simnum, zonenum, bcnum, name, value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bc_category_value_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: value
    END FUNCTION

    FUNCTION dtf_update_bc_categories_f(fh, simnum, zonenum, bcnum, n_categories, name, value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bc_categories_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(in) :: n_categories
      character(len=80), dimension(*), intent(in) :: name
      character(len=80), dimension(*), intent(in) :: value
    END FUNCTION

    FUNCTION dtf_query_bcval_name_f(fh, simnum, zonenum, bcnum, bcvalnum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bcval_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      integer(selected_int_kind(8)), intent(inout) :: bcvalnum
      character(len=80), intent(inout) :: name
    END FUNCTION

    FUNCTION dtf_query_bcval_eval_method_f(fh, simnum, zonenum, bcnum, name, eval_method)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bcval_eval_method_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: eval_method
    END FUNCTION

    FUNCTION dtf_query_bcval_eval_data_f(fh, simnum, zonenum, bcnum, name, nints, nreals, nstrings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_bcval_eval_data_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: nints
      integer(selected_int_kind(8)), intent(inout) :: nreals
      integer(selected_int_kind(8)), intent(inout) :: nstrings
    END FUNCTION

    FUNCTION dtf_read_bcval_eval_data_d_f(fh, simnum, zonenum, bcnum, name, var_ints, ints, var_reals, reals, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_eval_data_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: ints
      character(len=80), dimension(*), intent(inout) :: var_reals
      real(selected_real_kind(8)), dimension(*), intent(inout) :: reals
      character(len=80), dimension(*), intent(inout) :: var_strings
      character(len=80), dimension(*), intent(inout) :: strings
    END FUNCTION

    FUNCTION dtf_read_bcval_eval_data_s_f(fh, simnum, zonenum, bcnum, name, var_ints, ints, var_reals, reals, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_eval_data_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: ints
      character(len=80), dimension(*), intent(inout) :: var_reals
      real(selected_real_kind(4)), dimension(*), intent(inout) :: reals
      character(len=80), dimension(*), intent(inout) :: var_strings
      character(len=80), dimension(*), intent(inout) :: strings
    END FUNCTION

    FUNCTION dtf_read_bcval_int_f(fh, simnum, zonenum, bcnum, name, int_name, int_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_int_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: int_name
      integer(selected_int_kind(8)), intent(inout) :: int_value
    END FUNCTION

    FUNCTION dtf_read_bcval_real_d_f(fh, simnum, zonenum, bcnum, name, real_name, real_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_real_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: real_name
      real(selected_real_kind(8)), intent(inout) :: real_value
    END FUNCTION

    FUNCTION dtf_read_bcval_real_s_f(fh, simnum, zonenum, bcnum, name, real_name, real_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_real_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: real_name
      real(selected_real_kind(4)), intent(inout) :: real_value
    END FUNCTION

    FUNCTION dtf_read_bcval_string_f(fh, simnum, zonenum, bcnum, name, string_name, string_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_bcval_string_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: string_name
      character(len=80), intent(inout) :: string_value
    END FUNCTION

    FUNCTION dtf_update_bcval_f(fh, simnum, zonenum, bcnum, name, eval_method)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: eval_method
    END FUNCTION

    FUNCTION dtf_update_bcval_ints_f(fh, simnum, zonenum, bcnum, name, nints, var_ints, ints)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_ints_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nints
      character(len=80), dimension(*), intent(in) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(in) :: ints
    END FUNCTION

    FUNCTION dtf_update_bcval_reals_d_f(fh, simnum, zonenum, bcnum, name, nreals, var_reals, reals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_reals_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nreals
      character(len=80), dimension(*), intent(in) :: var_reals
      real(selected_real_kind(8)), dimension(*), intent(in) :: reals
    END FUNCTION

    FUNCTION dtf_update_bcval_reals_s_f(fh, simnum, zonenum, bcnum, name, nreals, var_reals, reals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_reals_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nreals
      character(len=80), dimension(*), intent(in) :: var_reals
      real(selected_real_kind(4)), dimension(*), intent(in) :: reals
    END FUNCTION

    FUNCTION dtf_update_bcval_strings_f(fh, simnum, zonenum, bcnum, name, nstrings, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_strings_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nstrings
      character(len=80), dimension(*), intent(in) :: var_strings
      character(len=80), dimension(*), intent(in) :: strings
    END FUNCTION

    FUNCTION dtf_update_bcval_int_by_name_f(fh, simnum, zonenum, bcnum, val_name, elem_name, elem_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_bcval_int_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(in) :: val_name
      character(len=80), intent(in) :: elem_name
      integer(selected_int_kind(8)), intent(in) :: elem_value
    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_bcval_real_d_by_name_f(fh, simnum, zonenum, bcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_bcval_real_d_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: bcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      real(selected_real_kind(8)), intent(in) :: elem_value
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_bcval_real_s_by_name_f(fh, simnum, zonenum, bcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_bcval_real_s_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: bcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      real(selected_real_kind(4)), intent(in) :: elem_value
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_bcval_string_by_name_f(fh, simnum, zonenum, bcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_bcval_string_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: bcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      character(len=80), intent(in) :: elem_value
!    END FUNCTION

    FUNCTION dtf_delete_bcval_f(fh, simnum, zonenum, bcnum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_bcval_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
      character(len=80), intent(inout) :: name
    END FUNCTION

    FUNCTION dtf_delete_all_bcvals_f(fh, simnum, zonenum, bcnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_all_bcvals_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: bcnum
    END FUNCTION

    FUNCTION dtf_query_npatches_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_npatches_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_patch_f(fh, simnum, zonenum, patchnum, imin, imax, jmin, jmax, kmin, kmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_patch_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: patchnum
      integer(selected_int_kind(8)), intent(inout) :: imin
      integer(selected_int_kind(8)), intent(inout) :: imax
      integer(selected_int_kind(8)), intent(inout) :: jmin
      integer(selected_int_kind(8)), intent(inout) :: jmax
      integer(selected_int_kind(8)), intent(inout) :: kmin
      integer(selected_int_kind(8)), intent(inout) :: kmax
    END FUNCTION

    FUNCTION dtf_read_patch_f(fh, simnum, zonenum, patchnum, records)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_patch_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: patchnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: records
    END FUNCTION

    FUNCTION dtf_update_patch_f(fh, simnum, zonenum, patchnum, imin, imax, jmin, jmax, kmin, kmax, records)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_patch_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: patchnum
      integer(selected_int_kind(8)), intent(in) :: imin
      integer(selected_int_kind(8)), intent(in) :: imax
      integer(selected_int_kind(8)), intent(in) :: jmin
      integer(selected_int_kind(8)), intent(in) :: jmax
      integer(selected_int_kind(8)), intent(in) :: kmin
      integer(selected_int_kind(8)), intent(in) :: kmax
      integer(selected_int_kind(8)), dimension(*), intent(in) :: records
    END FUNCTION

    FUNCTION dtf_delete_patch_f(fh, simnum, zonenum, patchnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_patch_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: patchnum
    END FUNCTION

    FUNCTION dtf_query_nrecords_in_patch_f(imin, imax, jmin, jmax, kmin, kmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nrecords_in_patch_f
      integer(selected_int_kind(8)), intent(inout) :: imin
      integer(selected_int_kind(8)), intent(inout) :: imax
      integer(selected_int_kind(8)), intent(inout) :: jmin
      integer(selected_int_kind(8)), intent(inout) :: jmax
      integer(selected_int_kind(8)), intent(inout) :: kmin
      integer(selected_int_kind(8)), intent(inout) :: kmax
    END FUNCTION

    FUNCTION dtf_query_nface_groups_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nface_groups_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_face_group_f(fh, simnum, zonenum, face_groupnum, key)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_face_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: face_groupnum
      integer(selected_int_kind(8)), intent(inout) :: key
    END FUNCTION

    FUNCTION dtf_read_face_group_f(fh, simnum, zonenum, face_groupnum, faces)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_face_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: face_groupnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: faces
    END FUNCTION

    FUNCTION dtf_query_nsurface_conditions_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nsurface_conditions_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_surface_condition_f(fh, simnum, zonenum, surface_conditionnum, sc_group_num, bc_record_num)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_surface_condition_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: surface_conditionnum
      integer(selected_int_kind(8)), intent(inout) :: sc_group_num
      integer(selected_int_kind(8)), intent(inout) :: bc_record_num
    END FUNCTION

    FUNCTION dtf_query_nvcrecords_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nvcrecords_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_vcrecord_f(fh, simnum, zonenum, vcnum, category, name, n_vcvals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: category
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: n_vcvals
    END FUNCTION

    FUNCTION dtf_update_vcrecord_f(fh, simnum, zonenum, vcnum, category, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: category
      character(len=80), intent(in) :: name
    END FUNCTION

    FUNCTION dtf_copy_vcrecord_f(fh, simnum_from, zonenum_from, vcnum, simnum_to, zonenum_to)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_copy_vcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum_from
      integer(selected_int_kind(8)), intent(in) :: zonenum_from
      integer(selected_int_kind(8)), intent(in) :: vcnum
      integer(selected_int_kind(8)), intent(in) :: simnum_to
      integer(selected_int_kind(8)), intent(in) :: zonenum_to
    END FUNCTION

    FUNCTION dtf_delete_vcrecord_f(fh, simnum, zonenum, vcnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_vcrecord_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
    END FUNCTION

    FUNCTION dtf_query_vcval_name_f(fh, simnum, zonenum, vcnum, vcvalnum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vcval_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      integer(selected_int_kind(8)), intent(inout) :: vcvalnum
      character(len=80), intent(inout) :: name
    END FUNCTION

    FUNCTION dtf_query_vcval_eval_method_f(fh, simnum, zonenum, vcnum, name, eval_method)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vcval_eval_method_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: eval_method
    END FUNCTION

    FUNCTION dtf_query_vcval_eval_data_f(fh, simnum, zonenum, vcnum, name, nints, nreals, nstrings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_vcval_eval_data_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      integer(selected_int_kind(8)), intent(inout) :: nints
      integer(selected_int_kind(8)), intent(inout) :: nreals
      integer(selected_int_kind(8)), intent(inout) :: nstrings
    END FUNCTION

    FUNCTION dtf_read_vcval_eval_data_d_f(fh, simnum, zonenum, vcnum, name, var_ints, ints, var_reals, reals, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_eval_data_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: ints
      character(len=80), dimension(*), intent(inout) :: var_reals
      real(selected_real_kind(8)), dimension(*), intent(inout) :: reals
      character(len=80), dimension(*), intent(inout) :: var_strings
      character(len=80), dimension(*), intent(inout) :: strings
    END FUNCTION

    FUNCTION dtf_read_vcval_eval_data_s_f(fh, simnum, zonenum, vcnum, name, var_ints, ints, var_reals, reals, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_eval_data_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), dimension(*), intent(inout) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: ints
      character(len=80), dimension(*), intent(inout) :: var_reals
      real(selected_real_kind(4)), dimension(*), intent(inout) :: reals
      character(len=80), dimension(*), intent(inout) :: var_strings
      character(len=80), dimension(*), intent(inout) :: strings
    END FUNCTION

    FUNCTION dtf_read_vcval_int_f(fh, simnum, zonenum, vcnum, name, int_name, int_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_int_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: int_name
      integer(selected_int_kind(8)), intent(inout) :: int_value
    END FUNCTION

    FUNCTION dtf_read_vcval_real_d_f(fh, simnum, zonenum, vcnum, name, real_name, real_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_real_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: real_name
      real(selected_real_kind(8)), intent(inout) :: real_value
    END FUNCTION

    FUNCTION dtf_read_vcval_real_s_f(fh, simnum, zonenum, vcnum, name, real_name, real_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_real_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: real_name
      real(selected_real_kind(4)), intent(inout) :: real_value
    END FUNCTION

    FUNCTION dtf_read_vcval_string_f(fh, simnum, zonenum, vcnum, name, string_name, string_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_vcval_string_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
      character(len=80), intent(inout) :: string_name
      character(len=80), intent(inout) :: string_value
    END FUNCTION

    FUNCTION dtf_update_vcval_f(fh, simnum, zonenum, vcnum, name, eval_method)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: name
      character(len=80), intent(in) :: eval_method
    END FUNCTION

    FUNCTION dtf_update_vcval_ints_f(fh, simnum, zonenum, vcnum, name, nints, var_ints, ints)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_ints_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nints
      character(len=80), dimension(*), intent(in) :: var_ints
      integer(selected_int_kind(8)), dimension(*), intent(in) :: ints
    END FUNCTION

    FUNCTION dtf_update_vcval_reals_d_f(fh, simnum, zonenum, vcnum, name, nreals, var_reals, reals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_reals_d_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nreals
      character(len=80), dimension(*), intent(in) :: var_reals
      real(selected_real_kind(8)), dimension(*), intent(in) :: reals
    END FUNCTION

    FUNCTION dtf_update_vcval_reals_s_f(fh, simnum, zonenum, vcnum, name, nreals, var_reals, reals)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_reals_s_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nreals
      character(len=80), dimension(*), intent(in) :: var_reals
      real(selected_real_kind(4)), dimension(*), intent(in) :: reals
    END FUNCTION

    FUNCTION dtf_update_vcval_strings_f(fh, simnum, zonenum, vcnum, name, nstrings, var_strings, strings)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_strings_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: name
      integer(selected_int_kind(8)), intent(in) :: nstrings
      character(len=80), dimension(*), intent(in) :: var_strings
      character(len=80), dimension(*), intent(in) :: strings
    END FUNCTION

    FUNCTION dtf_update_vcval_int_by_name_f(fh, simnum, zonenum, vcnum, val_name, elem_name, elem_value)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_vcval_int_by_name_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(in) :: val_name
      character(len=80), intent(in) :: elem_name
      integer(selected_int_kind(8)), intent(in) :: elem_value
    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_vcval_real_d_by_name_f(fh, simnum, zonenum, vcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_vcval_real_d_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: vcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      real(selected_real_kind(8)), intent(in) :: elem_value
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_vcval_real_s_by_name_f(fh, simnum, zonenum, vcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_vcval_real_s_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: vcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      real(selected_real_kind(4)), intent(in) :: elem_value
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_update_vcval_string_by_name_f(fh, simnum, zonenum, vcnum, val_name, elem_name, elem_value)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_update_vcval_string_by_name_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: simnum
!      integer(selected_int_kind(8)), intent(in) :: zonenum
!      integer(selected_int_kind(8)), intent(in) :: vcnum
!      character(len=80), intent(in) :: val_name
!      character(len=80), intent(in) :: elem_name
!      character(len=80), intent(in) :: elem_value
!    END FUNCTION

    FUNCTION dtf_delete_vcval_f(fh, simnum, zonenum, vcnum, name)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_vcval_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
      character(len=80), intent(inout) :: name
    END FUNCTION

    FUNCTION dtf_delete_all_vcvals_f(fh, simnum, zonenum, vcnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_all_vcvals_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: vcnum
    END FUNCTION

    FUNCTION dtf_query_nblocks_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nblocks_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_block_f(fh, simnum, zonenum, blocknum, key, imin, imax, jmin, jmax, kmin, kmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_block_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: blocknum
      integer(selected_int_kind(8)), intent(inout) :: key
      integer(selected_int_kind(8)), intent(inout) :: imin
      integer(selected_int_kind(8)), intent(inout) :: imax
      integer(selected_int_kind(8)), intent(inout) :: jmin
      integer(selected_int_kind(8)), intent(inout) :: jmax
      integer(selected_int_kind(8)), intent(inout) :: kmin
      integer(selected_int_kind(8)), intent(inout) :: kmax
    END FUNCTION

    FUNCTION dtf_update_block_f(fh, simnum, zonenum, blocknum, key, imin, imax, jmin, jmax, kmin, kmax)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_block_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: blocknum
      integer(selected_int_kind(8)), intent(in) :: key
      integer(selected_int_kind(8)), intent(in) :: imin
      integer(selected_int_kind(8)), intent(in) :: imax
      integer(selected_int_kind(8)), intent(in) :: jmin
      integer(selected_int_kind(8)), intent(in) :: jmax
      integer(selected_int_kind(8)), intent(in) :: kmin
      integer(selected_int_kind(8)), intent(in) :: kmax
    END FUNCTION

    FUNCTION dtf_delete_block_f(fh, simnum, zonenum, blocknum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_block_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: blocknum
    END FUNCTION

    FUNCTION dtf_query_ncell_groups_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_ncell_groups_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_cell_group_f(fh, simnum, zonenum, cell_groupnum, key)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_cell_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cell_groupnum
      integer(selected_int_kind(8)), intent(inout) :: key
    END FUNCTION

    FUNCTION dtf_read_cell_group_f(fh, simnum, zonenum, cell_groupnum, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_cell_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cell_groupnum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: cells
    END FUNCTION

    FUNCTION dtf_update_cell_group_f(fh, simnum, zonenum, cell_groupnum, key, ncells, cells)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_cell_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: cell_groupnum
      integer(selected_int_kind(8)), intent(in) :: key
      integer(selected_int_kind(8)), intent(in) :: ncells
      integer(selected_int_kind(8)), dimension(*), intent(in) :: cells
    END FUNCTION

    FUNCTION dtf_delete_cell_group_f(fh, simnum, zonenum, cell_groupnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_cell_group_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: cell_groupnum
    END FUNCTION

    FUNCTION dtf_query_nvolume_conditions_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nvolume_conditions_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_volume_condition_f(fh, simnum, zonenum, volume_conditionnum, vc_group_num, vc_record_num)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_volume_condition_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: volume_conditionnum
      integer(selected_int_kind(8)), intent(inout) :: vc_group_num
      integer(selected_int_kind(8)), intent(inout) :: vc_record_num
    END FUNCTION

    FUNCTION dtf_update_volume_condition_f(fh, simnum, zonenum, volume_conditionnum, vc_group_num, vc_record_num)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_update_volume_condition_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(in) :: volume_conditionnum
      integer(selected_int_kind(8)), intent(in) :: vc_group_num
      integer(selected_int_kind(8)), intent(in) :: vc_record_num
    END FUNCTION

    FUNCTION dtf_delete_volume_condition_f(fh, simnum, zonenum, volume_conditionnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_delete_volume_condition_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: volume_conditionnum
    END FUNCTION

    FUNCTION dtf_check_rep_dup_nodes_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_check_rep_dup_nodes_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_add_p3d_to_sim_f(fh, simnum, filenamePFG, has_blanking, nzones_added)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_p3d_to_sim_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: has_blanking
      integer(selected_int_kind(8)), intent(inout) :: nzones_added
      character(len=80), intent(inout) :: filenamePFG
    END FUNCTION

    FUNCTION dtf_strcasecmp_f(s1, s2)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_strcasecmp_f
      character(len=80), intent(inout) :: s1
      character(len=80), intent(inout) :: s2
    END FUNCTION

    FUNCTION dtf_query_nzi_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzi_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

    FUNCTION dtf_query_zi_f(fh, simnum, zinum, zone_L, zone_R, nfaces)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zi_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), intent(inout) :: zone_L
      integer(selected_int_kind(8)), intent(inout) :: zone_R
      integer(selected_int_kind(8)), intent(inout) :: nfaces
    END FUNCTION

    FUNCTION dtf_read_zi_f(fh, simnum, zinum, facenum_L, facenum_R)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zi_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_R
    END FUNCTION

    FUNCTION dtf_query_nzi_zone_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzi_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_zi_zone_f(fh, simnum, zonenum, zinum, zone_L, zone_R, nfaces)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zi_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), intent(inout) :: zone_L
      integer(selected_int_kind(8)), intent(inout) :: zone_R
      integer(selected_int_kind(8)), intent(inout) :: nfaces
    END FUNCTION

    FUNCTION dtf_read_zi_zone_f(fh, simnum, zonenum, zinum, facenum_L, facenum_R)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zi_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_R
    END FUNCTION

    FUNCTION dtf_query_nzi_ss_f(fh, simnum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzi_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
    END FUNCTION

    FUNCTION dtf_query_zi_ss_f(fh, simnum, zinum, zone_L, zone_R, nfaces)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zi_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), intent(inout) :: zone_L
      integer(selected_int_kind(8)), intent(inout) :: zone_R
      integer(selected_int_kind(8)), intent(inout) :: nfaces
    END FUNCTION

    FUNCTION dtf_read_zi_ss_f(fh, simnum, zinum, facenum_L, facenum_R, patchnum_L, patchnum_R)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zi_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_R
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: patchnum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: patchnum_R
    END FUNCTION

    FUNCTION dtf_query_nzi_zone_ss_f(fh, simnum, zonenum)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_nzi_zone_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
    END FUNCTION

    FUNCTION dtf_query_zi_zone_ss_f(fh, simnum, zonenum, zinum, zone_L, zone_R, nfaces)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_query_zi_zone_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), intent(inout) :: zone_L
      integer(selected_int_kind(8)), intent(inout) :: zone_R
      integer(selected_int_kind(8)), intent(inout) :: nfaces
    END FUNCTION

    FUNCTION dtf_read_zi_zone_ss_f(fh, simnum, zonenum, zinum, facenum_L, facenum_R, patchnum_L, patchnum_R)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_read_zi_zone_ss_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: simnum
      integer(selected_int_kind(8)), intent(in) :: zonenum
      integer(selected_int_kind(8)), intent(inout) :: zinum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: facenum_R
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: patchnum_L
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: patchnum_R
    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_clear_file_connectivity_table_f(fh)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_clear_file_connectivity_table_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_clear_sim_connectivity_table_f(fh, sim)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_clear_sim_connectivity_table_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: sim
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_clear_zone_connectivity_table_f(fh, sim, zone)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_clear_zone_connectivity_table_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: sim
!      integer(selected_int_kind(8)), intent(in) :: zone
!    END FUNCTION

    FUNCTION dtf_flush_file_cache_f(fh)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_flush_file_cache_f
      integer(selected_int_kind(8)), intent(in) :: fh
    END FUNCTION

    FUNCTION dtf_set_destruct_mode_f(fh, sim, zone, mode)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_set_destruct_mode_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
      integer(selected_int_kind(8)), intent(inout) :: mode
    END FUNCTION

    FUNCTION dtf_get_destruct_mode_f(fh, sim, zone)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_get_destruct_mode_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
    END FUNCTION

    FUNCTION dtf_map_nodes_f(xmap, ymap, zmap, nnodes_map, nodenum, map, tolerance)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_map_nodes_f
      real(selected_real_kind(8)), dimension(*), intent(inout) :: xmap
      real(selected_real_kind(8)), dimension(*), intent(inout) :: ymap
      real(selected_real_kind(8)), dimension(*), intent(inout) :: zmap
      integer(selected_int_kind(8)), intent(inout) :: nnodes_map
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: nodenum
      integer(selected_int_kind(8)), dimension(*), intent(inout) :: map
      real(selected_real_kind(8)), intent(inout) :: tolerance
    END FUNCTION

    FUNCTION dtf_test_validity_f(filename, sim)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_test_validity_f
      character(len=80), intent(inout) :: filename
      integer(selected_int_kind(8)), intent(in) :: sim
    END FUNCTION

    FUNCTION dtf_create_empty_poly_zone_f( fh,  sim,  nnodes,  n_faces_total,  is_2D)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_create_empty_poly_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(inout) :: nnodes
      integer(selected_int_kind(8)), intent(inout) :: n_faces_total
      integer(selected_int_kind(8)), intent(inout) :: is_2D
    END FUNCTION

    FUNCTION dtf_add_x_to_empty_poly_zone_f( fh,  sim,  zone,  x)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_x_to_empty_poly_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
      real(selected_real_kind(8)), dimension(*), intent(in) :: x
    END FUNCTION

    FUNCTION dtf_add_y_to_empty_poly_zone_f( fh,  sim,  zone,  y)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_y_to_empty_poly_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
      real(selected_real_kind(8)), dimension(*), intent(in) :: y
    END FUNCTION

    FUNCTION dtf_add_z_to_empty_poly_zone_f( fh,  sim,  zone,  z)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_add_z_to_empty_poly_zone_f
      integer(selected_int_kind(8)), intent(in) :: fh
      integer(selected_int_kind(8)), intent(in) :: sim
      integer(selected_int_kind(8)), intent(in) :: zone
      real(selected_real_kind(8)), dimension(*), intent(in) :: z
    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_add_f2n_to_empty_poly_zone_f( fh,  sim,  zone,  len_f2n,  f2n)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_add_f2n_to_empty_poly_zone_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: sim
!      integer(selected_int_kind(8)), intent(in) :: zone
!      integer(selected_int_kind(8)), intent(in) :: len_f2n
!      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2n
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_add_n_nodes_per_face_to_empty_poly_zone_f( fh,  sim,  zone,  n_nodes_per_face)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_add_n_nodes_per_face_to_empty_poly_zone_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: sim
!      integer(selected_int_kind(8)), intent(in) :: zone
!      integer(selected_int_kind(8)), dimension(*), intent(in) :: n_nodes_per_face
!    END FUNCTION

! Non-standard function name (length > 31)
! Uncomment this interface if your compiler supports the name length
!    FUNCTION dtf_add_f2c_to_empty_poly_zone_f( fh,  sim,  zone,  f2c)
!      IMPLICIT NONE
!      integer(selected_int_kind(8)) :: dtf_add_f2c_to_empty_poly_zone_f
!      integer(selected_int_kind(8)), intent(in) :: fh
!      integer(selected_int_kind(8)), intent(in) :: sim
!      integer(selected_int_kind(8)), intent(in) :: zone
!      integer(selected_int_kind(8)), dimension(*), intent(in) :: f2c
!    END FUNCTION

    FUNCTION dtf_perturb_node_f(px, plev0)
      IMPLICIT NONE
      integer(selected_int_kind(8)) :: dtf_perturb_node_f
      real(selected_real_kind(8)), dimension(*), intent(inout) :: px
      integer(selected_int_kind(8)), intent(inout) :: plev0
    END FUNCTION


  END INTERFACE

  ! Non-conforming API (long function names, void* arguments, etc)
  ! The EXTERNAL declaration is obsoleted by the F90 standard.
  ! Uncomment any of these lines if your compiler supports EXTERNAL.
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_struct_zone_virtual_nodenums_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_vzd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_vzd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_vzd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_vzd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_add_zd_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_zd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_zd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_zd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_zd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_query_zd_minmax_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_add_sd_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_sd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_sd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_sd_by_num_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_read_sd_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_query_sd_minmax_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_bcval_real_d_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_bcval_real_s_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_bcval_string_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_vcval_real_d_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_vcval_real_s_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_update_vcval_string_by_name_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_clear_file_connectivity_table_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_clear_sim_connectivity_table_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_clear_zone_connectivity_table_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_add_f2n_to_empty_poly_zone_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_add_n_nodes_per_face_to_empty_poly_zone_f
  ! integer(selected_int_kind(8)), EXTERNAL :: dtf_add_f2c_to_empty_poly_zone_f

END MODULE DTF_protos

