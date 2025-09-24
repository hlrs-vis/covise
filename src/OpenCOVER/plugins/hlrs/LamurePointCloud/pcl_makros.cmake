
macro(pcl_project_files)
    set(out_file_list       ${ARGV0})
    set(search_dir          ${ARGV1})
    set(in_glob_expr        ${ARGV})

    get_filename_component(src_path     ${SRC_DIR} ABSOLUTE)
    get_filename_component(search_path  ${search_dir} ABSOLUTE)

    #message(${src_path})
    #message(${search_path})

    # remove the leading output and input variable
    list(REMOVE_AT in_glob_expr 0 1)
    set(out_proj_files    "")

    set(SOURCE_GROUP_DELIMITER "/")
    

    foreach(glob_expr ${in_glob_expr})
        file(GLOB proj_files ${search_path}/${glob_expr})
        foreach(proj_file ${proj_files})
            file(RELATIVE_PATH rel_path ${src_path} ${proj_file})
            if (rel_path)
                file(TO_CMAKE_PATH ${rel_path} rel_path)
            endif (rel_path)

            # message(${rel_path} " " ${proj_file})

            list(APPEND out_proj_files ${proj_file})

        endforeach(proj_file)
    endforeach(glob_expr)

    list(APPEND ${out_file_list} ${out_proj_files})

endmacro(pcl_project_files)