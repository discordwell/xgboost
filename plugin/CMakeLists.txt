if(PLUGIN_SYCL)
  set(CMAKE_CXX_COMPILER "icpx")
  file(GLOB_RECURSE SYCL_SOURCES "sycl/*.cc")
    list(APPEND SYCL_SOURCES
    ${xgboost_SOURCE_DIR}/src/objective/regression_obj.cc
    ${xgboost_SOURCE_DIR}/src/objective/hinge.cc
    ${xgboost_SOURCE_DIR}/src/objective/quantile_obj.cc
    ${xgboost_SOURCE_DIR}/src/objective/multiclass_obj.cc)
  add_library(plugin_sycl OBJECT ${SYCL_SOURCES})
  target_include_directories(plugin_sycl
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)
    target_compile_definitions(plugin_sycl PUBLIC -DXGBOOST_USE_SYCL=1)
    target_link_libraries(plugin_sycl PUBLIC -fsycl)
    set_target_properties(plugin_sycl PROPERTIES
    COMPILE_FLAGS "-fsycl -fno-sycl-id-queries-fit-in-int"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set_target_properties(plugin_sycl PROPERTIES
    COMPILE_FLAGS "-fsycl -fno-sycl-id-queries-fit-in-int -qopenmp")
  endif()
  # Get compilation and link flags of plugin_sycl and propagate to objxgboost
  target_link_libraries(objxgboost PUBLIC plugin_sycl)
  # Add all objects of plugin_sycl to objxgboost
  target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:plugin_sycl>)
endif()

if(PLUGIN_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "PLUGIN_METAL is only supported on macOS with Apple Silicon")
  endif()

  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

  file(GLOB_RECURSE METAL_SOURCES "metal/*.mm")
  add_library(plugin_metal OBJECT ${METAL_SOURCES})
  target_include_directories(plugin_metal
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)
  target_compile_definitions(plugin_metal PUBLIC -DXGBOOST_USE_METAL=1)
  set_target_properties(plugin_metal PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  set_source_files_properties(${METAL_SOURCES} PROPERTIES
    LANGUAGE OBJCXX
    COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(plugin_metal PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

  # Compile Metal kernels into metallib (required, like nvcc for CUDA)
  set(METAL_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels")
  file(GLOB METAL_KERNELS "${METAL_KERNEL_DIR}/*.metal")
  set(METAL_AIR_FILES "")
  foreach(metal_src IN LISTS METAL_KERNELS)
    get_filename_component(metal_name "${metal_src}" NAME_WE)
    set(air_file "${CMAKE_BINARY_DIR}/${metal_name}.air")
    add_custom_command(
      OUTPUT "${air_file}"
      COMMAND xcrun -sdk macosx metal -c "${metal_src}" -o "${air_file}"
      DEPENDS "${metal_src}"
      COMMENT "Compiling Metal kernel ${metal_name}.metal")
    list(APPEND METAL_AIR_FILES "${air_file}")
  endforeach()
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/xgboost.metallib"
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o "${CMAKE_BINARY_DIR}/xgboost.metallib"
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking Metal kernels into xgboost.metallib")
  add_custom_target(metal_kernels ALL DEPENDS "${CMAKE_BINARY_DIR}/xgboost.metallib")
  add_dependencies(plugin_metal metal_kernels)
  install(FILES "${CMAKE_BINARY_DIR}/xgboost.metallib" DESTINATION lib)

  # Link plugin into main XGBoost library
  target_link_libraries(objxgboost PUBLIC plugin_metal)
  target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:plugin_metal>)
endif()

# Add the Federate Learning plugin if enabled.
if(PLUGIN_FEDERATED)
  add_subdirectory(federated)
endif()
