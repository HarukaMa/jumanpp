function(JPP_PROTOBUF_COMPILE_CPP SRCS HDRS)
  cmake_parse_arguments(protobuf "" "EXPORT_MACRO;DESCRIPTORS;OUTPUT_DIR" "" ${ARGN})

  set(PROTO_FILES "${protobuf_UNPARSED_ARGUMENTS}")
  if(NOT PROTO_FILES)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(protobuf_EXPORT_MACRO)
    set(DLL_EXPORT_DECL "dllexport_decl=${protobuf_EXPORT_MACRO}:")
  endif()

  if (NOT protobuf_OUTPUT_DIR)
    set(protobuf_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  else()
    file(MAKE_DIRECTORY ${protobuf_OUTPUT_DIR})
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${PROTO_FILES})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  if (protobuf_DESCRIPTORS)
    set(${protobuf_DESCRIPTORS})
  endif()

  foreach(FIL ${PROTO_FILES})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
      get_filename_component(FIL_DIR ${FIL} DIRECTORY)
      if(FIL_DIR)
        set(FIL_WE "${FIL_DIR}/${FIL_WE}")
      endif()
    endif()

    set(_protobuf_protoc_src "${protobuf_OUTPUT_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${protobuf_OUTPUT_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    if(protobuf_DESCRIPTORS)
      set(_protobuf_protoc_desc "${protobuf_OUTPUT_DIR}/${FIL_WE}.desc")
      set(_protobuf_protoc_flags "--descriptor_set_out=${_protobuf_protoc_desc}")
      list(APPEND ${protobuf_DESCRIPTORS} "${_protobuf_protoc_desc}")
    else()
      set(_protobuf_protoc_desc "")
      set(_protobuf_protoc_flags "")
    endif()

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
      "${_protobuf_protoc_hdr}"
      ${_protobuf_protoc_desc}
      COMMAND  protobuf::protoc
      "--cpp_out=${DLL_EXPORT_DECL}${protobuf_OUTPUT_DIR}"
      ${_protobuf_protoc_flags}
      ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} protobuf::protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
  set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
  if(protobuf_DESCRIPTORS)
    set(${protobuf_DESCRIPTORS} "${${protobuf_DESCRIPTORS}}" PARENT_SCOPE)
  endif()
endfunction()