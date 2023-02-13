function(link_binary target)
  add_custom_command(
    OUTPUT
      ${target}.bin.o
    WORKING_DIRECTORY
      ${PROJECT_SOURCE_DIR}/data
    COMMAND
      ${CMAKE_LINKER} -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/${target}.bin.o t50-images-idx3-ubyte t50-labels-idx1-ubyte
    #  COMMAND
    #    objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents ${CMAKE_CURRENT_BINARY_DIR}/template.o ${CMAKE_CURRENT_BINARY_DIR}/template.o
  )
  add_library(${target}_bin
    STATIC
      ${target}.bin.o
  )

  set_source_files_properties(${target}.bin.o
    PROPERTIES
      EXTERNAL_OBJECT true
      GENERATED true
  )

  set_target_properties(${target}_bin
    PROPERTIES
      LINKER_LANGUAGE C
  )

  target_link_libraries(${target} PRIVATE ${target}_bin)
endfunction()
