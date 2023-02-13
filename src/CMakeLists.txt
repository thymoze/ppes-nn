function(ADD_COMPILE_LINK_EXECUTABLE EXE_NAME MAIN_FILE)
  add_executable(${EXE_NAME} ${MAIN_FILE})
  target_link_libraries(${EXE_NAME} PRIVATE nn)
  target_compile_options(${EXE_NAME} 
    PRIVATE 
      -Wall -Wextra -Wpedantic
      $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
  )
  target_link_options(${EXE_NAME}
    PRIVATE
      -pthread
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
endfunction()

if(BUILD_PICO)
  add_subdirectory(pico)
else()
  ADD_COMPILE_LINK_EXECUTABLE(xor xor.cpp)
  ADD_COMPILE_LINK_EXECUTABLE(xor_pretrained xor_pretrained.cpp)

  ADD_COMPILE_LINK_EXECUTABLE(mnist_train mnist_train.cpp)
  ADD_COMPILE_LINK_EXECUTABLE(mnist_pretrained mnist_pretrained.cpp)

  ADD_COMPILE_LINK_EXECUTABLE(mnist_bin mnist_bin.cpp)
  # Exclude as this requires data files to be present
  set_target_properties(mnist_bin PROPERTIES EXCLUDE_FROM_ALL=true)

  ADD_COMPILE_LINK_EXECUTABLE(main_pruning main_pruning.cpp)
  link_binary(mnist_bin t50-images-idx3-ubyte t50-labels-idx1-ubyte)
endif()
