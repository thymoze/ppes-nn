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
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
endfunction()

if(BUILD_PICO)
  add_subdirectory(pico)
else()
  ADD_COMPILE_LINK_EXECUTABLE(main main.cpp)
  ADD_COMPILE_LINK_EXECUTABLE(main_pretrained main_pretrained.cpp)

  ADD_COMPILE_LINK_EXECUTABLE(mnist_train mnist_train.cpp)
  ADD_COMPILE_LINK_EXECUTABLE(mnist_pretrained mnist_pretrained.cpp)

  ADD_COMPILE_LINK_EXECUTABLE(mnist_bin mnist_bin.cpp)
  link_binary(mnist_bin t50-images-idx3-ubyte t50-labels-idx1-ubyte)
endif()
