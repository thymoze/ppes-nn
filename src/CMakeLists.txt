function(ADD_COMPILE_LINK_EXECUTABLE EXE_NAME MAIN_FILE)
  add_executable(${EXE_NAME} ${MAIN_FILE})
  target_link_libraries(${EXE_NAME} PRIVATE nn)
  target_compile_options(${EXE_NAME} 
    PRIVATE 
      -Wall -Wextra -Wpedantic -std=c++17 -g -O3 #-fsanitize=address -fsanitize=undefined
  )
  target_link_options(${EXE_NAME}
    PRIVATE
      #-fsanitize=address -fsanitize=undefined
  )
endfunction()

ADD_COMPILE_LINK_EXECUTABLE(main main.cpp)
ADD_COMPILE_LINK_EXECUTABLE(main_pretrained main_pretrained.cpp)
ADD_COMPILE_LINK_EXECUTABLE(mnist mnist.cpp)
ADD_COMPILE_LINK_EXECUTABLE(mnist_bin mnist_bin.cpp)

add_custom_command(
  OUTPUT
    mnist_train.o
  WORKING_DIRECTORY
    ${PROJECT_SOURCE_DIR}/data
  COMMAND
    ld -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/mnist_train.o train-images-idx3-ubyte train-labels-idx1-ubyte
#  COMMAND
#    objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents ${CMAKE_CURRENT_BINARY_DIR}/template.o ${CMAKE_CURRENT_BINARY_DIR}/template.o
)
add_library(mnist_train
  STATIC
    mnist_train.o
)

set_source_files_properties(mnist_train.o
  PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true
)

set_target_properties(mnist_train
  PROPERTIES
    LINKER_LANGUAGE C
)

target_link_libraries(mnist_bin PRIVATE mnist_train)

ADD_COMPILE_LINK_EXECUTABLE(tensor tensor.cpp)
ADD_COMPILE_LINK_EXECUTABLE(mnist_pretrained mnist_pretrained.cpp)

if(BUILD_PICO)
  add_subdirectory(pico)
endif()
