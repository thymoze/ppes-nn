function(ADD_COMPILE_LINK_EXECUTABLE EXE_NAME MAIN_FILE)
  add_executable(${EXE_NAME} ${MAIN_FILE})
  target_link_libraries(${EXE_NAME} nn)
  target_compile_options(${EXE_NAME} 
    PRIVATE 
      -Wall -Wextra -Wpedantic -std=c++20 -g -Og -fsanitize=address -fsanitize=undefined
  )
  target_link_options(${EXE_NAME}
    PRIVATE
      -fsanitize=address -fsanitize=undefined
  )
endfunction()

ADD_COMPILE_LINK_EXECUTABLE(main main.cpp)
ADD_COMPILE_LINK_EXECUTABLE(main_pretrained main_pretrained.cpp)
ADD_COMPILE_LINK_EXECUTABLE(mnist mnist.cpp)
ADD_COMPILE_LINK_EXECUTABLE(leak leak.cpp)

if(BUILD_PICO)
  add_subdirectory(pico)
endif()
