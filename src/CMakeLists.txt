add_executable(main 
    main.cpp)

add_executable(main_pretrained
    main_pretrained.cpp)

add_executable(mnist 
    mnist.cpp)

# target_include_directories(main
#     PRIVATE
#         src
# )

target_link_libraries(main nn)
target_link_libraries(main_pretrained nn)
target_link_libraries(mnist nn)

if(BUILD_PICO)
  add_subdirectory(pico)
endif()
