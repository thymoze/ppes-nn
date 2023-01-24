add_executable(main 
    main.cpp)

add_executable(main_pretrained
    main_pretrained.cpp)

# target_include_directories(main
#     PRIVATE
#         src
# )

target_link_libraries(main nn)
target_link_libraries(main_pretrained nn)

if(BUILD_PICO)
  add_subdirectory(pico)
endif()
