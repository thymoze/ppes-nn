add_executable(main 
    main.cpp)

add_executable(main_pretrained
    main_pretrained.cpp)

add_executable(mnist 
    mnist.cpp)

add_executable(leak
    leak.cpp)

# target_include_directories(main
#     PRIVATE
#         src
# )

target_link_libraries(main nn)
target_link_libraries(main_pretrained nn)
target_link_libraries(mnist nn)
target_link_libraries(leak nn)

target_compile_options(leak 
  PRIVATE 
    -g -fsanitize=address -fsanitize=undefined
)
target_link_options(leak
  PRIVATE
    -fsanitize=address -fsanitize=undefined
)

if(BUILD_PICO)
  add_subdirectory(pico)
endif()
