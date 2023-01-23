add_executable(main 
    main.cpp)

# target_include_directories(main
#     PRIVATE
#         src
# )

target_link_libraries(main nn)