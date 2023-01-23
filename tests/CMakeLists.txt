include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(matrix_test matrix.test.cpp)
target_link_libraries(matrix_test PRIVATE Catch2::Catch2WithMain nn)

catch_discover_tests(matrix_test)
