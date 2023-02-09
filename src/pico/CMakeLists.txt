set(PICO_CXX_ENABLE_EXCEPTIONS 1)

pico_sdk_init()

add_executable(main_pico main_pico.cpp)
target_link_libraries(main_pico nn pico_stdlib)
target_compile_options(main_pico PRIVATE -Wall -Wextra -Wpedantic -std=c++17 -O0 -ggdb)
pico_enable_stdio_usb(main_pico 1)
pico_enable_stdio_uart(main_pico 0)
pico_add_extra_outputs(main_pico)
add_pico_upload(main_pico)


add_executable(mnist_pico mnist_pico.cpp)
target_link_libraries(mnist_pico nn pico_stdlib pico_multicore)
target_compile_options(mnist_pico PRIVATE -Wall -Wextra -Wpedantic -std=c++17 -O2 -ggdb)
pico_enable_stdio_usb(mnist_pico 1)
pico_enable_stdio_uart(mnist_pico 0)
pico_add_extra_outputs(mnist_pico)
