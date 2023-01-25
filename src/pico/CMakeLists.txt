pico_sdk_init()

add_executable(main_pico main_pico.cpp)
target_link_libraries(main_pico nn pico_stdlib)
pico_enable_stdio_usb(main_pico 1)
pico_enable_stdio_uart(main_pico 0)
pico_add_extra_outputs(main_pico)
add_pico_upload(main_pico)
