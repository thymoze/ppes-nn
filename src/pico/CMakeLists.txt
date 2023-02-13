set(PICO_CXX_ENABLE_EXCEPTIONS 1)
set(PICO_CXX_ENABLE_RTTI 1)

pico_sdk_init()

function(pico_app target file)
  add_executable(${target} ${file}.cpp)
  target_link_libraries(${target} PRIVATE nn pico_stdlib)
  target_compile_options(${target}
    PRIVATE
      -Wall -Wextra -Wpedantic
  )

  pico_enable_stdio_usb(${target} 1)
  pico_enable_stdio_uart(${target} 0)
  pico_add_extra_outputs(${target})

  add_pico_upload(${target})
endfunction()

pico_app(main_pico main_pico)

pico_app(mnist_pico mnist_pico)
link_binary(mnist_pico t50-images-idx3-ubyte t50-labels-idx1-ubyte)
