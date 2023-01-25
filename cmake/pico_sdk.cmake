find_program(PICOTOOL picotool REQUIRED)

include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  pico_sdk
  GIT_SUBMODULES         lib/tinyusb
  GIT_SUBMODULES_RECURSE FALSE
  GIT_SHALLOW            TRUE
  GIT_PROGRESS           TRUE
  GIT_REPOSITORY         https://github.com/raspberrypi/pico-sdk
  GIT_TAG                1.4.0
)

FetchContent_Populate(pico_sdk)

include(${pico_sdk_SOURCE_DIR}/pico_sdk_init.cmake)

function(add_pico_upload target)
  add_custom_target(
    upload_${target}
    COMMAND
      sudo ${PICOTOOL} reboot -uf
    COMMAND
      sleep 1
    COMMAND
      cp ${target}.uf2 /tmp/
    COMMAND
      sudo ${PICOTOOL} load -x /tmp/${target}.uf2
    DEPENDS ${target}
    COMMENT "Uploading ${target}.uf2..."
  )
endfunction()
