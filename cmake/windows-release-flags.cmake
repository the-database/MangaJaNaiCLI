# Windows only: put CMake's Release optimization flags back for libjxl (the `jxl` feature).
#
# With the Visual Studio generator, the cmake crate passes CMAKE_<LANG>_FLAGS_<CONFIG> as the
# compiler's base flags with every /O flag removed (cmake 0.1.58), so libjxl built unoptimized and
# without NDEBUG, far slower than the same code on Linux, where other generators keep CMake's
# defaults. .cargo/config.toml points the cmake crate at this toolchain file for
# x86_64-pc-windows-msvc only; jpegxl-src is the only CMake build here. A forced cache entry,
# because libjxl's old cmake_minimum_required (CMP0126 OLD) would drop a normal variable.
foreach(lang C CXX)
  if(NOT CMAKE_${lang}_FLAGS_RELEASE MATCHES "/O2")
    set(CMAKE_${lang}_FLAGS_RELEASE "${CMAKE_${lang}_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG" CACHE STRING "" FORCE)
  endif()
endforeach()
