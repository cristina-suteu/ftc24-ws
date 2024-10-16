cmake_minimum_required(VERSION 3.9)

project(hello_iio)

add_executable(hello_iio hello_iio.c)

target_link_libraries(hello_iio iio)