project(hello-iio)

add_exectuable(hello hello_iio.c)
traget_link_libraries(hello iio)
