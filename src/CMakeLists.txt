set(target ${MP2_LIBRARY})

file(GLOB hdrs "*.h*" "${MP2_INCLUDE}/*.h*")
file(GLOB srcs "*.cpp")

add_library(${target} ${srcs} ${hdrs})
#target_link_libraries(${target} ${MP2_LIBRARY})


set_property(TARGET ${MP2_LIBRARY} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${MP2_LIBRARY} PROPERTY CXX_STANDARD_REQUIRED ON)
#target_link_libraries(${target} stdc++)