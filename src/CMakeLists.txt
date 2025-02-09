set(target ${MP2_LIBRARY})

file(GLOB hdrs "*.h*" "${MP2_INCLUDE}/*.h*")
file(GLOB srcs "*.cpp")

add_library(${target} ${srcs} ${hdrs})
#target_link_libraries(${target} ${MP2_LIBRARY})

#target_link_libraries(${target} stdc++)