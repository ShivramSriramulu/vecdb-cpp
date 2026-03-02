 cmake_minimum_required(VERSION 3.16)

 project(vecdb
   VERSION 0.1.0
   LANGUAGES CXX
 )

 set(CMAKE_CXX_STANDARD 17)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
 set(CMAKE_CXX_EXTENSIONS OFF)

 add_library(vecdb
   src/hnsw_index.cpp
   src/collection.cpp
   src/storage.cpp
   src/bruteforce.cpp
  src/wal.cpp
  src/transaction.cpp
 )

 target_include_directories(vecdb
   PUBLIC
     ${CMAKE_CURRENT_SOURCE_DIR}/include
   PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/src
 )

 add_executable(vecdb_example
   src/main.cpp
 )

 target_include_directories(vecdb_example
   PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/src
 )

 target_link_libraries(vecdb_example
   PRIVATE
     vecdb
 )

 add_executable(vecdb_tests
   tests/test_insert_search_persistence.cpp
 )
 target_include_directories(vecdb_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
 target_link_libraries(vecdb_tests PRIVATE vecdb)

 enable_testing()
 add_test(NAME VecDbTests COMMAND vecdb_tests)
