# Install script for directory: /home/mikawa/kenlm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake/kenlmTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake/kenlmTargets.cmake"
         "/home/mikawa/kenlm/build/CMakeFiles/Export/share/kenlm/cmake/kenlmTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake/kenlmTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake/kenlmTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake" TYPE FILE FILES "/home/mikawa/kenlm/build/CMakeFiles/Export/share/kenlm/cmake/kenlmTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake" TYPE FILE FILES "/home/mikawa/kenlm/build/CMakeFiles/Export/share/kenlm/cmake/kenlmTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/util" TYPE FILE FILES
    "/home/mikawa/kenlm/util/bit_packing.hh"
    "/home/mikawa/kenlm/util/ersatz_progress.hh"
    "/home/mikawa/kenlm/util/exception.hh"
    "/home/mikawa/kenlm/util/fake_ostream.hh"
    "/home/mikawa/kenlm/util/file.hh"
    "/home/mikawa/kenlm/util/file_piece.hh"
    "/home/mikawa/kenlm/util/file_stream.hh"
    "/home/mikawa/kenlm/util/fixed_array.hh"
    "/home/mikawa/kenlm/util/float_to_string.hh"
    "/home/mikawa/kenlm/util/getopt.hh"
    "/home/mikawa/kenlm/util/have.hh"
    "/home/mikawa/kenlm/util/integer_to_string.hh"
    "/home/mikawa/kenlm/util/joint_sort.hh"
    "/home/mikawa/kenlm/util/mmap.hh"
    "/home/mikawa/kenlm/util/multi_intersection.hh"
    "/home/mikawa/kenlm/util/murmur_hash.hh"
    "/home/mikawa/kenlm/util/parallel_read.hh"
    "/home/mikawa/kenlm/util/pcqueue.hh"
    "/home/mikawa/kenlm/util/pool.hh"
    "/home/mikawa/kenlm/util/probing_hash_table.hh"
    "/home/mikawa/kenlm/util/proxy_iterator.hh"
    "/home/mikawa/kenlm/util/read_compressed.hh"
    "/home/mikawa/kenlm/util/scoped.hh"
    "/home/mikawa/kenlm/util/sized_iterator.hh"
    "/home/mikawa/kenlm/util/sorted_uniform.hh"
    "/home/mikawa/kenlm/util/spaces.hh"
    "/home/mikawa/kenlm/util/string_piece.hh"
    "/home/mikawa/kenlm/util/string_piece_hash.hh"
    "/home/mikawa/kenlm/util/string_stream.hh"
    "/home/mikawa/kenlm/util/thread_pool.hh"
    "/home/mikawa/kenlm/util/tokenize_piece.hh"
    "/home/mikawa/kenlm/util/usage.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/util/double-conversion" TYPE FILE FILES
    "/home/mikawa/kenlm/util/double-conversion/bignum-dtoa.h"
    "/home/mikawa/kenlm/util/double-conversion/bignum.h"
    "/home/mikawa/kenlm/util/double-conversion/cached-powers.h"
    "/home/mikawa/kenlm/util/double-conversion/diy-fp.h"
    "/home/mikawa/kenlm/util/double-conversion/double-conversion.h"
    "/home/mikawa/kenlm/util/double-conversion/fast-dtoa.h"
    "/home/mikawa/kenlm/util/double-conversion/fixed-dtoa.h"
    "/home/mikawa/kenlm/util/double-conversion/ieee.h"
    "/home/mikawa/kenlm/util/double-conversion/strtod.h"
    "/home/mikawa/kenlm/util/double-conversion/utils.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/util/stream" TYPE FILE FILES
    "/home/mikawa/kenlm/util/stream/block.hh"
    "/home/mikawa/kenlm/util/stream/chain.hh"
    "/home/mikawa/kenlm/util/stream/config.hh"
    "/home/mikawa/kenlm/util/stream/count_records.hh"
    "/home/mikawa/kenlm/util/stream/io.hh"
    "/home/mikawa/kenlm/util/stream/line_input.hh"
    "/home/mikawa/kenlm/util/stream/multi_progress.hh"
    "/home/mikawa/kenlm/util/stream/multi_stream.hh"
    "/home/mikawa/kenlm/util/stream/rewindable_stream.hh"
    "/home/mikawa/kenlm/util/stream/sort.hh"
    "/home/mikawa/kenlm/util/stream/stream.hh"
    "/home/mikawa/kenlm/util/stream/typed_stream.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/lm" TYPE FILE FILES
    "/home/mikawa/kenlm/lm/bhiksha.hh"
    "/home/mikawa/kenlm/lm/binary_format.hh"
    "/home/mikawa/kenlm/lm/blank.hh"
    "/home/mikawa/kenlm/lm/config.hh"
    "/home/mikawa/kenlm/lm/enumerate_vocab.hh"
    "/home/mikawa/kenlm/lm/facade.hh"
    "/home/mikawa/kenlm/lm/left.hh"
    "/home/mikawa/kenlm/lm/lm_exception.hh"
    "/home/mikawa/kenlm/lm/max_order.hh"
    "/home/mikawa/kenlm/lm/model.hh"
    "/home/mikawa/kenlm/lm/model_type.hh"
    "/home/mikawa/kenlm/lm/ngram_query.hh"
    "/home/mikawa/kenlm/lm/partial.hh"
    "/home/mikawa/kenlm/lm/quantize.hh"
    "/home/mikawa/kenlm/lm/read_arpa.hh"
    "/home/mikawa/kenlm/lm/return.hh"
    "/home/mikawa/kenlm/lm/search_hashed.hh"
    "/home/mikawa/kenlm/lm/search_trie.hh"
    "/home/mikawa/kenlm/lm/sizes.hh"
    "/home/mikawa/kenlm/lm/state.hh"
    "/home/mikawa/kenlm/lm/trie.hh"
    "/home/mikawa/kenlm/lm/trie_sort.hh"
    "/home/mikawa/kenlm/lm/value.hh"
    "/home/mikawa/kenlm/lm/value_build.hh"
    "/home/mikawa/kenlm/lm/virtual_interface.hh"
    "/home/mikawa/kenlm/lm/vocab.hh"
    "/home/mikawa/kenlm/lm/weights.hh"
    "/home/mikawa/kenlm/lm/word_index.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/lm/builder" TYPE FILE FILES
    "/home/mikawa/kenlm/lm/builder/adjust_counts.hh"
    "/home/mikawa/kenlm/lm/builder/combine_counts.hh"
    "/home/mikawa/kenlm/lm/builder/corpus_count.hh"
    "/home/mikawa/kenlm/lm/builder/debug_print.hh"
    "/home/mikawa/kenlm/lm/builder/discount.hh"
    "/home/mikawa/kenlm/lm/builder/hash_gamma.hh"
    "/home/mikawa/kenlm/lm/builder/header_info.hh"
    "/home/mikawa/kenlm/lm/builder/initial_probabilities.hh"
    "/home/mikawa/kenlm/lm/builder/interpolate.hh"
    "/home/mikawa/kenlm/lm/builder/output.hh"
    "/home/mikawa/kenlm/lm/builder/payload.hh"
    "/home/mikawa/kenlm/lm/builder/pipeline.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/lm/common" TYPE FILE FILES
    "/home/mikawa/kenlm/lm/common/compare.hh"
    "/home/mikawa/kenlm/lm/common/joint_order.hh"
    "/home/mikawa/kenlm/lm/common/model_buffer.hh"
    "/home/mikawa/kenlm/lm/common/ngram.hh"
    "/home/mikawa/kenlm/lm/common/ngram_stream.hh"
    "/home/mikawa/kenlm/lm/common/print.hh"
    "/home/mikawa/kenlm/lm/common/renumber.hh"
    "/home/mikawa/kenlm/lm/common/size_option.hh"
    "/home/mikawa/kenlm/lm/common/special.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/lm/filter" TYPE FILE FILES
    "/home/mikawa/kenlm/lm/filter/arpa_io.hh"
    "/home/mikawa/kenlm/lm/filter/count_io.hh"
    "/home/mikawa/kenlm/lm/filter/format.hh"
    "/home/mikawa/kenlm/lm/filter/phrase.hh"
    "/home/mikawa/kenlm/lm/filter/thread.hh"
    "/home/mikawa/kenlm/lm/filter/vocab.hh"
    "/home/mikawa/kenlm/lm/filter/wrapper.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kenlm/lm/interpolate" TYPE FILE FILES
    "/home/mikawa/kenlm/lm/interpolate/backoff_matrix.hh"
    "/home/mikawa/kenlm/lm/interpolate/backoff_reunification.hh"
    "/home/mikawa/kenlm/lm/interpolate/bounded_sequence_encoding.hh"
    "/home/mikawa/kenlm/lm/interpolate/interpolate_info.hh"
    "/home/mikawa/kenlm/lm/interpolate/merge_probabilities.hh"
    "/home/mikawa/kenlm/lm/interpolate/merge_vocab.hh"
    "/home/mikawa/kenlm/lm/interpolate/normalize.hh"
    "/home/mikawa/kenlm/lm/interpolate/pipeline.hh"
    "/home/mikawa/kenlm/lm/interpolate/split_worker.hh"
    "/home/mikawa/kenlm/lm/interpolate/tune_derivatives.hh"
    "/home/mikawa/kenlm/lm/interpolate/tune_instances.hh"
    "/home/mikawa/kenlm/lm/interpolate/tune_matrix.hh"
    "/home/mikawa/kenlm/lm/interpolate/tune_weights.hh"
    "/home/mikawa/kenlm/lm/interpolate/universal_vocab.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kenlm/cmake" TYPE FILE FILES "/home/mikawa/kenlm/build/kenlmConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/mikawa/kenlm/build/util/cmake_install.cmake")
  include("/home/mikawa/kenlm/build/lm/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/mikawa/kenlm/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
