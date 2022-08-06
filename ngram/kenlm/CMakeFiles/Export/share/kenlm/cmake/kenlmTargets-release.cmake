#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kenlm::kenlm_util" for configuration "Release"
set_property(TARGET kenlm::kenlm_util APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kenlm::kenlm_util PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkenlm_util.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS kenlm::kenlm_util )
list(APPEND _IMPORT_CHECK_FILES_FOR_kenlm::kenlm_util "${_IMPORT_PREFIX}/lib/libkenlm_util.a" )

# Import target "kenlm::kenlm_builder" for configuration "Release"
set_property(TARGET kenlm::kenlm_builder APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kenlm::kenlm_builder PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkenlm_builder.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS kenlm::kenlm_builder )
list(APPEND _IMPORT_CHECK_FILES_FOR_kenlm::kenlm_builder "${_IMPORT_PREFIX}/lib/libkenlm_builder.a" )

# Import target "kenlm::kenlm_filter" for configuration "Release"
set_property(TARGET kenlm::kenlm_filter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kenlm::kenlm_filter PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkenlm_filter.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS kenlm::kenlm_filter )
list(APPEND _IMPORT_CHECK_FILES_FOR_kenlm::kenlm_filter "${_IMPORT_PREFIX}/lib/libkenlm_filter.a" )

# Import target "kenlm::kenlm" for configuration "Release"
set_property(TARGET kenlm::kenlm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kenlm::kenlm PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkenlm.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS kenlm::kenlm )
list(APPEND _IMPORT_CHECK_FILES_FOR_kenlm::kenlm "${_IMPORT_PREFIX}/lib/libkenlm.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
