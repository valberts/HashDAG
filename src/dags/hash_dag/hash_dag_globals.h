#pragma once

#include "typedefs.h"
#include <limits>
#include "utils.h"

// in number of uint32
constexpr uint32 C_pageSize = PAGE_SIZE;
constexpr uint32 C_maxNumberOfLevels = MAX_LEVELS;
// Consider adding a static_assert for C_maxNumberOfLevels >= 2 if C_leafLevel logic depends on it
// static_assert(C_maxNumberOfLevels >= 2, "MAX_LEVELS must be at least 2 for the current C_leafLevel definition");
constexpr uint32 C_leafLevel = (C_maxNumberOfLevels >= 2) ? (C_maxNumberOfLevels - 2) : 0; // Made C_leafLevel safer

constexpr uint32 C_colorTreeLevels_MaxDesired = 10;
constexpr uint32 C_colorTreeLevels = (C_maxNumberOfLevels < C_colorTreeLevels_MaxDesired) ? C_maxNumberOfLevels : C_colorTreeLevels_MaxDesired;

constexpr uint32 C_bucketSizeForTopLevels = BUCKETS_SIZE_FOR_TOP_LEVELS;
constexpr uint32 C_bucketSizeForLowLevels = BUCKETS_SIZE_FOR_LOW_LEVELS;
constexpr uint32 C_pagesPerBucketForTopLevels = C_bucketSizeForTopLevels / C_pageSize;
constexpr uint32 C_pagesPerBucketForLowLevels = C_bucketSizeForLowLevels / C_pageSize;

constexpr uint32 C_bucketsPerTopLevel = 1 << BUCKETS_BITS_FOR_TOP_LEVELS;
constexpr uint32 C_bucketsPerLowLevel = 1 << BUCKETS_BITS_FOR_LOW_LEVELS;
// BUCKETS_NUM_TOP_LEVELS (from typedefs.h, default 9) is the threshold that distinguishes
// top-level configuration from low-level configuration.

// Correctly define the *actual number of levels* that will use "top level" configuration.
// If C_maxNumberOfLevels is less than BUCKETS_NUM_TOP_LEVELS, all C_maxNumberOfLevels use top config.
// Otherwise, BUCKETS_NUM_TOP_LEVELS levels use top config.
constexpr uint32 C_bucketsNumTopLevels = (C_maxNumberOfLevels < BUCKETS_NUM_TOP_LEVELS) ? C_maxNumberOfLevels : BUCKETS_NUM_TOP_LEVELS;

// Correctly define the *actual number of levels* that will use "low level" configuration.
// This will be 0 if C_maxNumberOfLevels <= BUCKETS_NUM_TOP_LEVELS.
constexpr uint32 C_bucketsNumLowLevels = (C_maxNumberOfLevels > BUCKETS_NUM_TOP_LEVELS) ? (C_maxNumberOfLevels - BUCKETS_NUM_TOP_LEVELS) : 0;

constexpr uint32 C_totalNumberOfBucketsInTopLevels = C_bucketsNumTopLevels * C_bucketsPerTopLevel;
constexpr uint32 C_totalNumberOfBucketsInLowLevels = C_bucketsNumLowLevels * C_bucketsPerLowLevel;
constexpr uint32 C_totalNumberOfBuckets = C_totalNumberOfBucketsInTopLevels + C_totalNumberOfBucketsInLowLevels;

constexpr uint32 C_totalPages = C_totalNumberOfBucketsInTopLevels * C_pagesPerBucketForTopLevels + C_totalNumberOfBucketsInLowLevels * C_pagesPerBucketForLowLevels;
constexpr uint32 C_totalVirtualAddresses = C_totalPages * C_pageSize;

static_assert(C_bucketSizeForTopLevels % C_pageSize == 0, "Top bucket size must be a multiple of page size");
static_assert(C_bucketSizeForLowLevels % C_pageSize == 0, "Low bucket size must be a multiple of page size");
static_assert((C_pageSize & (C_pageSize - 1)) == 0, "Page size must be a power of 2");
static_assert((C_bucketSizeForTopLevels & (C_bucketSizeForTopLevels - 1)) == 0, "Top bucket size must be a power of 2");
// static_assert((C_bucketSizeForLowLevels & (C_bucketSizeForLowLevels - 1)) == 0, ""); //XXX-probably too restrictive?
static_assert((C_bucketsPerTopLevel & (C_bucketsPerTopLevel - 1)) == 0, "Buckets per top level must be a power of 2");
static_assert((C_bucketsPerLowLevel & (C_bucketsPerLowLevel - 1)) == 0, "Buckets per low level must be a power of 2");
static_assert(uint64(C_totalPages) * uint64(C_pageSize) < std::numeric_limits<uint32>::max(), "virtual address space too big");
static_assert(uint64(C_totalNumberOfBucketsInTopLevels) * uint64(C_pagesPerBucketForTopLevels) + uint64(C_totalNumberOfBucketsInLowLevels) + uint64(C_pagesPerBucketForLowLevels) < std::numeric_limits<uint32>::max(), "virtual address space really too big");
