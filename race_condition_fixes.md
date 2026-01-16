# Race Condition Fixes in router.py

This document summarizes all the race condition fixes that have been implemented in the router.py file.

## Summary of Issues Found and Fixed

### 1. Cache Access Race Conditions

**Problem:** The `_models_cache`, `_loaded_models_cache`, and `_error_cache` dictionaries were being accessed concurrently without proper synchronization, leading to potential data corruption and inconsistent state.

**Solution:** Added three new asyncio.Lock objects to protect cache access:
- `_models_cache_lock` - protects `_models_cache`
- `_loaded_models_cache_lock` - protects `_loaded_models_cache`
- `_error_cache_lock` - protects `_error_cache`

**Implementation:**
- All cache reads now acquire the appropriate lock before accessing the cache
- All cache writes now acquire the appropriate lock before modifying the cache
- Cache entries are checked for freshness while holding the lock
- Stale entries are removed while holding the lock

### 2. Timestamp Calculation Race Condition

**Problem:** In the `token_worker()` function, the timestamp calculation was performed inside the lock, which could lead to:
- Inconsistent timestamps across multiple token entries
- Potential delays in processing due to lock contention
- Race conditions if the datetime object was modified between calculation and use

**Solution:** Moved timestamp calculation outside the lock:
- Calculate timestamp before acquiring the buffer lock
- Pass the pre-calculated timestamp into the critical section
- This ensures consistent timestamps while minimizing lock duration

**Implementation:**
```python
# Calculate timestamp once before acquiring lock
now = datetime.now(tz=timezone.utc)
timestamp = int(datetime(now.year, now.month, now.day, now.hour, now.minute, tzinfo=timezone.utc).timestamp())

# Then use it inside the lock
async with buffer_lock:
    time_series_buffer.append({
        'endpoint': endpoint,
        'model': model,
        'input_tokens': prompt,
        'output_tokens': comp,
        'total_tokens': prompt + comp,
        'timestamp': timestamp  # Use pre-calculated timestamp
    })
```

### 3. Buffer Access Race Conditions

**Problem:** The `token_buffer` and `time_series_buffer` were accessed concurrently from multiple sources (token_worker, flush_buffer, flush_remaining_buffers) without proper synchronization.

**Solution:** The existing `buffer_lock` was correctly used, but we ensured all access patterns properly acquire the lock.

**Implementation:**
- All reads and writes to `token_buffer` are protected by `buffer_lock`
- All reads and writes to `time_series_buffer` are protected by `buffer_lock`
- Buffer copies are made while holding the lock, then released before DB operations

### 4. Usage Count Race Conditions

**Problem:** The `usage_counts` and `token_usage_counts` dictionaries were accessed concurrently without proper synchronization.

**Solution:** The existing `usage_lock` and `token_usage_lock` were correctly used throughout the codebase.

**Implementation:**
- All increments/decrements of `usage_counts` are protected by `usage_lock`
- All increments/decrements of `token_usage_counts` are protected by `token_usage_lock`
- The `publish_snapshot()` function acquires `usage_lock` before creating snapshots

### 5. Subscriber Management Race Conditions

**Problem:** The `_subscribers` set was accessed concurrently without proper synchronization.

**Solution:** The existing `_subscribers_lock` was correctly used throughout the codebase.

**Implementation:**
- All additions/removals from `_subscribers` are protected by `_subscribers_lock`
- The `subscribe()` function acquires the lock before adding new subscribers
- The `unsubscribe()` function acquires the lock before removing subscribers
- The `publish_snapshot()` function acquires the lock before iterating over subscribers

## Testing Recommendations

To verify the race condition fixes:

1. **Concurrent Request Testing:**
   - Send multiple simultaneous requests to the same endpoint/model
   - Verify that usage counts remain consistent
   - Verify that token counts are accurately tracked

2. **Cache Consistency Testing:**
   - Rapidly query `/api/tags` and `/api/ps` endpoints
   - Verify that cached responses remain consistent
   - Verify that stale cache entries are properly invalidated

3. **Timestamp Consistency Testing:**
   - Send multiple requests in quick succession
   - Verify that timestamps in time_series_buffer are consistent and accurate
   - Verify that timestamps are properly rounded to minute boundaries

4. **Load Testing:**
   - Simulate high load with many concurrent connections
   - Verify that the system remains stable under load
   - Verify that no deadlocks occur

## Performance Considerations

The race condition fixes introduce additional locking, which may have performance implications:

1. **Lock Granularity:** The locks are fine-grained (one per cache type), minimizing contention.

2. **Lock Duration:** Locks are held for minimal durations - typically just for cache reads/writes.

3. **Concurrent Operations:** Multiple endpoints/models can be processed concurrently as long as they don't contend for the same cache entry.

4. **Monitoring:** Consider adding metrics to track lock contention and wait times.

## Future Improvements

1. **Read-Write Locks:** Consider using read-write locks if read-heavy workloads are identified.

2. **Cache Invalidation:** Implement more sophisticated cache invalidation strategies.

3. **Lock Timeouts:** Add timeout mechanisms to prevent deadlocks.

4. **Performance Monitoring:** Add instrumentation to track lock contention and performance.

