# TODO: Next Steps for Animation Interpolation App

## ✅ COMPLETED: Supabase Migration (2025-07-10)
- [x] Migrate backend from SQLite to Supabase Postgres
- [x] Update Job model to include project_id, scene_id, user_id
- [x] Create Supabase client module for storage and database operations
- [x] Refactor API endpoints to use Supabase Storage for file uploads
- [x] Update Celery worker to download from Supabase URLs and upload results
- [x] Fix timeline slot upsert logic with correct on_conflict format
- [x] Update job status and result endpoints to use Supabase URLs
- [x] Ensure frontend integration works with Supabase data flow

## ✅ COMPLETED: Cleanup & Frontend Updates (2025-07-10)
- [x] **Cleanup**: Remove temp static mount for serving images externally; keep local temp only for processing
- [x] **Frontend Updates**: 
  - [x] Ensure `submitInterpolationJob` includes user_id/project_id/scene_id
  - [x] Update polling & display logic to use Supabase URLs
  - [x] Refresh project browser & timeline after job completion by listening for new uploaded_files and timeline_slots rows
- [x] **Testing**: Delete local test.db file and related tests or adapt tests to Supabase mocks

## ✅ COMPLETED: Real-time Timeline Updates (2025-07-10)
- [x] **Fixed**: Allow dragging both keyframes and generated frames on timeline
- [x] **Fixed**: Real-time timeline updates not working after job completion
- [x] **Fixed**: Supabase real-time replication not enabled for timeline_slots table
- [x] **Fixed**: Frontend timeline slot updates not triggering real-time events
- [x] **Added**: Enhanced debugging for real-time subscription status and events
- [x] **Added**: Migration to enable real-time replication for timeline_slots, uploaded_files, and project_settings tables

## User Experience
- [ ] Add visual indicator for empty slots in the timeline and preview (e.g., checkerboard or faded overlay)
- [ ] Allow user to scrub or jump to next/previous keyframe or generated frame quickly
- [ ] Add keyboard shortcuts for timeline navigation and playback controls

## Robustness
- [ ] Handle edge cases where generated images fail to load (show error or retry)
- [ ] Add tests for timeline slot assignment and playback logic
- [ ] Ensure undo/redo works correctly with generated in-betweens
- [ ] Add error handling for Supabase connection issues
- [ ] Implement retry logic for failed uploads/downloads

## Features
- [ ] Allow user to select and delete generated in-betweens
- [ ] Support batch export of timeline as video or GIF
- [ ] Add support for multiple scenes/projects
- [ ] Integrate more interpolation engines and allow engine selection per job
- [ ] Add per-frame metadata (e.g., notes, tags)
- [ ] Add user authentication and project sharing
- [ ] Implement real-time collaboration features

## Performance & Optimization
- [ ] Optimize image loading and caching for timeline playback
- [ ] Implement progressive loading for large projects
- [ ] Add background job queue monitoring and management
- [ ] Optimize Supabase queries and storage usage

## Onion Skinning Improvements
### Quick Wins
- [ ] Show multiple ghost frames with configurable range (-3 … +3)
- [ ] Skip duplicate frames created by hold-last-frame logic
- [ ] Expose onion-skin settings (range, opacity, tint, direction) via UI pop-over
- [ ] Improve colouring (greyscale tint + `multiply` blend mode) for readability
- [ ] Pre-load and cache ghost images to avoid flicker

### Phase 2 – Performance & UX
- [ ] Introduce image caching & pre-loading optimisation
- [ ] Add keyboard shortcuts: `O` toggle, `[` / `]` adjust range
- [ ] Persist onion-skin preferences in localStorage / user profile

### Phase 3 – Canvas Renderer
- [ ] Replace stacked `<img>` layers with a Canvas 2D compositor for better performance
- [ ] Investigate WebGL shader for tint / opacity ramp

### Phase 4 – Shared Logic
- [ ] Extract a reusable `useOnionSkin` hook for preview player, future drawing canvas, and timeline thumbnails
- [ ] Ensure onion-skin feature works during scrubbing and playback across components

---

_Last updated: 2025-07-10 (Onion Skinning plan added)_