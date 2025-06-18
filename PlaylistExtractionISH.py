import os
import cv2
import yt_dlp
import tempfile
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
import concurrent.futures
import datetime
import subprocess
import shutil
import csv
import time
import random

# Global variable specifying the base output directory
OUTPUT_PATH = r"Awareness"
# Global variable for the CSV output file
CSV_OUTPUT_FILE = os.path.join(OUTPUT_PATH, "clips_transcript_mapping.csv")
# Max number of retries for failed downloads
MAX_RETRIES = 3
# Delay between API calls to avoid rate limiting (in seconds)
MIN_DELAY = 1
MAX_DELAY = 5

def download_video(youtube_url, temp_dir, retry_count=0):
    """
    Download a YouTube video as an MP4 file using yt_dlp.
    Uses cookies from the browser to access private videos if available.
    If the video is private (or inaccessible), skip further processing.
    Returns the local filename and the video ID.
    """
    # Extract video ID from URL for better logging
    video_id = youtube_url.split("v=")[-1].split("&")[0] if "v=" in youtube_url else "unknown"
    print(f"\n{'='*80}")
    print(f"⏳ STARTING DOWNLOAD: Video ID {video_id}")
    print(f"🔗 URL: {youtube_url}")
    print(f"⏱️ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define a progress hook to show download progress
    def my_hook(d):
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes'] > 0:
                percent = d['downloaded_bytes'] / d['total_bytes'] * 100
                print(f"\r📥 Downloading: {percent:.1f}% of {d['total_bytes']/1024/1024:.1f} MB", end='')
            elif 'downloaded_bytes' in d:
                print(f"\r📥 Downloaded: {d['downloaded_bytes']/1024/1024:.1f} MB so far", end='')
        elif d['status'] == 'finished':
            print(f"\r✅ Download complete! Total size: {d['total_bytes']/1024/1024:.1f} MB")

    # Create a unique filename in the temp directory
    video_output_path = os.path.join(temp_dir, f"temp_video_{video_id}.mp4")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': video_output_path,
        'quiet': True,
        'no_warnings': True,
        'cookies_from_browser': 'chrome',
        'progress_hooks': [my_hook],
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'socket_timeout': 30,  # Increased timeout
        'retries': 10,         # More retries for transient issues
        'fragment_retries': 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)

        if info is None:
            print(f"❌ Failed to extract info for video {video_id}")
            if retry_count < MAX_RETRIES:
                retry_count += 1
                wait_time = random.uniform(MIN_DELAY, MAX_DELAY) * retry_count
                print(f"🔄 Retrying download ({retry_count}/{MAX_RETRIES}) after {wait_time:.1f}s...")
                time.sleep(wait_time)
                return download_video(youtube_url, temp_dir, retry_count)
            else:
                print(f"❌ Max retries reached for video {video_id}. Skipping.")
                return None, video_id

        if info.get("is_private", False):
            print(f"❌ The video {video_id} is private. Skipping processing.")
            return None, video_id

        # Get details about the video for better logging
        title = info.get('title', 'Unknown Title')
        ext = info.get('ext', 'mp4')
        video_id = info.get('id')

        print(f"📋 Video details:")
        print(f"   📌 Title: {title}")
        print(f"   📌 ID: {video_id}")
        print(f"   📌 Format: {ext}")
        print(f"   📌 Saved as: {video_output_path}")

        # Verify file exists and has content
        if not os.path.exists(video_output_path) or os.path.getsize(video_output_path) < 10000:
            print(f"⚠️ Downloaded file is missing or too small: {video_output_path}")
            if retry_count < MAX_RETRIES:
                retry_count += 1
                wait_time = random.uniform(MIN_DELAY, MAX_DELAY) * retry_count
                print(f"🔄 Retrying download ({retry_count}/{MAX_RETRIES}) after {wait_time:.1f}s...")
                time.sleep(wait_time)
                return download_video(youtube_url, temp_dir, retry_count)
            else:
                print(f"❌ Max retries reached for video {video_id}. Skipping.")
                return None, video_id

        return video_output_path, video_id

    except Exception as e:
        if "private" in str(e).lower():
            print(f"❌ ERROR: The video {video_id} is private. Skipping processing.")
            return None, video_id
        else:
            print(f"❌ ERROR downloading video {video_id}: {str(e)}")
            if retry_count < MAX_RETRIES:
                retry_count += 1
                wait_time = random.uniform(MIN_DELAY, MAX_DELAY) * retry_count
                print(f"🔄 Retrying download ({retry_count}/{MAX_RETRIES}) after {wait_time:.1f}s...")
                time.sleep(wait_time)
                return download_video(youtube_url, temp_dir, retry_count)
            else:
                print(f"❌ Max retries reached for video {video_id}. Skipping.")
                return None, video_id
def get_middle_frame(video_path):
    """
    Extract the middle frame from a video file.
    Returns the frame as a numpy array.
    """
    try:
        print(f"\n🎞️ Extracting middle frame from video...")
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"⚠️ Could not determine frame count, using 30-second mark instead")
            # If frame count is unavailable, jump to 30-second mark
            video.set(cv2.CAP_PROP_POS_MSEC, 30000)  # 30 seconds in milliseconds
        else:
            # Set position to middle frame
            middle_frame_idx = total_frames // 2
            video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            print(f"   📊 Using frame {middle_frame_idx} of {total_frames}")
        
        # Read the frame
        ret, frame = video.read()
        video.release()
        
        if not ret or frame is None:
            print(f"⚠️ Failed to extract middle frame, returning None")
            return None
            
        print(f"✅ Middle frame extracted successfully")
        return frame
    except Exception as e:
        print(f"❌ Error extracting middle frame: {e}")
        return None

def select_roi(frame):
    """
    Display a frame to the user and let them select a rectangle (ROI).
    Returns a tuple (x, y, w, h) to use for cropping.
    """
    print("\n🖱️ ROI SELECTION: Please define the region of interest")
    
    if frame is None:
        # If no frame provided, use default ROI
        roi = [100, 21, 1000, 1030]
        print(f"⚠️ No frame available for selection. Using predefined ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        return roi
    
    print("   - Draw a rectangle by clicking and dragging")
    print("   - Press ENTER or SPACE to confirm")
    print("   - Press 'c' to cancel selection")
    
    # Resize frame if it's too large for display
    h, w = frame.shape[:2]
    display_scale = 1.0
    if w > 1280:
        display_scale = 1280 / w
        frame = cv2.resize(frame, (int(w * display_scale), int(h * display_scale)))
        print(f"   📏 Resized frame for display (scale: {display_scale:.2f})")
    
    # Display the frame and let the user select ROI
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    roi = cv2.selectROI('Select ROI', frame, False)
    cv2.destroyAllWindows()
    
    # Scale ROI back to original dimensions if frame was resized
    if display_scale != 1.0:
        roi = tuple(int(x / display_scale) for x in roi)
        print(f"   📏 Adjusted ROI back to original scale")
    
    print(f"✅ ROI selected: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
    return roi

def extract_transcript(video_id, retry_count=0):
    """
    Extract the English (India) transcript (subtitles) for the given video ID.
    Falls back to standard English if English (India) isn't available.
    Returns a list of entries (each with 'text', 'start', and 'duration').
    """
    print(f"\n📝 Extracting English (India) transcript for video {video_id}...")
    try:
        # Try to get the transcript list
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # First try to get English (India) transcript
        try:
            # The language code for English (India) is 'en-IN'
            transcript = transcript_list.find_transcript(['en-IN']).fetch()
            print(f"✅ English (India) transcript successfully extracted: {len(transcript)} segments found")
            return transcript
        except:
            # If English (India) is not available, try standard English
            try:
                transcript = transcript_list.find_transcript(['en']).fetch()
                print(f"⚠️ English (India) transcript not available. Using standard English: {len(transcript)} segments found")
                return transcript
            except:
                # If no English transcript at all, skip this video
                print(f"⚠️ No English transcript available for video {video_id}. Skipping.")
                return []

    except Exception as e:
        print(f"❌ Error extracting transcript: {e}")

        # Handle retries
        if retry_count < MAX_RETRIES:
            retry_count += 1
            wait_time = random.uniform(MIN_DELAY, MAX_DELAY) * retry_count
            print(f"🔄 Retrying transcript extraction ({retry_count}/{MAX_RETRIES}) after {wait_time:.1f}s...")
            time.sleep(wait_time)
            return extract_transcript(video_id, retry_count)

        # Return empty list if all attempts fail
        print(f"⚠️ Could not extract transcript after {MAX_RETRIES} attempts. Skipping video {video_id}.")
        return []

def extract_audio_segment(video_path, start_time, duration, output_file):
    """
    Extract audio segment directly using ffmpeg for more reliable extraction.

    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        duration: Duration in seconds
        output_file: Output audio file path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"   🔊 Extracting audio using ffmpeg from {start_time:.2f}s for {duration:.2f}s...")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-vn', '-acodec', 'libmp3lame',
            '-q:a', '2', output_file
        ]
        result = subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"   ✅ Audio saved successfully to: {output_file}")
            return True
        else:
            print(f"   ⚠️ Audio file exists but may be empty")
            return False
    except Exception as e:
        print(f"   ❌ Error extracting audio: {e}")
        return False

def write_to_csv(video_id, clip_idx, subtitle_text):
    """
    Write a single row to the CSV file, creating it if it doesn't exist.

    Args:
        video_id: YouTube video ID
        clip_idx: Clip index number
        subtitle_text: Text content from the subtitle
    """
    clip_identifier = f"{video_id}_clip_{clip_idx+1}"

    # Create or append to the CSV file
    file_exists = os.path.isfile(CSV_OUTPUT_FILE)

    try:
        with open(CSV_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if file is being created
            if not file_exists:
                writer.writerow(['clip_id', 'transcript_text'])
                print(f"📊 Created new CSV file: {CSV_OUTPUT_FILE}")

            # Write the data row
            writer.writerow([clip_identifier, subtitle_text])

        return True
    except Exception as e:
        print(f"   ❌ Error writing to CSV: {e}")
        return False

def process_clips(video_path, transcript, roi, output_folder, video_id):
    """
    For each transcript entry, clip the corresponding video segment, crop it using the ROI,
    mute the clip, extract its audio, and store the results in separate folders under output_folder.
    Also writes transcript information to a central CSV file.
    """
    print(f"\n🔪 PROCESSING CLIPS: Creating segments based on transcript")
    print(f"📂 Output folder: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    x, y, w, h = roi
    x2, y2 = x + w, y + h
    print(f"🔍 Using ROI: x={x}, y={y}, width={w}, height={h}")

    print(f"🎬 Opening main video file: {video_path}")

    try:
        video = VideoFileClip(video_path)
        total_clips = len(transcript)
    except Exception as e:
        print(f"❌ Error opening video file: {e}")
        return False

    for idx, entry in enumerate(transcript):
        try:
            start = entry.start if hasattr(entry, 'start') else entry['start']
            duration = entry.duration if hasattr(entry, 'duration') else entry['duration']
            text = entry.text if hasattr(entry, 'text') else entry['text']
            end = start + duration

            progress = (idx + 1) / total_clips * 100
            print(f"\n🔄 Processing clip {idx+1}/{total_clips} ({progress:.1f}%)")
            print(f"   ⏱️ Time segment: {start:.2f}s to {end:.2f}s (duration: {duration:.2f}s)")
            print(f"   💬 Text: \"{text}\"")

            # Create a folder for this clip.
            clip_folder = os.path.join(output_folder, f"clip_{idx+1}")
            os.makedirs(clip_folder, exist_ok=True)
            print(f"   📂 Saving to: {clip_folder}")

            # Create a subclip for this transcript segment.
            print(f"   ✂️ Creating subclip...")

            try:
                # Handle cases where subclip extends beyond video duration
                if end > video.duration:
                    print(f"   ⚠️ Clip end time ({end:.2f}s) exceeds video duration ({video.duration:.2f}s)")
                    end = video.duration
                    duration = end - start
                    print(f"   🔄 Adjusted clip duration to {duration:.2f}s")

                if start >= video.duration:
                    print(f"   ⚠️ Clip start time ({start:.2f}s) exceeds video duration ({video.duration:.2f}s)")
                    print(f"   ⏩ Skipping this clip")
                    continue

                subclip = video.subclip(start, end)

                # Crop the clip according to the ROI.
                print(f"   🔍 Cropping to ROI...")
                cropped_clip = subclip.crop(x1=x, y1=y, x2=x2, y2=y2)

                # Mute the cropped clip.
                print(f"   🔇 Removing audio from cropped clip...")
                muted_clip = cropped_clip.without_audio()

                # Save the muted, cropped video.
                video_filename = os.path.join(clip_folder, "video.mp4")
                print(f"   💾 Writing video clip...")
                muted_clip.write_videofile(video_filename, codec="libx264", audio=False,
                                          verbose=False, logger=None, threads=2)
                print(f"   ✅ Video saved to: {video_filename}")

                # Extract audio directly using ffmpeg instead of MoviePy
                audio_filename = os.path.join(clip_folder, "audio.mp3")
                extract_audio_segment(video_path, start, duration, audio_filename)

                # Save the subtitle text for this clip.
                subtitle_filename = os.path.join(clip_folder, "subtitle.txt")
                print(f"   📝 Saving subtitle...")
                with open(subtitle_filename, "w", encoding="utf-8") as f:
                    f.write(entry['text'])
                print(f"   ✅ Subtitle saved to: {subtitle_filename}")

                # Write clip and transcript to the central CSV file
                print(f"   📊 Adding to CSV mapping...")
                write_to_csv(video_id, idx, entry['text'])
                print(f"   ✅ Added to CSV mapping")

                # Close clips to free memory
                muted_clip.close()
                cropped_clip.close()
                subclip.close()

                print(f"   ✅ Clip {idx+1}/{total_clips} complete")

            except Exception as e:
                print(f"   ❌ Error processing clip {idx+1}: {e}")
                print(f"   ⏩ Skipping to next clip")
                # Save the subtitle text even if video processing fails
                subtitle_filename = os.path.join(clip_folder, "subtitle.txt")
                with open(subtitle_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                write_to_csv(video_id, idx, text)
                continue

        except Exception as e:
            print(f"   ❌ Error processing clip {idx+1}: {e}")
            print(f"   ⏩ Skipping to next clip")
            continue

    video.close()
    print(f"\n✅ All {total_clips} processed clips completed successfully.")
    return True

def process_video(video_url, roi=None, temp_dir=None):
    """
    Process a single video:
      1. Download the video to a temporary directory.
      2. Capture its middle frame and, if roi is None, prompt for ROI selection.
      3. Extract the transcript.
      4. Process the video segments.
      5. Clean up by deleting the downloaded video.
    All output for this video is stored in OUTPUT_PATH/<video_id>/.
    If roi is provided, it is used; otherwise, interactive selection is performed.
    Returns the ROI used (if selected interactively).
    """
    # Create a temp directory if none provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="youtube_processor_")
        local_temp_dir = True
    else:
        local_temp_dir = False

    # Extract video ID from URL for better logging
    video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else "unknown"
    print(f"\n{'#'*100}")
    print(f"🎬 PROCESSING VIDEO: {video_id}")
    print(f"{'#'*100}")

    # Initialize video_path to None to avoid reference errors
    video_path = None

    try:
        # Add delay between video processing to avoid rate limiting
        wait_time = random.uniform(MIN_DELAY, MAX_DELAY)
        print(f"⏱️ Adding delay of {wait_time:.1f}s before processing to avoid rate limiting...")
        time.sleep(wait_time)

        video_path, video_id = download_video(video_url, temp_dir)
        if not video_path or not video_id:
            print(f"⏩ Skipping video {video_id} due to privacy or download issues.")
            return roi  # return passed roi even if video skipped

        # If no ROI provided, select from the middle frame
        if roi is None:
            print("🖌️ No ROI provided - extracting middle frame for selection")
            middle_frame = get_middle_frame(video_path)
            roi = select_roi(middle_frame)
        else:
            print(f"🔄 Using pre-selected ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")

        transcript = extract_transcript(video_id)
        if not transcript:
            print(f"⚠️ No transcript available for video {video_id}. Skipping.")
            return roi

        # Create a subfolder for this video using its video_id.
        video_output_folder = os.path.join(OUTPUT_PATH, video_id)
        success = process_clips(video_path, transcript, roi, video_output_folder, video_id)

        if success:
            print(f"\n✅ COMPLETED PROCESSING FOR VIDEO {video_id}")
            print(f"📂 Results saved to: {video_output_folder}")
        else:
            print(f"\n⚠️ PARTIAL PROCESSING COMPLETED FOR VIDEO {video_id}")

        print(f"{'='*100}")
        return roi

    except Exception as e:
        print(f"❌ Error in process_video for {video_id}: {e}")
        return roi

    finally:
        # Clean up the downloaded video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"🗑️ Deleted temporary video file: {video_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temporary file: {e}")

        # Clean up temp directory if we created it locally
        if local_temp_dir and temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️ Deleted temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Could not delete temporary directory: {e}")

def get_playlist_video_urls(playlist_url, batch_size=50):
    """
    Extract video URLs from the playlist using yt_dlp.
    Returns a list of video URLs.
    Handles large playlists by processing in batches.
    """
    print(f"\n{'='*80}")
    print(f"📋 ANALYZING PLAYLIST")
    print(f"🔗 URL: {playlist_url}")
    print(f"⏱️ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Track seen video IDs to prevent duplicates
    seen_video_ids = set()
    video_urls = []

    # First try as a playlist
    try:
        # Handle large playlists by processing in batches
        start_idx = 1
        more_videos = True
        batch_count = 1

        while more_videos:
            print(f"🔍 Extracting batch {batch_count} (items {start_idx}-{start_idx+batch_size-1})...")

            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'skip_download': True,
                'playliststart': start_idx,
                'playlistend': start_idx + batch_size - 1,
                'cookies_from_browser': 'chrome',
                'ignoreerrors': True,
                'nocheckcertificate': True,
                'socket_timeout': 30
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)

            if playlist_info is None:
                print("❌ Failed to extract playlist info.")
                break

            if batch_count == 1:
                playlist_title = playlist_info.get('title', 'Unknown Playlist')
                print(f"✅ Found playlist: \"{playlist_title}\"")

            entries = playlist_info.get('entries', [])

            if not entries:
                print(f"📊 No more videos found after {len(video_urls)} videos.")
                more_videos = False
                break

            new_count = 0
            for entry in entries:
                if entry:
                    video_id = entry.get('id')
                    if video_id and video_id not in seen_video_ids:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        video_urls.append(video_url)
                        seen_video_ids.add(video_id)
                        new_count += 1

            print(f"✅ Batch {batch_count}: Found {new_count} new videos")

            if new_count < batch_size:
                more_videos = False
            else:
                start_idx += batch_size
                batch_count += 1
                # Add a delay to avoid rate limiting
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        print(f"\n✅ Successfully extracted {len(video_urls)} unique video URLs from playlist")

    except Exception as e:
        print(f"⚠️ Error processing playlist: {e}")

        # If failed as a playlist, check if it's a single video URL
        if "youtube.com/watch" in playlist_url or "youtu.be/" in playlist_url:
            print("🔍 URL appears to be a single video. Treating as video instead of playlist.")
            video_id = None

            if "youtube.com/watch" in playlist_url and "v=" in playlist_url:
                video_id = playlist_url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in playlist_url:
                video_id = playlist_url.split("youtu.be/")[-1].split("?")[0]

            if video_id:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                return [video_url]

    return video_urls

def process_playlist_parallel(playlist_url, max_workers=4, batch_size=50):
    """
    Process each video in the playlist.
    The first video is processed sequentially (to select ROI), then the remaining videos are processed in parallel.
    Private videos or videos causing errors are skipped.
    """
    print(f"\n{'#'*100}")
    print(f"🎬 STARTING PLAYLIST PROCESSING")
    print(f"{'#'*100}")

    print(f"🧰 Configuration:")
    print(f"   📂 Output directory: {OUTPUT_PATH}")
    print(f"   📊 Max parallel workers: {max_workers}")
    print(f"   📦 Batch size for playlist fetching: {batch_size}")
    print(f"   🔄 Max retries: {MAX_RETRIES}")

    # Create or clear the CSV file
    if os.path.exists(CSV_OUTPUT_FILE):
        os.remove(CSV_OUTPUT_FILE)
        print(f"🗑️ Removed existing CSV file for fresh start: {CSV_OUTPUT_FILE}")

    # Create a main temporary directory for all downloads
    main_temp_dir = tempfile.mkdtemp(prefix="youtube_playlist_")
    print(f"📂 Created temporary directory for downloads: {main_temp_dir}")

    try:
        video_urls = get_playlist_video_urls(playlist_url, batch_size)
        if not video_urls:
            print("❌ No videos found in the playlist. Exiting.")
            return

        # Process first video to get ROI from its middle frame
        if video_urls:
            print("\n🔍 PROCESSING FIRST VIDEO TO SELECT ROI")
            first_video_temp = os.path.join(main_temp_dir, "first_video")
            os.makedirs(first_video_temp, exist_ok=True)
            
            # Download first video and get middle frame for ROI selection
            first_url = video_urls[0]
            roi = process_video(first_url, None, first_video_temp)
            
            # Remove first video from the list if it was processed successfully
            if roi and (roi[2] > 0 and roi[3] > 0):
                print(f"✅ Successfully processed first video and selected ROI")
                video_urls = video_urls[1:]  # Remove first video from list
            else:
                # If ROI selection failed, use default values
                print("❌ Failed to get a valid ROI. Using default values.")
                roi = [100, 21, 1000, 1030]  # Fallback to default values
        else:
            # Fallback if no videos were found
            roi = [100, 21, 1000, 1030]

        # Process videos in batches
        remaining_count = len(video_urls)
        print(f"\n🚀 PROCESSING {remaining_count} VIDEOS")
        print(f"🖌️ Using ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")

        # Process in smaller batches to avoid memory issues
        batch_size = min(100, remaining_count)
        total_batches = (remaining_count + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, remaining_count)
            current_batch = video_urls[start_idx:end_idx]

            print(f"\n{'='*80}")
            print(f"🔄 Processing batch {batch_num+1}/{total_batches}")
            print(f"📊 Videos in batch: {len(current_batch)} (overall: {start_idx+1}-{end_idx}/{remaining_count})")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs with the same ROI.
                print(f"⏳ Submitting {len(current_batch)} videos to process pool...")

                # Create individual temp directories for each video
                futures = {}
                for i, video_url in enumerate(current_batch):
                    video_temp = os.path.join(main_temp_dir, f"batch_{batch_num}_video_{i}")
                    os.makedirs(video_temp, exist_ok=True)
                    futures[executor.submit(process_video, video_url, roi, video_temp)] = video_url

                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    video_url = futures[future]
                    video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else "unknown"
                    completed += 1

                    try:
                        _ = future.result()
                        print(f"\n✅ Completed video {start_idx+completed}/{remaining_count}: {video_id}")
                    except Exception as exc:
                        print(f"\n❌ Error processing video {start_idx+completed}/{remaining_count}: {video_id}")
                        print(f"   Error details: {exc}")

                    progress = ((start_idx + completed) / remaining_count) * 100
                    print(f"📊 Overall progress: {progress:.1f}% ({start_idx+completed}/{remaining_count} videos processed)")

            # Clean up batch temporary directories
            for i in range(len(current_batch)):
                batch_dir = os.path.join(main_temp_dir, f"batch_{batch_num}_video_{i}")
                if os.path.exists(batch_dir):
                    try:
                        shutil.rmtree(batch_dir)
                    except Exception:
                        pass

    finally:
        # Clean up main temporary directory
        if os.path.exists(main_temp_dir):
            try:
                shutil.rmtree(main_temp_dir)
                print(f"🗑️ Deleted main temporary directory: {main_temp_dir}")
            except Exception as e:
                print(f"⚠️ Could not delete main temporary directory: {e}")

    print(f"\n{'#'*100}")
    print(f"✅ ALL VIDEOS IN THE PLAYLIST HAVE BEEN PROCESSED")
    print(f"📂 Results saved to: {OUTPUT_PATH}")
    print(f"📊 CSV mapping file: {CSV_OUTPUT_FILE}")
    print(f"⏱️ Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*100}")

def main():
    print(f"\n{'*'*100}")
    print(f"🎬 YOUTUBE PLAYLIST PROCESSOR")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*100}")

    # Check if output directory exists, create if not
    if not os.path.exists(OUTPUT_PATH):
        print(f"📂 Creating output directory: {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    else:
        print(f"📂 Using existing output directory: {OUTPUT_PATH}")
    print("FOR PLAYLIST URL ENTER THE ONE GENERATED FROM THE SHARE BUTTON IN THE PLAYLIST PAGE")
    playlist_url = input("🔗 Enter the YouTube playlist URL: ").strip()

    # Get number of parallel workers (default: 4)
    try:
        max_workers = int(input("🧰 Number of parallel workers (default: 4): ").strip() or "4")
        if max_workers < 1:
            print("⚠️ Workers must be at least 1. Setting to 1.")
            max_workers = 1
        elif max_workers > 8:
            print("⚠️ Too many workers may cause issues. Limiting to 8.")
            max_workers = 8
    except ValueError:
        print("⚠️ Invalid input. Using default: 4 workers")
        max_workers = 4

    # Get batch size for playlist fetching (default: 50)
    try:
        batch_size = int(input("📦 Batch size for playlist fetching (default: 50): ").strip() or "50")
        if batch_size < 10:
            print("⚠️ Batch size must be at least 10. Setting to 10.")
            batch_size = 10
        elif batch_size > 200:
            print("⚠️ Large batch sizes may cause issues. Limiting to 200.")
            batch_size = 200
    except ValueError:
        print("⚠️ Invalid input. Using default: 50 videos per batch")
        batch_size = 50

    print(f"\n⚙️ Starting with {max_workers} parallel workers and batch size of {batch_size}")

    try:
        process_playlist_parallel(playlist_url, max_workers, batch_size)
        print("\n✅ Processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")

    print(f"\n{'*'*100}")
    print(f"🎬 YOUTUBE PLAYLIST PROCESSOR - COMPLETE")
    print(f"📅 Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*100}")

if __name__ == "__main__":
    main()