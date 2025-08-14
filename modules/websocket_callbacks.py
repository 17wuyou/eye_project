# modules/websocket_callbacks.py
import uuid, time, json, base64, logging, torchaudio
from typing import Any, Optional
from flask_sock import ConnectionClosed
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from . import state_manager
from . import client_actions
from . import audio_processing_logic # Now uses the wrapper
from . import video_processing_logic
from . import performance_utils
from . import event_recorder
from . import gui_callbacks
from .general_utils import parse_iso_timestamp_to_unix
from . import latency_logger

from encryption_util import decrypt_from_string
from keyframe_detector import KeyframeDetector
from kws_service import KwsService 
from audio_utils import is_audio_active_by_rms # For VAD check before KWS/accumulation
import configs

logger = logging.getLogger(__name__)

def _resample_audio_for_kws(audio_bytes: bytes, original_sr: int, target_sr: int) -> bytes:
    if original_sr == target_sr:
        return audio_bytes
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        # Ensure tensor is float32 for torchaudio resample
        audio_tensor = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        resampled_tensor = resampler(audio_tensor)
        resampled_np = (resampled_tensor.squeeze(0).numpy() * 32768.0).astype(np.int16)
        return resampled_np.tobytes()
    except Exception as e:
        logger.error(f"音频重采样至 {target_sr}Hz 失败: {e}", exc_info=False) # Less verbose for this common op
        return b''


def register_websocket_handlers(
    sock: Any,
    socketio: Any,
    keyframe_executor: Optional[ThreadPoolExecutor],
    face_executor: Optional[ThreadPoolExecutor],
    llm_executor: Optional[ThreadPoolExecutor],
    audio_executor: Optional[ThreadPoolExecutor] # Added audio_executor
):
    @sock.route('/ws')
    def android_ws_handler(ws: Any):
        client_ip = ws.environ.get('REMOTE_ADDR', '未知IP')
        client_port = ws.environ.get('REMOTE_PORT', '未知端口')
        client_uuid_str = str(uuid.uuid4())
        client_desc = f"android_{client_uuid_str}({client_ip}:{client_port})"

        logger.info(f"{client_desc} 正在连接...")

        state_manager.raw_ws_clients[client_uuid_str] = ws
        client_actions.initialize_client_locks(client_uuid_str)
        state_manager.app_audio_save_counter[client_uuid_str] = 0
        client_actions.initialize_client_accumulator(client_uuid_str)

        try:
            if configs.PICOVOICE_ACCESS_KEY and configs.KWS_KEYWORD_PATHS:
                kws_detector_instance = KwsService(
                    access_key=configs.PICOVOICE_ACCESS_KEY,
                    model_path=configs.KWS_MODEL_PATH,
                    keyword_paths=configs.KWS_KEYWORD_PATHS,
                    sensitivities=configs.KWS_KEYWORD_SENSITIVITIES
                )
                state_manager.client_kws_detectors[client_uuid_str] = kws_detector_instance
                state_manager.llm_query_active[client_uuid_str] = False # Initialize LLM query state
                logger.info(f"[{client_desc}] KWS 服务已成功初始化。")
            else:
                 logger.warning(f"[{client_desc}] KWS 配置不完整，KWS功能将不可用。")
                 state_manager.client_kws_detectors[client_uuid_str] = None
        except Exception as e_kws_init:
            logger.error(f"[{client_desc}] KWS 服务初始化失败: {e_kws_init}. KWS功能将不可用。", exc_info=True)
            state_manager.client_kws_detectors[client_uuid_str] = None

        if configs.DATASET_SAVE_EVENTS:
            client_actions.initialize_client_event_data(client_uuid_str)

        try:
            state_manager.keyframe_detectors[client_uuid_str] = KeyframeDetector(
                threshold=configs.KEYFRAME_DETECTOR_THRESHOLD,
                history_size_config=configs.KEYFRAME_HISTORY_SIZE,
                resize_dim=configs.KEYFRAME_RESIZE_DIM,
                motion_threshold_percent=configs.KEYFRAME_MOTION_THRESHOLD_PERCENT,
                motion_diff_threshold_value=configs.KEYFRAME_MOTION_DIFF_THRESHOLD_VALUE,
                cooldown_period=configs.KEYFRAME_COOLDOWN_SECONDS,
                use_h_channel=configs.KEYFRAME_USE_H_CHANNEL_FOR_HIST,
                hist_channels_h=configs.KEYFRAME_HIST_CHANNELS_H, hist_size_h=configs.KEYFRAME_HIST_SIZE_H, hist_ranges_h=configs.KEYFRAME_HIST_RANGES_H,
                hist_channels_gray=configs.KEYFRAME_HIST_CHANNELS_GRAY, hist_size_gray=configs.KEYFRAME_HIST_SIZE_GRAY, hist_ranges_gray=configs.KEYFRAME_HIST_RANGES_GRAY
            )
            state_manager.last_keyframe_check_times[client_uuid_str] = 0.0

            if configs.ENABLE_FACE_SERVICE:
                state_manager.last_face_check_times[client_uuid_str] = 0.0

            logger.info(f"{client_desc} 已连接。当前原始客户端总数: {len(state_manager.raw_ws_clients)}")
            socketio.emit('server_log', {'message': f"{client_desc} 已连接."}, room=state_manager.GUI_LISTENERS_ROOM)
            gui_callbacks.update_gui_client_dropdowns(socketio)
            performance_utils.start_performance_reporting_if_needed(socketio)

            while True:
                trace_id = latency_logger.generate_trace_id()
                server_receive_time_unix = time.time() # Unix timestamp (seconds)
                
                encrypted_data_str = ws.receive(timeout=None) # Blocking call
                if encrypted_data_str is None:
                    logger.info(f"{client_desc} 发送了 None (连接正常关闭或超时)。")
                    break

                decrypt_start_perf = time.perf_counter()
                decrypted_json_str = decrypt_from_string(encrypted_data_str)
                decrypt_duration_ms = (time.perf_counter() - decrypt_start_perf) * 1000
                
                if not decrypted_json_str:
                    logger.error(f"{client_desc} [{trace_id}] 数据解密失败。跳过此消息。")
                    latency_logger.log_event(trace_id, client_uuid_str, 'decryption', 'ERROR', {'reason': 'decryption failed'})
                    continue

                try:
                    data = json.loads(decrypted_json_str)
                    video_frame_b64 = data.get("video_frame")
                    audio_chunk_b64 = data.get("audio_chunk")
                    client_timestamp_str = data.get("timestamp") # This is ISO string or number
                    client_timestamp_unix = parse_iso_timestamp_to_unix(client_timestamp_str) # Convert to Unix timestamp (seconds)
                    
                    initial_latency_ms = (server_receive_time_unix - client_timestamp_unix) * 1000 if client_timestamp_unix else -1
                    latency_logger.log_event(
                        trace_id, client_uuid_str, 'websocket_receive', 'INFO', 
                        {
                            'client_timestamp_orig': client_timestamp_str, 
                            'client_timestamp_unix': f"{client_timestamp_unix:.3f}" if client_timestamp_unix else "N/A",
                            'server_receive_unix': f"{server_receive_time_unix:.3f}",
                            'initial_queue_latency_ms': f'{initial_latency_ms:.2f}', # Latency from client send to server sock read
                            'decryption_ms': f'{decrypt_duration_ms:.2f}'
                        }
                    )

                    # --- Smart Skip Logic ---
                    skip_processing_video = False
                    skip_processing_audio = False

                    if client_timestamp_unix and configs.MAX_INPUT_QUEUE_TIME_MS > 0:
                        if initial_latency_ms > configs.MAX_INPUT_QUEUE_TIME_MS:
                            logger.warning(f"[{client_desc}] [{trace_id}] 输入队列延迟过高 ({initial_latency_ms:.0f}ms > {configs.MAX_INPUT_QUEUE_TIME_MS}ms)。可能跳过部分处理。")
                            skip_processing_video = True # Prioritize skipping video first
                            skip_processing_audio = True # Also skip audio if delay is very high

                    # Check video processing backlog
                    if keyframe_executor and hasattr(keyframe_executor, '_work_queue'): # Check if ThreadPoolExecutor
                        kf_qsize = keyframe_executor._work_queue.qsize()
                        if kf_qsize > configs.MAX_VIDEO_PROCESSING_BACKLOG_FRAMES:
                            logger.warning(f"[{client_desc}] [{trace_id}] 关键帧检测队列积压 ({kf_qsize} > {configs.MAX_VIDEO_PROCESSING_BACKLOG_FRAMES})。跳过当前视频帧处理。")
                            skip_processing_video = True
                            latency_logger.log_event(trace_id, client_uuid_str, 'skip_video_processing', 'WARN', {'reason': 'keyframe_backlog', 'qsize': kf_qsize})


                    if face_executor and hasattr(face_executor, '_work_queue'):
                         face_qsize = face_executor._work_queue.qsize()
                         if face_qsize > configs.MAX_VIDEO_PROCESSING_BACKLOG_FRAMES : # Use same backlog for faces
                            logger.warning(f"[{client_desc}] [{trace_id}] 人脸处理队列积压 ({face_qsize} > {configs.MAX_VIDEO_PROCESSING_BACKLOG_FRAMES})。跳过当前视频帧处理。")
                            skip_processing_video = True # Already true if KF skipped, but good to check
                            latency_logger.log_event(trace_id, client_uuid_str, 'skip_video_processing', 'WARN', {'reason': 'face_backlog', 'qsize': face_qsize})
                    
                    # Check audio processing backlog (more complex to estimate accurately)
                    if audio_executor and hasattr(audio_executor, '_work_queue'):
                        audio_qsize = audio_executor._work_queue.qsize()
                        # Estimate backlog duration: qsize * avg_processing_time_per_item
                        # For simplicity, let's use a fixed number of items as threshold first
                        # A better way would be to track actual processing times.
                        # Using MAX_AUDIO_PROCESSING_BACKLOG_MS as a qsize limit for now, assuming each item is ~200-500ms
                        # This is a very rough estimate.
                        # Assume each audio task might take up to 500ms on average.
                        # So, if MAX_AUDIO_PROCESSING_BACKLOG_MS = 2000ms, then qsize_limit = 2000/500 = 4.
                        estimated_audio_q_limit = max(1, configs.MAX_AUDIO_PROCESSING_BACKLOG_MS // 500)
                        if audio_qsize > estimated_audio_q_limit :
                            logger.warning(f"[{client_desc}] [{trace_id}] 音频处理队列积压 ({audio_qsize} > {estimated_audio_q_limit})。可能跳过当前音频块。")
                            skip_processing_audio = True
                            latency_logger.log_event(trace_id, client_uuid_str, 'skip_audio_processing', 'WARN', {'reason': 'audio_backlog', 'qsize': audio_qsize})
                    
                    # --- End Smart Skip Logic ---


                    if configs.ENABLE_PERFORMANCE_MONITORING and audio_chunk_b64 and not skip_processing_audio: # Only update if audio is processed
                        perf_lock = state_manager.client_data_locks.get(client_uuid_str, {}).get("performance_lock")
                        if perf_lock:
                            with perf_lock:
                                if client_uuid_str in state_manager.client_performance_data:
                                    perf_entry = state_manager.client_performance_data[client_uuid_str]
                                    if perf_entry.get("cycle_server_recv_wall_time") is None: # First chunk of a cycle
                                        perf_entry["cycle_client_timestamp_unix"] = client_timestamp_unix
                                        perf_entry["cycle_server_recv_wall_time"] = server_receive_time_unix
                                    perf_entry["cycle_chunk_count"] = perf_entry.get("cycle_chunk_count",0) + 1
                                    if client_timestamp_unix and server_receive_time_unix:
                                        perf_entry["latency_client_to_server_recv_ms"] = initial_latency_ms

                    if video_frame_b64:
                        state_manager.last_video_frames[client_uuid_str] = video_frame_b64 # Cache frame even if skipped for LLM

                        socketio.emit('decoded_video_frame', {
                            'sid': f"android_{client_uuid_str}",
                            'frame': video_frame_b64,
                            'timestamp': client_timestamp_str # Send original client timestamp string
                        }, room=state_manager.GUI_LISTENERS_ROOM)

                        if not skip_processing_video:
                            current_frame_time_perf = time.perf_counter() # Use perf_counter for intervals
                            video_frame_bytes = None # Decode only if needed

                            keyframe_check_interval_sec = getattr(configs, 'KEYFRAME_CHECK_INTERVAL_SECONDS', 0.1) # Check if this config exists
                            # last_keyframe_check_times stores perf_counter time
                            if (current_frame_time_perf - state_manager.last_keyframe_check_times.get(client_uuid_str, 0.0)) >= keyframe_check_interval_sec:
                                state_manager.last_keyframe_check_times[client_uuid_str] = current_frame_time_perf
                                try:
                                    if video_frame_bytes is None: video_frame_bytes = base64.b64decode(video_frame_b64)
                                    detector = state_manager.keyframe_detectors.get(client_uuid_str)
                                    if detector and keyframe_executor and not keyframe_executor._shutdown:
                                        keyframe_executor.submit(
                                            video_processing_logic._run_keyframe_detection_task,
                                            socketio, detector, video_frame_bytes, client_uuid_str, client_desc, trace_id
                                        )
                                    elif detector and not keyframe_executor: # Synchronous fallback
                                         video_processing_logic._run_keyframe_detection_task(socketio, detector, video_frame_bytes, client_uuid_str, client_desc, trace_id)

                                except Exception as ekf_submit:
                                    logger.error(f"[{client_desc}] [{trace_id}] 提交/执行关键帧检测任务时出错: {ekf_submit}", exc_info=True)

                            if configs.ENABLE_FACE_SERVICE and face_executor and not face_executor._shutdown:
                                if (current_frame_time_perf - state_manager.last_face_check_times.get(client_uuid_str, 0.0)) >= configs.FACE_PROCESSING_INTERVAL_SECONDS:
                                    state_manager.last_face_check_times[client_uuid_str] = current_frame_time_perf
                                    try:
                                        if video_frame_bytes is None: video_frame_bytes = base64.b64decode(video_frame_b64)
                                        face_executor.submit(
                                            video_processing_logic._run_face_processing_task,
                                            socketio, client_uuid_str, client_desc, video_frame_bytes, trace_id
                                        )
                                    except Exception as e_face_submit:
                                        logger.error(f"[{client_desc}] [{trace_id}] 提交人脸处理任务时出错: {e_face_submit}", exc_info=True)
                            elif configs.ENABLE_FACE_SERVICE and not face_executor:
                                logger.warning(f"[{client_desc}] [{trace_id}] 人脸服务已启用但处理执行器未配置或已关闭。")
                        else: # Video processing skipped
                            latency_logger.log_event(trace_id, client_uuid_str, 'video_processing', 'SKIP', {'reason': 'smart_skip_logic'})


                    if audio_chunk_b64 and not skip_processing_audio:
                        try:
                            audio_chunk_bytes = base64.b64decode(audio_chunk_b64)
                            acc_state = state_manager.client_audio_accumulators.get(client_uuid_str)
                            if not acc_state: # Should not happen if initialized correctly
                                logger.warning(f"[{client_desc}] [{trace_id}] 音频累积器未找到，重新初始化。")
                                client_actions.initialize_client_accumulator(client_uuid_str)
                                acc_state = state_manager.client_audio_accumulators[client_uuid_str]

                            vad_start_perf = time.perf_counter()
                            is_active_chunk = is_audio_active_by_rms(audio_chunk_bytes, sample_width=configs.AUDIO_SAMPLE_WIDTH)
                            vad_duration_ms = (time.perf_counter() - vad_start_perf) * 1000
                            latency_logger.log_event(trace_id, client_uuid_str, 'vad_check', 'END', {'duration_ms': f'{vad_duration_ms:.2f}', 'is_active': is_active_chunk})

                            if configs.ENABLE_PERFORMANCE_MONITORING:
                                # ... (performance lock and update for VAD duration)
                                perf_lock_audio = state_manager.client_data_locks.get(client_uuid_str, {}).get("performance_lock")
                                if perf_lock_audio:
                                    with perf_lock_audio:
                                        if client_uuid_str in state_manager.client_performance_data:
                                            perf_entry_audio = state_manager.client_performance_data[client_uuid_str]
                                            perf_entry_audio["cycle_accumulated_vad_duration_ms"] = \
                                                perf_entry_audio.get("cycle_accumulated_vad_duration_ms", 0.0) + vad_duration_ms

                            if is_active_chunk:
                                kws_detector_instance = state_manager.client_kws_detectors.get(client_uuid_str)
                                # Check KWS only if not already triggered in this dialogue and LLM not active
                                if kws_detector_instance and not acc_state.get('kws_was_triggered') and not state_manager.llm_query_active.get(client_uuid_str):
                                    kws_start_perf = time.perf_counter()
                                    resampled_chunk_for_kws = _resample_audio_for_kws(
                                        audio_chunk_bytes, 
                                        configs.AUDIO_SAMPLE_RATE, 
                                        kws_detector_instance.sample_rate
                                    )
                                    kws_triggered_flag = kws_detector_instance.process(resampled_chunk_for_kws) if resampled_chunk_for_kws else False
                                    kws_duration_ms = (time.perf_counter() - kws_start_perf) * 1000
                                    latency_logger.log_event(trace_id, client_uuid_str, 'kws_check', 'END', {'duration_ms': f'{kws_duration_ms:.2f}', 'triggered': kws_triggered_flag})

                                    if kws_triggered_flag:
                                        logger.info(f"[{client_desc}] [{trace_id}] KWS 关键词触发！")
                                        acc_state['kws_was_triggered'] = True # Mark KWS triggered for this dialogue

                                # Dialogue accumulation logic
                                if not acc_state['vad_is_active']: # Start of speech segment
                                    acc_state['vad_is_active'] = True
                                    if not acc_state['dialogue_buffer']: # Very first chunk of a new dialogue
                                        acc_state['dialogue_start_client_timestamp'] = client_timestamp_unix
                                
                                acc_state['dialogue_buffer'].append(audio_chunk_bytes)
                                acc_state['dialogue_accumulated_chunks_count'] += 1
                                acc_state['dialogue_last_speech_time_ns'] = time.monotonic_ns()
                                acc_state['vad_consecutive_silent_chunks'] = 0
                            elif acc_state['vad_is_active']: # Was active, now silent chunk
                                acc_state['vad_consecutive_silent_chunks'] += 1
                                if acc_state['vad_consecutive_silent_chunks'] <= configs.SHORT_SILENCE_PADDING_CHUNKS:
                                    acc_state['dialogue_buffer'].append(audio_chunk_bytes) # Pad with a bit of silence
                                    acc_state['dialogue_accumulated_chunks_count'] += 1
                                else: # Too much silence, VAD considers speech ended
                                    acc_state['vad_is_active'] = False
                            
                            # Call the wrapper which handles submitting to audio_executor
                            audio_processing_logic.process_accumulated_audio_wrapper(
                                client_uuid_str, socketio, llm_executor, audio_executor
                            )

                        except base64.binascii.Error as e_b64_aud:
                            logger.error(f"[{client_desc}] [{trace_id}] 音频块Base64解码失败: {e_b64_aud}")
                        except Exception as e_aud_proc:
                            logger.error(f"[{client_desc}] [{trace_id}] 音频块处理错误: {e_aud_proc}", exc_info=True)
                    elif audio_chunk_b64 and skip_processing_audio:
                        latency_logger.log_event(trace_id, client_uuid_str, 'audio_processing', 'SKIP', {'reason': 'smart_skip_logic'})
                        # If audio is skipped, we still need to potentially trigger process_accumulated_audio
                        # if there's old data and a timeout occurs.
                        # However, the current chunk is NOT added.
                        # The existing timeout logic in process_accumulated_audio_wrapper should handle this.
                        # We might want to update dialogue_last_speech_time_ns if VAD was active to not immediately timeout.
                        acc_state = state_manager.client_audio_accumulators.get(client_uuid_str)
                        if acc_state and acc_state['vad_is_active']:
                             # If VAD was active, but we are skipping this silent chunk due to load,
                             # reset consecutive silent chunks but don't mark VAD inactive yet.
                             # This prevents premature dialogue termination if the system recovers.
                             # This is a nuanced part of skip logic. For now, let's keep it simple:
                             # if skipped, it's as if the chunk didn't arrive for accumulation.
                             pass
                        audio_processing_logic.process_accumulated_audio_wrapper(
                                client_uuid_str, socketio, llm_executor, audio_executor
                            )


                except json.JSONDecodeError:
                    logger.error(f"{client_desc} [{trace_id}] 发送的数据不是有效的JSON: '{decrypted_json_str[:100]}...'")
                except Exception as e_proc_data:
                    logger.error(f"{client_desc} [{trace_id}] 内部数据处理错误: {e_proc_data}", exc_info=True)

        except ConnectionClosed:
            logger.info(f"{client_desc} 连接已关闭 (ConnectionClosed)。")
        except Exception as e_ws_handler_main:
            logger.critical(f"WebSocket处理器 for {client_desc} 发生严重错误: {e_ws_handler_main}", exc_info=True)
        finally:
            logger.info(f"正在为 {client_desc} 进行主清理...")

            # Force process any remaining audio in the accumulator
            audio_processing_logic.process_accumulated_audio_wrapper(
                client_uuid_str, socketio, llm_executor, audio_executor, force_process=True
            )
            event_recorder.save_event_data_on_disconnect(client_uuid_str) # Save any pending event

            if client_uuid_str in state_manager.raw_ws_clients:
                del state_manager.raw_ws_clients[client_uuid_str]

            client_actions.cleanup_client_resources(client_uuid_str, client_desc)

            logger.info(f"{client_desc} 已断开连接并清理完毕。当前客户端总数: {len(state_manager.raw_ws_clients)}")
            socketio.emit('server_log', {'message': f"{client_desc} 已断开连接."}, room=state_manager.GUI_LISTENERS_ROOM)
            gui_callbacks.update_gui_client_dropdowns(socketio)

            if not state_manager.raw_ws_clients: # If no clients left
                performance_utils.stop_performance_reporting()